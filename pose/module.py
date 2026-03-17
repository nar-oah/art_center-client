import torch
import torch.nn as nn
from inspect import isfunction
import copy
import einops
from einops import rearrange, repeat
from typing import Callable, Optional, List, Tuple, Dict, Any, Union

# 修正后的 timm 引入方式
from timm.layers.drop import drop_path
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_
import torch.utils.checkpoint as checkpoint


# ==========================================
# 辅助函数
# ==========================================
def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ==========================================
# Transformer 解码器核心 (彻底硬编码剥离 SMPLX)
# ==========================================
class TransformerDecoderHead(nn.Module):
    """Cross-attention based Transformer decoder"""

    def __init__(
        self, feat_dim: int = 1080, dim_out: int = 512, task_tokens_num: int = 80
    ) -> None:
        super().__init__()

        self.dim: int = feat_dim
        self.dim_out: int = dim_out
        self.token_dim: int = task_tokens_num

        # 彻底移除 smpl_x = SMPLX()，直接使用学术界常量！
        HAND_JOINT_NUM: int = 15
        BODY_JOINT_NUM: int = 22
        SHAPE_NUM: int = 10
        EXPRESSION_NUM: int = 10

        transformer_args: Dict[str, Any] = dict(
            num_tokens=1,
            token_dim=self.token_dim,
            dim=self.dim,
            depth=6,
            heads=8,
            mlp_dim=self.dim,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=self.dim,
        )
        self.transformer: TransformerDecoder = TransformerDecoder(**transformer_args)

        # [b, token_dim, dim] -> [b, token_dim, dim_out]
        self.token_conv: nn.Linear = nn.Linear(self.dim, self.dim_out)

        # heads - body
        self.dec_body_root_pose: nn.Linear = nn.Linear(1 * self.dim_out, 6)  # 1 [b, 6]
        self.dec_body_pose: nn.Linear = nn.Linear(
            (BODY_JOINT_NUM - 1) * self.dim_out, (BODY_JOINT_NUM - 1) * 6
        )  # 21 [b, 21 ,6]
        self.dec_body_shape: nn.Linear = nn.Linear(
            SHAPE_NUM * self.dim_out, SHAPE_NUM
        )  # 10 [b, 10]
        self.dec_body_cam: nn.Linear = nn.Linear(1 * self.dim_out, 3)  # 1 [b, 3]

        # heads - left and right hand
        self.dec_hand_root_pose: nn.Linear = nn.Linear(
            2 * self.dim_out, 2 * 6
        )  # 2 [b, 2, 6]
        self.dec_hand_pose: nn.Linear = nn.Linear(
            2 * HAND_JOINT_NUM * self.dim_out, 2 * HAND_JOINT_NUM * 6
        )  # 30 [b, 30, 6]
        self.dec_hand_cam: nn.Linear = nn.Linear(2 * self.dim_out, 2 * 3)  # 2 [b, 2, 3]

        # heads - face
        self.dec_face_root_pose: nn.Linear = nn.Linear(1 * self.dim_out, 6)  # 1 [b, 6]
        self.dec_face_expression: nn.Linear = nn.Linear(
            EXPRESSION_NUM * self.dim_out, EXPRESSION_NUM
        )  # 10 [b, 10]
        self.dec_face_jaw_pose: nn.Linear = nn.Linear(1 * self.dim_out, 6)  # 1 [b, 6]
        self.dec_face_cam: nn.Linear = nn.Linear(1 * self.dim_out, 3)  # 1 [b, 3]

    def forward(
        self, token: torch.Tensor, x: torch.Tensor, **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        batch_size: int = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        token = torch.cat((token, x), dim=1)  # Concatenated input to decoder

        # Pass through transformer
        token_out: torch.Tensor = self.transformer(token, context=x)
        token_out = self.token_conv(token_out)
        token_out = token_out[:, : self.token_dim, :]  # (B, C)

        # Readout from token_out
        token_body_root: torch.Tensor = token_out[:, :1, :].reshape(batch_size, -1)
        token_body_pose: torch.Tensor = token_out[:, 1:22, :].reshape(batch_size, -1)
        token_body_shape: torch.Tensor = token_out[:, 22:32, :].reshape(batch_size, -1)
        token_body_cam: torch.Tensor = token_out[:, 32:33, :].reshape(batch_size, -1)

        token_hand_root: torch.Tensor = token_out[:, 33:35, :].reshape(batch_size, -1)
        token_hand_pose: torch.Tensor = token_out[:, 35:65, :].reshape(batch_size, -1)
        token_hand_cam: torch.Tensor = token_out[:, 65:67, :].reshape(batch_size, -1)

        token_face_root: torch.Tensor = token_out[:, 67:68, :].reshape(batch_size, -1)
        token_face_expression: torch.Tensor = token_out[:, 68:78, :].reshape(
            batch_size, -1
        )
        token_face_jaw: torch.Tensor = token_out[:, 78:79, :].reshape(batch_size, -1)
        token_face_cam: torch.Tensor = token_out[:, 79:80, :].reshape(batch_size, -1)

        # Decode
        pred_body_root_pose: torch.Tensor = self.dec_body_root_pose(token_body_root)
        pred_body_pose: torch.Tensor = self.dec_body_pose(token_body_pose)
        pred_body_betas: torch.Tensor = self.dec_body_shape(token_body_shape)
        pred_body_cam: torch.Tensor = self.dec_body_cam(token_body_cam)

        pred_hand_root_pose: torch.Tensor = self.dec_hand_root_pose(token_hand_root)
        pred_hand_pose: torch.Tensor = self.dec_hand_pose(token_hand_pose)
        pred_hand_cam: torch.Tensor = self.dec_hand_cam(token_hand_cam)

        pred_face_root_pose: torch.Tensor = self.dec_face_root_pose(token_face_root)
        pred_face_expression: torch.Tensor = self.dec_face_expression(
            token_face_expression
        )
        pred_face_jaw_pose: torch.Tensor = self.dec_face_jaw_pose(token_face_jaw)
        pred_face_cam: torch.Tensor = self.dec_face_cam(token_face_cam)

        # all rotations in rot6d
        pred_params: Dict[str, torch.Tensor] = {
            "body_root_pose": pred_body_root_pose,
            "body_pose": pred_body_pose,
            "body_betas": pred_body_betas,
            "body_cam": pred_body_cam,
            "lhand_root_pose": pred_hand_root_pose[:, :6],
            "rhand_root_pose": pred_hand_root_pose[:, 6:],
            "lhand_pose": pred_hand_pose[:, :90],
            "rhand_pose": pred_hand_pose[:, 90:],
            "lhand_cam": pred_hand_cam[:, :3],
            "rhand_cam": pred_hand_cam[:, 3:],
            "face_root_pose": pred_face_root_pose,
            "face_expression": pred_face_expression,
            "face_jaw_pose": pred_face_jaw_pose,
            "face_cam": pred_face_cam,
        }

        return pred_params


# ==========================================
# 基础 Transformer 模块
# ==========================================
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ) -> None:
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding: nn.Module = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding: nn.Parameter = nn.Parameter(torch.randn(1, num_tokens, dim))

        self.dropout: nn.Module
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer: TransformerCrossAttn = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        context: Optional[torch.Tensor] = None,
        context_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        x = self.dropout(x)
        x = x + self.pos_embedding[:, :n]
        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x


class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        context: Optional[torch.Tensor] = None,
        context_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if context_list is None:
            context_list = [context] * len(self.layers)  # type: ignore
        if len(context_list) != len(self.layers):
            raise ValueError(
                f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})"
            )

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):  # type: ignore
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


class PreNorm(nn.Module):
    def __init__(
        self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1
    ) -> None:
        super().__init__()
        self.norm: nn.Module = normalization_layer(norm, dim, norm_cond_dim)
        self.fn: Callable = fn

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim: int = dim_head * heads
        project_out: bool = not (heads == 1 and dim_head == dim)

        self.heads: int = heads
        self.scale: float = dim_head**-0.5

        self.attend: nn.Softmax = nn.Softmax(dim=-1)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.to_qkv: nn.Linear = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out: nn.Module = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv: Tuple[torch.Tensor, ...] = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn: torch.Tensor = self.attend(dots)
        attn = self.dropout(attn)

        out: torch.Tensor = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p: float = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            zero_mask: torch.Tensor = (
                torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            )
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p: float = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            zero_mask: torch.Tensor = (
                torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            )
            x[zero_mask, :] = 0
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim: int = dim_head * heads
        project_out: bool = not (heads == 1 and dim_head == dim)

        self.heads: int = heads
        self.scale: float = dim_head**-0.5

        self.attend: nn.Softmax = nn.Softmax(dim=-1)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv: nn.Linear = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q: nn.Linear = nn.Linear(dim, inner_dim, bias=False)

        self.to_out: nn.Module = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        context = default(context, x)
        k_chunk, v_chunk = self.to_kv(context).chunk(2, dim=-1)
        q_tensor = self.to_q(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            [q_tensor, k_chunk, v_chunk],
        )

        dots: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn: torch.Tensor = self.attend(dots)
        attn = self.dropout(attn)

        out: torch.Tensor = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class AdaptiveLayerNorm1D(torch.nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int) -> None:
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm: nn.LayerNorm = torch.nn.LayerNorm(data_dim)
        self.linear: nn.Linear = torch.nn.Linear(norm_cond_dim, 2 * data_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])
        return x * (1 + alpha) + beta


class SequentialCond(torch.nn.Sequential):
    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        for module in self:  # type: ignore
            if isinstance(
                module, (AdaptiveLayerNorm1D, SequentialCond, ResidualMLPBlock)
            ):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input


def normalization_layer(
    norm: Optional[str], dim: int, norm_cond_dim: int = -1
) -> nn.Module:
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer":
        return torch.nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


def linear_norm_activ_dropout(
    input_dim: int,
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers: List[nn.Module] = []
    layers.append(torch.nn.Linear(input_dim, output_dim, bias=bias))
    if norm is not None:
        layers.append(normalization_layer(norm, output_dim, norm_cond_dim))
    layers.append(copy.deepcopy(activation))
    if dropout > 0.0:
        layers.append(torch.nn.Dropout(dropout))
    return SequentialCond(*layers)


def create_simple_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers: List[nn.Module] = []
    prev_dim: int = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            linear_norm_activ_dropout(
                prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            )
        )
        prev_dim = hidden_dim
    layers.append(torch.nn.Linear(prev_dim, output_dim, bias=bias))
    return SequentialCond(*layers)


class ResidualMLPBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",
        dropout: float = 0.0,
        norm_cond_dim: int = -1,
    ) -> None:
        super().__init__()
        if not (input_dim == output_dim == hidden_dim):
            raise NotImplementedError(
                f"input_dim {input_dim} != output_dim {output_dim} is not implemented"
            )

        layers: List[nn.Module] = []
        prev_dim: int = input_dim
        for i in range(num_hidden_layers):
            layers.append(
                linear_norm_activ_dropout(
                    prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
                )
            )
            prev_dim = hidden_dim
        self.model: SequentialCond = SequentialCond(*layers)
        self.skip: nn.Module = torch.nn.Identity()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x + self.model(x, *args, **kwargs)


class ResidualMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",
        dropout: float = 0.0,
        num_blocks: int = 1,
        norm_cond_dim: int = -1,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        block_list: List[nn.Module] = [
            ResidualMLPBlock(
                hidden_dim,
                hidden_dim,
                num_hidden_layers,
                hidden_dim,
                activation,
                bias,
                norm,
                dropout,
                norm_cond_dim,
            )
            for _ in range(num_blocks)
        ]
        self.model: SequentialCond = SequentialCond(
            linear_norm_activ_dropout(
                input_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            ),
            *block_list,
            torch.nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(x, *args, **kwargs)


class FrequencyEmbedder(torch.nn.Module):
    def __init__(self, num_frequencies: int, max_freq_log2: float) -> None:
        super().__init__()
        frequencies: torch.Tensor = 2 ** torch.linspace(
            0, max_freq_log2, steps=num_frequencies
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N: int = x.size(0)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_unsqueezed: torch.Tensor = x.unsqueeze(-1)
        # Type ignored for self.frequencies as register_buffer creates attribute
        scaled: torch.Tensor = self.frequencies.view(1, 1, -1) * x_unsqueezed  # type: ignore
        s: torch.Tensor = torch.sin(scaled)
        c: torch.Tensor = torch.cos(scaled)
        embedded: torch.Tensor = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)
        return embedded


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding"""

    def __init__(
        self,
        backbone: nn.Module,
        img_size: int = 224,
        feature_size: Optional[Tuple[int, int]] = None,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size_tuple: Tuple[int, int] = to_2tuple(img_size)
        self.img_size: Tuple[int, int] = img_size_tuple
        self.backbone: nn.Module = backbone

        feature_dim: int
        if feature_size is None:
            with torch.no_grad():
                training: bool = backbone.training
                if training:
                    backbone.eval()
                o: torch.Tensor = self.backbone(
                    torch.zeros(1, in_chans, img_size_tuple[0], img_size_tuple[1])
                )[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            # 假设 backbone 具有 feature_info
            feature_dim = self.backbone.feature_info.channels()[-1]  # type: ignore

        self.num_patches: int = feature_size[0] * feature_size[1]
        self.proj: nn.Linear = nn.Linear(feature_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat: torch.Tensor = self.backbone(x)[-1]
        x_feat = x_feat.flatten(2).transpose(1, 2)
        return self.proj(x_feat)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        ratio: int = 1,
    ) -> None:
        super().__init__()
        img_size_tuple: Tuple[int, int] = to_2tuple(img_size)
        patch_size_tuple: Tuple[int, int] = to_2tuple(patch_size)
        num_patches: int = (
            (img_size_tuple[1] // patch_size_tuple[1])
            * (img_size_tuple[0] // patch_size_tuple[0])
            * (ratio**2)
        )
        self.patch_shape: Tuple[int, int] = (
            int(img_size_tuple[0] // patch_size_tuple[0] * ratio),
            int(img_size_tuple[1] // patch_size_tuple[1] * ratio),
        )
        self.origin_patch_shape: Tuple[int, int] = (
            int(img_size_tuple[0] // patch_size_tuple[0]),
            int(img_size_tuple[1] // patch_size_tuple[1]),
        )
        self.img_size: Tuple[int, int] = img_size_tuple
        self.patch_size: Tuple[int, int] = patch_size_tuple
        self.num_patches: int = num_patches

        self.proj: nn.Conv2d = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size_tuple,
            stride=(patch_size_tuple[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        x_proj: torch.Tensor = self.proj(x)
        Hp: int = x_proj.shape[2]
        Wp: int = x_proj.shape[3]
        x_out: torch.Tensor = x_proj.flatten(2).transpose(1, 2)
        return x_out, (Hp, Wp)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1: nn.Module = norm_layer(dim)
        self.attn: Attention_ViT = Attention_ViT(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path: nn.Module = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2: nn.Module = norm_layer(dim)
        mlp_hidden_dim: int = int(dim * mlp_ratio)
        self.mlp: Mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1: nn.Linear = nn.Linear(in_features, hidden_features)
        self.act: nn.Module = act_layer()
        self.fc2: nn.Linear = nn.Linear(hidden_features, out_features)
        self.drop: nn.Dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob: Optional[float] = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Attention_ViT(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        head_dim: int = dim // num_heads
        self.dim: int = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim: int = head_dim * self.num_heads

        self.scale: float = qk_scale or head_dim**-0.5

        self.qkv: nn.Linear = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop: nn.Dropout = nn.Dropout(attn_drop)
        self.proj: nn.Linear = nn.Linear(all_head_dim, dim)
        self.proj_drop: nn.Dropout = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q: torch.Tensor = qkv[0]
        k: torch.Tensor = qkv[1]
        v: torch.Tensor = qkv[2]

        q = q * self.scale
        attn: torch.Tensor = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_out: torch.Tensor = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class ViT(torch.nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 80,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        hybrid_backbone: Optional[nn.Module] = None,
        norm_layer: Optional[Callable] = None,
        use_checkpoint: bool = False,
        frozen_stages: int = -1,
        ratio: int = 1,
        last_norm: bool = True,
        patch_padding: str = "pad",
        freeze_attn: bool = False,
        freeze_ffn: bool = False,
        task_tokens_num: int = 80,
    ) -> None:
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes: int = num_classes
        self.num_features: int = embed_dim
        self.embed_dim: int = embed_dim
        self.frozen_stages: int = frozen_stages
        self.use_checkpoint: bool = use_checkpoint
        self.patch_padding: str = patch_padding
        self.freeze_attn: bool = freeze_attn
        self.freeze_ffn: bool = freeze_ffn
        self.depth: int = depth
        self.task_tokens_num: int = task_tokens_num

        if hybrid_backbone is not None:
            self.patch_embed: nn.Module = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                ratio=ratio,
            )

        # 使用 type: ignore 忽略动态属性获取警告
        num_patches: int = self.patch_embed.num_patches  # type: ignore

        self.task_tokens: nn.Parameter = nn.Parameter(
            torch.zeros(1, task_tokens_num, embed_dim)
        )
        trunc_normal_(self.task_tokens, std=0.02)

        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        dpr: List[float] = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.last_norm: nn.Module = (
            norm_layer(embed_dim) if last_norm else nn.Identity()
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()  # type: ignore
                m.norm1.eval()  # type: ignore
                for param in m.attn.parameters():  # type: ignore
                    param.requires_grad = False
                for param in m.norm1.parameters():  # type: ignore
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()  # type: ignore
                m.norm2.eval()  # type: ignore
                for param in m.mlp.parameters():  # type: ignore
                    param.requires_grad = False
                for param in m.norm2.parameters():  # type: ignore
                    param.requires_grad = False

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {"pos_embed", "cls_token"}

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x_embed, (Hp, Wp) = self.patch_embed(x)  # type: ignore
        task_tokens: torch.Tensor = repeat(self.task_tokens, "() n d -> b n d", b=B)

        if self.pos_embed is not None:
            x_embed = x_embed + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        x_concat: torch.Tensor = torch.cat((task_tokens, x_embed), dim=1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x_concat = checkpoint.checkpoint(blk, x_concat)
            else:
                x_concat = blk(x_concat)

        x_norm: torch.Tensor = self.last_norm(x_concat)

        task_tokens_out: torch.Tensor = x_norm[:, : self.task_tokens_num]
        xp: torch.Tensor = x_norm[:, self.task_tokens_num :]

        xp_reshaped: torch.Tensor = (
            xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        )
        return xp_reshaped, task_tokens_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_features(x)

    def train(self, mode: bool = True) -> "ViT":
        super().train(mode)
        self._freeze_stages()
        return self
