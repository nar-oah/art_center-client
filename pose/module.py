import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import einops
from einops import rearrange, repeat
from timm.models.vision_transformer import Block
from timm.layers.patch_embed import PatchEmbed
from timm.layers.weight_init import trunc_normal_


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


class ViT(nn.Module):
    def __init__(
        self,
        img_size=(256, 192),
        patch_size=16,
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        task_tokens_num=80,
        **_,
    ):
        super().__init__()
        self.task_tokens_num = task_tokens_num
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.task_tokens = nn.Parameter(torch.zeros(1, task_tokens_num, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        trunc_normal_(self.task_tokens, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.last_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        x_embed = self.patch_embed(x)
        Hp, Wp = self.patch_embed.grid_size

        if self.pos_embed is not None:
            x_embed = x_embed + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        task_tokens = repeat(self.task_tokens, "() n d -> b n d", b=B)
        x_concat = torch.cat((task_tokens, x_embed), dim=1)

        for blk in self.blocks:
            x_concat = blk(x_concat)

        x_norm = self.last_norm(x_concat)

        task_tokens_out = x_norm[:, : self.task_tokens_num]
        xp = x_norm[:, self.task_tokens_num :]

        xp_reshaped = xp.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp_reshaped, task_tokens_out


class TransformerDecoderHead(nn.Module):
    def __init__(
        self, feat_dim: int = 1280, dim_out: int = 512, task_tokens_num: int = 80
    ) -> None:
        super().__init__()
        self.dim = feat_dim
        self.dim_out = dim_out
        self.token_dim = task_tokens_num

        HAND_JOINT_NUM, BODY_JOINT_NUM, SHAPE_NUM, EXPRESSION_NUM = 15, 22, 10, 10

        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=self.token_dim,
            dim=self.dim,
            depth=6,
            heads=8,
            mlp_dim=self.dim,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            context_dim=self.dim,
        )

        self.token_conv = nn.Linear(self.dim, self.dim_out)

        self.dec_body_root_pose = nn.Linear(1 * self.dim_out, 6)
        self.dec_body_pose = nn.Linear(
            (BODY_JOINT_NUM - 1) * self.dim_out, (BODY_JOINT_NUM - 1) * 6
        )
        self.dec_body_shape = nn.Linear(SHAPE_NUM * self.dim_out, SHAPE_NUM)
        self.dec_body_cam = nn.Linear(1 * self.dim_out, 3)

        self.dec_hand_root_pose = nn.Linear(2 * self.dim_out, 2 * 6)
        self.dec_hand_pose = nn.Linear(
            2 * HAND_JOINT_NUM * self.dim_out, 2 * HAND_JOINT_NUM * 6
        )
        self.dec_hand_cam = nn.Linear(2 * self.dim_out, 2 * 3)

        self.dec_face_root_pose = nn.Linear(1 * self.dim_out, 6)
        self.dec_face_expression = nn.Linear(
            EXPRESSION_NUM * self.dim_out, EXPRESSION_NUM
        )
        self.dec_face_jaw_pose = nn.Linear(1 * self.dim_out, 6)
        self.dec_face_cam = nn.Linear(1 * self.dim_out, 3)

    def forward(self, token: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        token = torch.cat((token, x), dim=1)

        token_out = self.transformer(token, context=x)
        token_out = self.token_conv(token_out)[:, : self.token_dim, :]

        pred_params = {
            "body_root_pose": self.dec_body_root_pose(
                token_out[:, :1, :].reshape(batch_size, -1)
            ),
            "body_pose": self.dec_body_pose(
                token_out[:, 1:22, :].reshape(batch_size, -1)
            ),
            "body_betas": self.dec_body_shape(
                token_out[:, 22:32, :].reshape(batch_size, -1)
            ),
            "body_cam": self.dec_body_cam(
                token_out[:, 32:33, :].reshape(batch_size, -1)
            ),
        }

        pred_hand_root = self.dec_hand_root_pose(
            token_out[:, 33:35, :].reshape(batch_size, -1)
        )
        pred_hand_pose = self.dec_hand_pose(
            token_out[:, 35:65, :].reshape(batch_size, -1)
        )
        pred_hand_cam = self.dec_hand_cam(
            token_out[:, 65:67, :].reshape(batch_size, -1)
        )

        pred_params.update(
            {
                "lhand_root_pose": pred_hand_root[:, :6],
                "rhand_root_pose": pred_hand_root[:, 6:],
                "lhand_pose": pred_hand_pose[:, :90],
                "rhand_pose": pred_hand_pose[:, 90:],
                "lhand_cam": pred_hand_cam[:, :3],
                "rhand_cam": pred_hand_cam[:, 3:],
                "face_root_pose": self.dec_face_root_pose(
                    token_out[:, 67:68, :].reshape(batch_size, -1)
                ),
                "face_expression": self.dec_face_expression(
                    token_out[:, 68:78, :].reshape(batch_size, -1)
                ),
                "face_jaw_pose": self.dec_face_jaw_pose(
                    token_out[:, 78:79, :].reshape(batch_size, -1)
                ),
                "face_cam": self.dec_face_cam(
                    token_out[:, 79:80, :].reshape(batch_size, -1)
                ),
            }
        )

        return pred_params


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
        context_dim: Optional[int] = None,
        **_,
    ) -> None:
        super().__init__()
        self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerCrossAttn(
            dim, depth, heads, dim_head, mlp_dim, dropout, context_dim=context_dim
        )

    def forward(
        self, x: torch.Tensor, *args: Any, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, n, _ = x.shape
        x = self.dropout(x)
        x = x + self.pos_embedding[:, :n]
        return self.transformer(x, *args, context=context)


class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
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
                nn.ModuleList([PreNorm(dim, sa), PreNorm(dim, ca), PreNorm(dim, ff)])
            )

    def forward(
        self, x: torch.Tensor, *args: Any, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context) + x
            x = ff(x, *args) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *_: Any, **kwargs: Any) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads, self.scale = heads, dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.to_out(out)


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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads, self.scale = heads, dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_kv = nn.Linear(default(context_dim, dim), inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        k_chunk, v_chunk = self.to_kv(default(context, x)).chunk(2, dim=-1)
        q_tensor = self.to_q(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            [q_tensor, k_chunk, v_chunk],
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.to_out(out)
