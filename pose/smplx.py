import torch
import torch.nn as nn
from typing import Dict
from pose.module import ViT, TransformerDecoderHead

ENCODER: dict = {
    "img_size": (256, 192),
    "patch_size": 16,
    "embed_dim": 1280,
    "depth": 32,
    "num_heads": 16,
}
DECODER: dict = {"feat_dim": 1280, "dim_out": 512, "task_tokens_num": 80}


class PureSMPLestX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ViT(**ENCODER)
        self.decoder = TransformerDecoderHead(**DECODER)

    def forward(self, img_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        img_feat, task_tokens = self.encoder(img_tensor)
        pred_params: Dict[str, torch.Tensor] = self.decoder(task_tokens, img_feat)
        return pred_params
