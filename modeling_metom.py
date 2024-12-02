# This file is a modified version of the Vision Transformer - Pytorch implementation
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from typing import List, Union, Tuple

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from torch import nn
from transformers import PreTrainedModel

from .configuration_metom import MetomConfig


try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_2_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_2_AVAILABLE = False


def size_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MetomFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MetomAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MetomSdpaAttention(MetomAttention):
    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h = self.heads), qkv)
        out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MetomFlashAttention2(MetomAttention):
    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h = self.heads), qkv)
        out = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MetomTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, _attn_implementation = "eager"):
        super().__init__()
        if _attn_implementation == "flash_attention_2":
            assert FLASH_ATTENTION_2_AVAILABLE, "FlashAttention-2 is not available. Please install `flash-attn`."
        attn_cls = (
            MetomAttention if _attn_implementation == "eager" else
            MetomSdpaAttention if _attn_implementation == "sdpa" else
            MetomFlashAttention2 if _attn_implementation == "flash_attention_2" else
            MetomAttention
        )
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                attn_cls(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                MetomFeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MetomModel(PreTrainedModel):
    config_class = MetomConfig
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: MetomConfig):
        super().__init__(config)
        image_height, image_width = size_pair(config.image_size)
        patch_height, patch_width = size_pair(config.patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.channels * patch_height * patch_width
        assert config.pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"
        assert len(config.labels) > 0, "labels must be composed of at least one label"
        assert config._attn_implementation in {"eager", "sdpa", "flash_attention_2"}, "Attention implementation must be either eager, sdpa or flash_attention_2"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.dim),
            nn.LayerNorm(config.dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.dropout = nn.Dropout(config.emb_dropout)
        self.transformer = MetomTransformer(
            config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim, config.dropout, config._attn_implementation
        )
        self.pool = config.pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(config.dim, len(config.labels))
        self.labels = config.labels

    def forward(self, processed_image):
        x = self.to_patch_embedding(processed_image)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

    def get_predictions(self, processed_image: torch.Tensor) -> List[str]:
        logits = self(processed_image)
        indices = torch.argmax(logits, dim=-1)
        return [self.labels[i] for i in indices]

    def get_topk_labels(
        self, processed_image: torch.Tensor, k: int = 5, return_probs: bool = False
    ) -> Union[List[List[str]], List[List[Tuple[str, float]]]]:
        assert 0 < k <= len(self.labels), "k must be a positive integer less than or equal to the number of labels"
        logits = self(processed_image)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        topk_labels = [[self.labels[i] for i in ti] for ti in topk_indices]
        if return_probs:
            return [
                [(label, prob.item()) for label, prob in zip(labels, probs)]
                for labels, probs in zip(topk_labels, topk_probs)
            ]
        return topk_labels
