from typing import List

from transformers import PretrainedConfig


class MetomConfig(PretrainedConfig):
    model_type = "metom"

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        labels: List[str] = [],
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 32,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            labels=labels,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            **kwargs
        )
