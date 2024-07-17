from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    sizes: list[int] = field(default_factory=lambda: [64, 256, 512])
    ff_sizes: list[int] = field(default_factory=lambda: [2048, 512])
    d_k: int = 64
    d_v: Optional[int] = None
    input_dim: int = 2
    out_dim: Optional[int] = None
    pe_bound: int = 1e4
    num_heads: int = 8
    num_layers: int = 6
    seq_len: int = 55
    dropout_rate: float = 0.1
    num_classes: int = 7
    seed: int = 0

    def __post_init__(self):
        if self.d_v is None:
            self.d_v = self.d_k
        if self.out_dim is None:
            self.out_dim = self.num_classes
        self.model_size = self.d_v * self.num_heads

    def __hash__(self):
        return str(self)
