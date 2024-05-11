from dataclasses import dataclass

@dataclass
class Config:
    """
    Class for hyperparameters of transformer model:

    -sizes: dimensions to map input_dim to input_embedding via MLP
    -model_size: dimension of domain of input embedding
    -input_dim: dimension of inputs prior to embedding to model
    -num_heads: number of heads in multi-head attention
    -num_layers: N in the paper, the number of times the encoder is stacked
    this is the number of transformer blocks used
    -seq_len: length of the sequence
    -num_classes: number of output classes
    for example, num_classes=2 means binary classification
    -seed: rng seed
    """
    sizes: list[int]
    model_size: int = 512
    input_dim: int = 2
    num_heads: int = 8
    num_layers: int = 6
    seq_len: int = 55
    dropout_rate: float = 0.1
    num_classes: int = 15
    seed: int = 0

config = Config(sizes=[64, 256, 512])
print(config.model_size)