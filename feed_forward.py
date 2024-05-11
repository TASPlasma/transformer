from config import Config

class MLP(nn.Module):
    """
    Embeds (55, 2) to (55, m) for a larger m
    2 is small an input dimension for multihead attention
    """
    config: Config
    block: bool = False

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        # Process each layer defined in layer_sizes
        array_sizes = cfg.ff_sizes if self.block else cfg.sizes
        for i, size in enumerate(array_sizes):
            x = nn.Dense(features=size)(x)  # Apply Dense layer
            if i < len(array_sizes) - 1:  # Apply ReLU activation to all but the last layer
                x = nn.relu(x)
        return x