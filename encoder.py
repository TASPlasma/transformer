from config import Config
from transformer_block import TransformerBlock


class Encoder(eqx.Module):
    config: Config
    masked: bool = False

    def __init__(self, key):
        # Create a ModuleList and add each TransformerBlock with a unique name
        cfg = self.config
        keys = jax.random.split(key, cfg.num_layers)
        self.layers = [TransformerBlock(cfg, self.masked, keys[i])
                       for i in range(cfg.num_layers)]

    def __call__(self, x):
        """
        x: embedded input that has already been superimposed with
        positional encoding
        """
        for layer in self.layers:
            x = layer(x)
        return x
