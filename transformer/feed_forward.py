import jax
import equinox as eqx
import equinox.nn as nn
from .config import Config


class MLP(eqx.Module):
    """
    Standard MLP neural network.
    Used to both embed the input to d_model
    and for the feed-forward portion of the transformer blocks
    block: if true indicates this is for a transformer block
    """
    layers: list

    def __init__(self, config: Config, key=None, block: bool = False, out_emb: bool = False):
        cfg = config
        array_sizes = cfg.ff_sizes if block else cfg.sizes
        if block:
            in_size = cfg.model_size
        elif out_emb:
            in_size = cfg.out_dim
        else:
            in_size = cfg.input_dim

        keys = jax.random.split(key, len(array_sizes))

        first_layer = nn.Linear(in_size, array_sizes[0], key=keys[0])
        rem_layers = [nn.Linear(array_sizes[i-1], array_sizes[i], key=keys[i])
                      for i in range(1, len(array_sizes))]

        self.layers = [first_layer] + rem_layers

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)
