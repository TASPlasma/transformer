import jax
import equinox as eqx
import equinox.nn as nn
from config import Config


class MLP(eqx.Module):
    config: Config
    block: bool = False

    def __init__(self, key):
        cfg = self.config
        array_sizes = cfg.ff_sizes if self.block else cfg.sizes
        in_size = cfg.model_size if self.block else cfg.input_dim
        keys = jax.random.split(key, len(array_sizes))

        first_layer = nn.Linear(in_size, array_sizes[0], key=keys[0])
        rem_layers = [nn.Linear(array_sizes[i-1], array_sizes[i], key=keys[i])
                      for i in range(1, len(array_sizes))]

        self.layers = [first_layer] + rem_layers
        self.activations = [
            nn.relu
            for _ in array_sizes[:-1]
        ]
        + [lambda x: x]

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x
