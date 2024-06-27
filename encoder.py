import jax
import equinox as eqx
import equinox.nn as nn
from config import Config
from encoder_block import EncoderBlock


class Encoder(eqx.Module):
    layers: list

    def __init__(self, config: Config, key=None, masked: bool = False):
        cfg = config
        keys = jax.random.split(key, cfg.num_layers)
        self.layers = [EncoderBlock(cfg, keys[i])
                       for i in range(cfg.num_layers)]

    def __call__(self, x, mask):
        """
        x: embedded input that has already been superimposed with
        positional encoding
        mask: (seq_len) sequence of 0s and 1s indicating where padding
        occurs
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
