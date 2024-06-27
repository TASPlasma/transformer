import jax
import equinox as eqx
import equinox.nn as nn
from config import Config
from decoder_block import DecoderBlock


class Decoder(eqx.Module):
    layers: list

    def __init__(self, config: Config, key=None, masked: bool = False):
        # Create a ModuleList and add each TransformerBlock with a unique name
        cfg = config
        keys = jax.random.split(key, cfg.num_layers)
        self.layers = [DecoderBlock(cfg, key=keys[i], masked=masked)
                       for i in range(cfg.num_layers)]

    def __call__(self, enc_out, x, mask):
        """
        x: embedded output that has already been superimposed with
        positional encoding
        enc_out: output from the encoder
        mask: padding mask
        """
        for layer in self.layers:
            x = layer(enc_out, x, mask)
        return x
