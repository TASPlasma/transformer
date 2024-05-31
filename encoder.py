from config import Config
from encoder_block import EncoderBlock


class Encoder(eqx.Module):
    config: Config
    masked: bool = False

    def __init__(self, key):
        cfg = self.config
        keys = jax.random.split(key, cfg.num_layers)
        self.layers = [EncoderBlock(cfg, self.masked, keys[i])
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
