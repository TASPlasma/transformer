from config import Config
from decoder_block import DecoderBlock


class Decoder(eqx.Module):
    # needs a look-ahead mask
    config: Config
    masked: bool = False
    decoder: bool = True

    def __init__(self, key):
        # Create a ModuleList and add each TransformerBlock with a unique name
        cfg = self.config
        keys = jax.random.split(key, cfg.num_layers)
        self.layers = [DecoderBlock(cfg, self.masked, self.decoder, keys[i])
                       for i in range(cfg.num_layers)]

    def __call__(self, x, mask):
        """
        x: embedded input that has already been superimposed with
        positional encoding
        mask: padding mask
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
