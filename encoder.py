from config import Config
from transformer_block import TransformerBlock

class Encoder(nn.Module):
    config: Config
    mask: bool = False

    @nn.compact
    def __call__(self, x):
        """
        x: embedded input that has already been superimposed with
        positional encoding
        """
        # setup
        cfg = self.config
        layers = [TransformerBlock(cfg, self.mask) for i in cfg.num_layers]

        for layer in layers:
            x = layer(x)
        
        return x