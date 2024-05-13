from config import Config
from transformer_block import TransformerBlock

class Encoder(nn.Module):
    config: Config
    mask: bool = False

    def setup(self):
        # Create a ModuleList and add each TransformerBlock with a unique name
        self.layers = [TransformerBlock(self.config, self.mask, name=f"layer_{i}")
                       for i in range(self.config.num_layers)]

    def __call__(self, x):
        """
        x: embedded input that has already been superimposed with
        positional encoding
        """
        for layer in self.layers:
            x = layer(x)
        return x