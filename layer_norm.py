from config import Config

class LayerNorm(eqx.Module):
    config: Config

    @nn.compact
    def __call__(self):
        """
        
        """
        cfg = self.config
        pass