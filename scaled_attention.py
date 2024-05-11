from config import Config

class ScaledDotProdAttention(nn.Module):
    config: Config

    @nn.compact
    def __call__(self):
        """
        
        """
        cfg = self.config
        pass