from config import Config
from multi_head_attention import MultiHeadAttention
from feed_forward import MLP
# from layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    config: Config
    block: bool = True
    mask: bool = False
    decoder: bool = False

    def setup(self):
        cfg = self.config
        multi_head_attn = MultiHeadAttention(cfg)
        layer_norm = nn.LayerNorm()
        ff = MLP(cfg, block=self.block)

    @nn.compact
    def __call__(self, x):
        """
        
        """

        # To Do: Handle masking separate encoder/decoder

        y = self.multi_head_attn(q=x, k=x, v=x)
        y = y + x # add

        y = self.layer_norm(y)

        x = self.ff(y)
        x = x + y # add

        x = self.layer_norm(x)

        return x