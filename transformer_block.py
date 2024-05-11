from config import Config
from multi_head_attention import MultiHeadAttention
from feed_forward import MLP

class TransformerBlock(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        """
        
        """
        # setup
        cfg = self.config
        multi_head_attn = MultiHeadAttention(cfg)

        # need to distinguish this from the input embedding layer
        ff = MLP(cfg)

        # To Do: Handle masking separate encoder/decoder

        y = multi_head_attn(q=x, k=x, v=x)
        y = y + x # add

        y = nn.LayerNorm()(y)

        x = ff(y)
        x = x + y # add

        x = nn.LayerNorm()(x)

        return x