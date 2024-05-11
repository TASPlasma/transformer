from config import Config
from multi_head_attention import MultiHeadAttention
from feed_forward import MLP
# from layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    config: Config
    block: bool = True

    @nn.compact
    def __call__(self, x):
        """
        
        """
        # setup
        cfg = self.config
        multi_head_attn = MultiHeadAttention(cfg)
        # layer_norm = LayerNorm
        layer_norm = nn.LayerNorm()

        # need to distinguish this from the input embedding layer
        ff = MLP(cfg, block=True) # want a boolean declaring this to be an MLP inside the transformer block
        # which pulls the ff_sizes array, instead of the sizes array

        # To Do: Handle masking separate encoder/decoder

        y = multi_head_attn(q=x, k=x, v=x)
        y = y + x # add

        y = layer_norm(y)

        x = ff(y)
        x = x + y # add

        x = layer_norm(x)

        return x