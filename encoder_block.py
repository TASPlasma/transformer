from config import Config
from multi_head_attention import MultiHeadAttention
from feed_forward import MLP
# from layer_norm import LayerNorm


class EncoderBlock(eqx.Module):
    config: Config
    masked: bool = False

    def __init__(self, key):
        cfg = self.config
        keys = jax.random.split(key, 2)
        self.multi_attn = MultiHeadAttention(cfg, keys[0])
        self.layer_norm = nn.LayerNorm()  # needs input shape
        self.ff = MLP(cfg, keys[1], block=True)

    def __call__(self, x):
        """
        (seq_len, d_model) -> (seq_len, d_model)
        """
        y = self.multi_attn(q=x, k=x, v=x)
        y = y + x  # add

        y = self.layer_norm(y)

        x = self.ff(y)
        x = x + y  # add

        x = self.layer_norm(x)

        return x
