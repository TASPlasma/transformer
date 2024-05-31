from config import Config
from multi_head_attention import MultiHeadAttention
from feed_forward import MLP
# from layer_norm import LayerNorm


class DecoderBlock(eqx.Module):
    config: Config
    masked: bool = False

    def __init__(self, key):
        cfg = self.config
        keys = jax.random.split(key, 3)
        self.masked_multi_attn = MultiHeadAttention(cfg, self.masked, keys[0])
        self.multi_attn = MultiHeadAttention(cfg, keys[1])
        self.layer_norm = nn.LayerNorm()
        self.ff = MLP(cfg, keys[2], block=True)

    def __call__(self, x, enc_out, mask=None):
        """
        Needs an input x, and the output of the encoder enc_out.
        (seq_len, d_model) x (seq_len, d_model) -> (seq_len, d_model)
        """

        y = self.masked_multi_attn(q=x, k=x, v=x, mask=mask)
        y = y + x  # add
        y = self.layer_norm(y)

        x = self.multi_attn(enc_out, enc_out, y, mask)
        x = x + y
        x = self.layer_norm(x)

        y = self.ff(x)
        y = x + y  # add

        y = self.layer_norm(y)

        return y
