import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from config import Config
from multi_head_attention import MultiHeadAttention
from feed_forward import MLP
# from layer_norm import LayerNorm


class TransformerBlock(eqx.Module):
    config: Config
    masked: bool = False
    decoder: bool = False

    def __init__(self, key):
        cfg = self.config
        keys = jax.random.split(key, 3)
        self.masked_multi_attn = MultiHeadAttention(cfg, self.masked, keys[0])
        self.multi_attn = MultiHeadAttention(cfg, keys[1])
        self.layer_norm = nn.LayerNorm()
        self.ff = MLP(cfg, keys[2], block=True)

    def __call__(self, x, enc_out):
        """

        """

        y = self.multi_attn(q=x, k=x, v=x)
        y = y + x  # add

        y = self.layer_norm(y)

        x = self.ff(y)
        x = x + y  # add

        x = self.layer_norm(x)

        return x
