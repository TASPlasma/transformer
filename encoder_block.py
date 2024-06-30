import jax
import equinox as eqx
import equinox.nn as nn
from config import Config
from multi_head_attention import MultiHeadAttention
from feed_forward import MLP


class EncoderBlock(eqx.Module):
    multi_attn: eqx.Module
    layer_norm: nn.LayerNorm
    ff: eqx.Module

    def __init__(self, config: Config, key=None):
        cfg = config
        keys = jax.random.split(key, 2)
        self.multi_attn = MultiHeadAttention(cfg, keys[0])
        self.layer_norm = nn.LayerNorm(
            shape=(cfg.seq_len, cfg.model_size))
        self.ff = MLP(cfg, keys[1], block=True)

    def __call__(self, x, mask):
        """
        (seq_len, d_model) -> (seq_len, d_model)
        """
        y = self.multi_attn(q=x, k=x, v=x, mask=mask)
        y = y + x  # add

        y = self.layer_norm(y)

        x = jax.vmap(self.ff)(y)

        x = x + y  # add

        x = self.layer_norm(x)

        return x
