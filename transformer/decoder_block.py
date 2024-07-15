import jax
import equinox as eqx
import equinox.nn as nn
from .config import Config
from .multi_head_attention import MultiHeadAttention
from .feed_forward import MLP


class DecoderBlock(eqx.Module):
    masked_multi_attn: eqx.Module
    multi_attn: eqx.Module
    layer_norm: nn.LayerNorm
    ff: eqx.Module

    def __init__(self, config: Config, key=None, masked: bool = False):
        cfg = config
        keys = jax.random.split(key, 3)
        self.masked_multi_attn = MultiHeadAttention(cfg, keys[0], masked)
        self.multi_attn = MultiHeadAttention(cfg, keys[1])
        self.layer_norm = nn.LayerNorm(shape=(cfg.seq_len, cfg.model_size))
        self.ff = MLP(cfg, keys[2], block=True)

    def __call__(self, enc_out, x, mask=None):
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

        y = jax.vmap(self.ff)(x)
        y = x + y  # add

        y = self.layer_norm(y)

        return y
