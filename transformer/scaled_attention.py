import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass
from typing import Tuple
from .config import Config


class ScaledDotProdAttention(eqx.Module):
    """
    masked: boolean for look-ahead mask
    """
    masked: bool
    shape: Tuple

    def __init__(self, config: Config, masked: bool = False):
        self.masked = masked
        self.shape = (config.seq_len, config.seq_len)

    def __call__(self, q, k, v, mask=None):
        """
        q: (seq_len, d_model) or (seq_len, d_k)
        k: (seq_len, d_model) or (seq_len, d_k)
        v: (seq_len, d_model) or (seq_len, d_v)
        """

        d_k = q.shape[-1]

        # q: (seq_len, d_model), k.T: (d_model, seq_len)
        qk_matmul = jnp.matmul(q, k.T)  # (seq_len, seq_len)

        scaled_logits = (qk_matmul) / jnp.sqrt(d_k)

        if mask is not None:
            # apply padding mask here
            scaled_logits += (mask - 1) * 1e9

        if self.masked:
            look_ahead_mask = jnp.triu(
                jnp.ones(shape=self.shape), k=1)
            scaled_logits -= look_ahead_mask * 1e9

        attn = jnp.matmul(jax.nn.softmax(scaled_logits), v)

        return attn
