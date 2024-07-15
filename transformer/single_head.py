import jax
import equinox as eqx
import equinox.nn as nn
from .config import Config
from .scaled_attention import ScaledDotProdAttention


class SingleHead(eqx.Module):
    """
    q: (seq_len, d_model)
    k: (seq_len, d_model)
    v: (seq_len, d_model)
    """
    q_layer: nn.Linear
    k_layer: nn.Linear
    v_layer: nn.Linear
    attn: ScaledDotProdAttention

    def __init__(self, config: Config, key=None, masked: bool = False):
        cfg = config
        keys = jax.random.split(key, 3)

        self.q_layer = nn.Linear(
            in_features=cfg.model_size, out_features=cfg.d_k, key=keys[0])
        self.k_layer = nn.Linear(
            in_features=cfg.model_size, out_features=cfg.d_k, key=keys[1])
        self.v_layer = nn.Linear(
            in_features=cfg.model_size, out_features=cfg.d_v, key=keys[2])
        self.attn = ScaledDotProdAttention(cfg, masked)

    def __call__(self, q, k, v, mask):
        """
        q, k, v: (seq_len, d_model)
        (seq_len, d_model)^3 -> (seq_len, d_v)
        mask: padding mask
        """
        q_embed = jax.vmap(self.q_layer)(q)
        k_embed = jax.vmap(self.k_layer)(k)
        v_embed = jax.vmap(self.v_layer)(v)

        head = self.attn(q_embed, k_embed, v_embed, mask)

        return head
