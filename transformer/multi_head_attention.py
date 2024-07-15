import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from .config import Config
from .single_head import SingleHead


class MultiHeadAttention(eqx.Module):
    """
    config: Config
    masked: bool indicating if a look ahead masked is to be used

    forward: (seq_len, d_model)^3 -> (seq_len, d_model)
    """
    f_embed: nn.Linear
    head_layers: list

    def __init__(self, config: Config, key=None, masked: bool = False):
        cfg = config
        keys = jax.random.split(key, cfg.num_heads+1)

        # final dense layer
        self.f_embed = nn.Linear(
            in_features=cfg.num_heads * cfg.d_v, out_features=cfg.model_size, key=keys[0])
        self.head_layers = [SingleHead(cfg, key=keys[i+1], masked=masked)
                            for i in range(cfg.num_heads)]

    def __call__(self, q, k, v, mask):
        """
        q, k, v
        (seq_len, d_model)^3 -> (seq_len, d_model)
        mask: padding mask
        """
        heads = [head_layer(q, k, v, mask) for head_layer in self.head_layers]

        output = jnp.concatenate(heads, axis=1)
        output = jax.vmap(self.f_embed)(output)

        return output
