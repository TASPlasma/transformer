import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass
from .config import Config


@dataclass
class PositionalEncoding(eqx.Module):
    """
    Positional encoding as in Attention Is All You Need
    avoids if statements/piecewise expressions via
    modulo 2 logic
    """
    pe: jax.Array

    def __init__(self, config: Config):
        self.config = config

    def __call__(self):
        """
        none -> (seq_len, d_model)
        """
        cfg = self.config
        shape = (cfg.seq_len, cfg.model_size)
        pos = jnp.arange(0, cfg.seq_len)[:, jnp.newaxis] * jnp.ones(shape)
        dims = jnp.arange(0, cfg.model_size) * jnp.ones(shape)

        self.pe = (dims % 2 == 0) * \
            (jnp.sin(pos/(cfg.pe_bound ** (dims / cfg.model_size))))
        self.pe = self.pe + (dims % 2 == 1) * \
            (jnp.cos(pos/(cfg.pe_bound ** ((dims - 1) / cfg.model_size))))

        return self.pe
