import jax
import jax.numpy as jnp
import equinox as eqx
from .config import Config


class PositionalEncoding(eqx.Module):
    """
    Positional encoding as in Attention Is All You Need
    avoids if statements/piecewise expressions via
    modulo 2 logic
    """
    config: Config
    pe: jax.Array

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.pe = self._calculate_pe()

    def _calculate_pe(self):
        cfg = self.config
        shape = (cfg.seq_len, cfg.model_size)
        pos = jnp.arange(0, cfg.seq_len)[:, jnp.newaxis] * jnp.ones(shape)
        dims = jnp.arange(0, cfg.model_size) * jnp.ones(shape)

        pe = (dims % 2 == 0) * \
            (jnp.sin(pos/(cfg.pe_bound ** (dims / cfg.model_size))))
        pe = pe + (dims % 2 == 1) * \
            (jnp.cos(pos/(cfg.pe_bound ** ((dims - 1) / cfg.model_size))))

        return pe

    def __call__(self):
        """
        none -> (seq_len, d_model)
        """
        return self.pe
