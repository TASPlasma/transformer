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
    pe: jax.Array
    # seq_len: int
    # model_size: int
    # pe_bound: int

    def __init__(self, config: Config, key):
        super().__init__()
        # self.seq_len = config.seq_len
        # self.model_size = config.seq_len
        # self.pe_bound = config.pe_bound
        # self.pe = self._calculate_pe()
        self.pe = jax.random.normal(key, (config.model_size, ))

    # def _calculate_pe(self):
    #     shape = (self.seq_len, self.model_size)
    #     pos = jnp.arange(0, self.seq_len)[:, jnp.newaxis] * jnp.ones(shape)
    #     dims = jnp.arange(0, self.model_size) * jnp.ones(shape)

    #     pe = (dims % 2 == 0) * \
    #         (jnp.sin(pos/(self.pe_bound ** (dims / self.model_size))))
    #     pe = pe + (dims % 2 == 1) * \
    #         (jnp.cos(pos/(self.pe_bound ** ((dims - 1) / self.model_size))))

    #     return pe

    def __call__(self, x):
        """
        (seq_len, d_model) -> (seq_len, d_model)
        """
        return x + self.pe
