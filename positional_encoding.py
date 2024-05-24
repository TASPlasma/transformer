from dataclasses import dataclass
from config import Config


@dataclass
class PositionalEncoding:
    """
    Uses np/jnp wizardry that I cooked up but barely understand
    avoids if statements/piecewise expressions via
    modulo 2 logic
    """
    config: Config

    def __call__(self):
        """
        none -> (seq_len, d_model)
        """
        cfg = self.config
        shape = (cfg.seq_len, cfg.model_size)
        pos = jnp.arange(0, cfg.seq_len)[:, jnp.newaxis] * jnp.ones(shape)
        dims = jnp.arange(0, cfg.model_size) * jnp.ones(shape)

        pe = (dims % 2 == 0) * \
            (jnp.sin(pos/(cfg.pe_bound ** (dims / cfg.model_size))))
        pe = pe + (dims % 2 == 1) * \
            (jnp.cos(pos/(cfg.pe_bound ** ((dims - 1) / cfg.model_size))))

        return pe
