from config import Config

class PositionalEncoding(nn.Module):
    """
    Uses np/jnp wizardry that I cooked up but barely understand
    avoids if statements/piecewise expressions via
    modulo 2 logic
    """
    config: Config

    @nn.compact
    def __call__(self):
        """
        
        """
        cfg = self.config
        shape = (cfg.seq_len, cfg.model_size)
        pos = jnp.arange(0, cfg.seq_len)[:, jnp.newaxis] * jnp.ones(shape)
        dims = jnp.arange(0, cfg.model_size) * jnp.ones(shape)

        pe = (dims % 2 == 0)*(jnp.sin(pos/(cfg.pe_bound ** dims)))
        pe = pe +  + (dims % 2 == 1)*(jnp.cos(pos/(cfg.pe_bound ** (dims - 1))))

        return pe