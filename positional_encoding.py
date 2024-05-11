from config import Config

seq_len = 7
d_m = 4

n = 5

shape = (seq_len, d_m)
pos = jnp.arange(0, seq_len)[:, np.newaxis] * jnp.ones(shape)

dims = jnp.arange(0, d_m) * jnp.ones((seq_len, d_m))

pe = (dims % 2 == 0)*(jnp.sin(pos/(n ** dims)))
pe = pe + + (dims % 2 == 1)*(jnp.cos(pos/(n ** (dims - 1))))

print('dims: \n', dims)

print('pos: \n', pos)

print('pe: \n', pe)

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