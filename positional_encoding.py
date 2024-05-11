from config import Config

class PositionalEncoding(nn.Module):
    config: Config

    @nn.compact
    def __call__(self):
        """
        
        """
        cfg = self.config
        pe = jnp.zeros((cfg.seq_len, cfg.model_size)) # hoping this returns a tensor of shape (s, d_m)
        pos = jnp.arange(0, cfg.seq_len) # makes [0, 1, 2, 3, ...]

        denom = 1 / cfg.pe_bound


        return pe