from config import Config

class ScaledDotProdAttention(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, q, k, v):
        """
        q: (seq_len, d_model)
        k: (seq_len, d_model)
        v: (seq_len, d_model)
        """
        cfg = self.config

        attn = nn.softmax((q * k.transpose()) / jnp.sqrt(cfg.model_size)) * v

        return attn
        