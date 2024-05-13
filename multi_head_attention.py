from config import Config
from single_head import SingleHead

class MultiHeadAttention(nn.Module):
    """
    q: (seq_len, d_model)
    k: (seq_len, d_model)
    v: (seq_len, d_model)
    """
    config: Config

    @nn.compact
    def __call__(self, q, k, v):
        """
        (seq_len, d_model)^3 -> (seq_len, d_model)
        """
        cfg = self.config

        # final dense layer
        f_embed = nn.Dense(features = cfg.model_size)

        heads = [SingleHead(cfg) for i in range(cfg.num_heads)]

        return