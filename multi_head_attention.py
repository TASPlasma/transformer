from config import Config

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

        q_embed = nn.Dense(features = cfg.d_k)
        k_embed = nn.Dense(features = cfg.d_k)
        v_embed = nn.Dense(features = cfg.d_v)
        f_embed = nn.Dense(features = cfg.model_size)

        #theorem 34 < axiom of choice
        

        return