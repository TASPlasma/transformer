from config import Config
from scaled_attention import ScaledDotProdAttention

class SingleHead(nn.Module):
    """
    q: (seq_len, d_model)
    k: (seq_len, d_model)
    v: (seq_len, d_model)
    """
    config: Config

    @nn.compact
    def __call__(self, q, k, v):
        """
        (seq_len, d_model)^3 -> (seq_len, d_v)
        """
        cfg = self.config

        q_layer = nn.Dense(features = cfg.d_k)
        k_layer = nn.Dense(features = cfg.d_k)
        v_layer = nn.Dense(features = cfg.d_v)

        q_embed = q_layer(q)
        k_embed = k_layer(k)
        v_embed = v_layer(v)

        attn = ScaledDotProdAttention(cfg)

        head = attn(q_embed, k_embed, v_embed)
        
        return head