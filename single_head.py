from config import Config
from scaled_attention import ScaledDotProdAttention


class SingleHead(eqx.Module):
    """
    q: (seq_len, d_model)
    k: (seq_len, d_model)
    v: (seq_len, d_model)
    """
    config: Config
    masked: bool = False
    decoder: bool = False

    def __init__(self, key):
        cfg = self.config
        keys = jax.random.split(key, 3)

        self.q_layer = nn.Linear(features=cfg.d_k, key=keys[0])
        self.k_layer = nn.Linear(features=cfg.d_k, key=keys[1])
        self.v_layer = nn.Linear(features=cfg.d_v, key=keys[2])
        self.attn = ScaledDotProdAttention(cfg, self.masked, self.decoder)

    def __call__(self, q, k, v):
        """
        (seq_len, d_model)^3 -> (seq_len, d_v)
        """

        q_embed = self.q_layer(q)
        k_embed = self.k_layer(k)
        v_embed = self.v_layer(v)

        head = self.attn(q_embed, k_embed, v_embed)

        return head
