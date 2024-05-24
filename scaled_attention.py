from dataclasses import dataclass
from config import Config


@dataclass
class ScaledDotProdAttention:
    config: Config
    masked: bool = False
    decoder: bool = False

    def __call__(self, q, k, v):
        """
        q: (seq_len, d_model)
        k: (seq_len, d_model)
        v: (seq_len, d_model)
        """
        cfg = self.config
        d_k = q.shape[-1]

        scaled_logits = (q * k.transpose()) / jnp.sqrt(d_k)

        attn = nn.softmax(scaled_logits) * v

        return attn
