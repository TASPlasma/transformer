from dataclasses import dataclass
from config import Config


@dataclass
class ScaledDotProdAttention:
    config: Config
    masked: bool = False
    decoder: bool = False

    def __call__(self, q, k, v, mask=None):
        """
        q: (seq_len, d_model)
        k: (seq_len, d_model)
        v: (seq_len, d_model)
        """
        cfg = self.config
        d_k = q.shape[-1]

        # q: (seq_len, d_model), k.T (d_model, seq_len)
        qk_matmul = jnp.matmul(q, k.T)  # (seq_len, seq_len)

        scaled_logits = (qk_matmul) / jnp.sqrt(d_k)

        if mask is not None:
            # apply mask here
            scaled_logits += (mask - 1) * 1e9

        attn = nn.softmax(scaled_logits) * v

        return attn
