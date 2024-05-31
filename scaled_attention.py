from dataclasses import dataclass
from config import Config

seq_len = 5
shape = (seq_len, seq_len)
arr = jnp.ones(shape)
array = jnp.triu(arr, 1)
print(array)


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
        cfg = self.config  # don't need this?
        d_k = q.shape[-1]

        # q: (seq_len, d_model), k.T (d_model, seq_len)
        qk_matmul = jnp.matmul(q, k.T)  # (seq_len, seq_len)

        scaled_logits = (qk_matmul) / jnp.sqrt(d_k)

        if mask is not None:
            # apply padding mask here
            scaled_logits += (mask - 1) * 1e9

        if self.masked:
            look_ahead_mask = jnp.triu(jnp.ones(cfg.seq_len, cfg.seq_len), k=1)
            scaled_logits += -look_ahead_mask * 1e9

        attn = nn.softmax(scaled_logits) * v

        return attn
