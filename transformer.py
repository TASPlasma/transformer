import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from config import Config
from feed_forward import MLP
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder


class Transformer(eqx.Module):
    """
    model_size: dimension of domain of input embedding
    input_dim: dimension of inputs prior to embedding to model
    num_heads: number of heads in multi-head attention
    num_layers: N in the paper, the number of times the encoder is stacked
    this is the number of transformer blocks used
    seq_len: length of the sequence
    num_classes: number of output classes
    for example, num_classes=2 means binary classification
    sizes: dimensions to map input_dim to input_embedding via MLP
    """
    pos_enc: PositionalEncoding
    embed: eqx.Module
    encoder: eqx.Module
    decoder: eqx.Module
    dense: nn.Linear

    def __init__(self, config: Config, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        cfg = config
        self.embed = MLP(cfg, key=key)  # block is default false
        keys = list(jax.random.split(key, 3))

        self.pos_enc = PositionalEncoding(cfg)
        self.encoder = Encoder(cfg, key=keys[0])
        self.decoder = Decoder(cfg, key=keys[1], masked=True)
        self.dense = nn.Linear(in_features=cfg.model_size,
                               out_features=cfg.num_classes, key=keys[2])

    # (seq_len, input_dim) -> (seq_len, 1)

    def __call__(self, input, output):
        """
        input: the input to the encoder model (before embedded)
        output: the output

        (seq_len, input_dim) x output_shape -> {num_classes}^(seq_len)
        """
        in_pad_mask = self.create_pad_mask(input)
        out_pad_mask = self.create_pad_mask(output)
        input_emb = jax.vmap(self.embed)(input)
        input_emb = input_emb + self.pos_enc()
        encoder_out = self.encoder(input_emb, in_pad_mask)
        output_emb = jax.vmap(self.embed)(output)
        output_emb = output_emb + self.pos_enc()
        decoder_out = self.decoder(encoder_out, output_emb, out_pad_mask)
        logits = jax.vmap(self.dense)(decoder_out)

        return logits

    def create_pad_mask(self, input):
        """
        Creates a mask on padded values
        (seq_len, input_dim) -> {0, 1}^seq_len

        e.g. max_seq_len = 3, input_dim = 2,
        input = 
        [[0.3, 1.0],
        [0.2, -0.4],
        [0.0, 0.0]]
        |->
        [1, 1, 0] (or [True, True, False], True when a row is not a padded row)
        """
        mask = jnp.sum(jnp.abs(input), axis=-1) != 0
        return mask.astype(int)
