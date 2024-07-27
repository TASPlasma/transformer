import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from .config import Config
from .feed_forward import MLP
from .positional_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder
from typing import Tuple


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

    custom branch: modifies architecture for racing simulation task
    specifically this archictecture splits input: (bs, seq_len, in_dim)
    into state: (bs, state_seq_len, in_dim), verts: (bs, vert_seq_len, in_dim)
    converts state into (bs, 1, in_dim * state_seq_len),
    maps it via an MLP to (bs, 1, state_out_dim), then merges with
    decoder_out: (bs, seq_len, model_size) to obtain
    merged: (bs, seq_len, model_size+state_out_dim)
    which gets passed into the final linear layer
    dense: (bs, seq_len, model_size+state_out_dim) -> (bs, seq_len, num_classes)
    """
    pos_enc: eqx.Module
    in_embed: eqx.Module
    out_embed: eqx.Module
    encoder: eqx.Module
    decoder: eqx.Module
    dense: nn.Linear
    state_dense: eqx.Module
    shape: Tuple

    def __init__(self, cfg: Config, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.shape = (cfg.seq_len, cfg.state_out_dim)
        # block is default false
        self.in_embed = MLP(cfg, key=key)
        keys = list(jax.random.split(key, 6))
        self.out_embed = MLP(
            cfg, key=keys[3], out_emb=True)
        self.state_dense = MLP(
            cfg, key=keys[5], state=True)

        self.pos_enc = PositionalEncoding(cfg, key=keys[4])
        self.encoder = Encoder(cfg, key=keys[0])
        self.decoder = Decoder(cfg, key=keys[1], masked=True)
        self.dense = nn.Linear(in_features=cfg.model_size+cfg.state_out_dim,
                               out_features=cfg.num_classes, key=keys[2])

    def __call__(self, input, output):
        """
        input: the input to the encoder model (before embedded)
        output: the output

        (seq_len, input_dim) x (seq_len, num_classes) -> {num_classes}^(seq_len)
        """
        state, verts = jnp.split(
            input, [3], axis=-2)  # first 3 tuples are the state
        state = state.ravel()  # (bs, 1, 3*in_dim)
        verts_pad_mask = self.create_pad_mask(verts)
        out_pad_mask = self.create_pad_mask(output)
        verts_emb = jax.vmap(self.in_embed)(verts)
        verts_emb = jax.vmap(self.pos_enc)(verts_emb)
        encoder_out = self.encoder(verts_emb, verts_pad_mask)
        output_emb = jax.vmap(self.out_embed)(output)
        output_emb = jax.vmap(self.pos_enc)(output_emb)
        decoder_out = self.decoder(encoder_out, output_emb, out_pad_mask)
        state_out = self.state_dense(state)
        state_out = jnp.repeat(state_out, decoder_out.shape[0], axis=0)
        state_out = jnp.reshape(state_out, shape=self.shape)
        merged = jnp.concatenate([decoder_out, state_out], axis=-1)
        logits = jax.vmap(self.dense)(merged)

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
