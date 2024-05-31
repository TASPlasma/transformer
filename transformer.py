from config import Config
from feed_forward import MLP
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

# recall (m, n) means m rows, n columns
# Dense/Linear/Affine etc. maps input_dim to output_dim
# and broadcasts the rest
# for example:
# (1, 5) -> (1, 7) gets broadcasted to (seq_len, 5) -> (seq_len, 7)
# (batch_size, seq_len, input_dim) -> (batch_size, seq_len, output_dim)

# Needs a mask and some other stuff


class Transformer(nn.Module):
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
    config: Config

    def __init__(self, key):
        cfg = self.config
        keys = jax.random.split(key, 5)
        self.embed = MLP(cfg, keys[0])  # block is default false

        # needs to use vmap
        # self.embed = nn.Linear(in_features=cfg.input_dim,
        #                        out_features=cfg.model_size, use_bias=False)
        self.pos_enc = PositionalEncoding(cfg, keys[1])
        self.encoder = Encoder(cfg, keys[2])
        self.decoder = Decoder(cfg, keys[3])
        self.dense = nn.Linear(features=cfg.num_classes, key=keys[4])

    # (seq_len, input_dim) -> (seq_len, 1)

    def __call__(self, input, output):
        """
        input: the input to the encoder model (before embedded)
        output: the output

        (seq_len, input_dim) x output_shape -> {num_classes}^(seq_len)
        """
        pad_mask = self.create_pad_mask(input)
        look_ahead_mask = self.create_look_ahead_mask(output)
        input_emb = jax.vmap(self.embed)(input)
        input_emb = input_emb + self.pos_enc()
        encoder_out = self.encoder(input_emb, pad_mask)
        decoder_out = self.decoder(encoder_out, output, look_ahead_mask)
        logits = self.dense(decoder_out)

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
