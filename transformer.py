from config import Config
from transformer.feed_forward import MLP
from transformer.positional_encoding import PositionalEncoding
from transformer.encoder import Encoder
from transformer.decoder import Decoder

# recall (m, n) means m rows, n columns
# Dense/Linear/Affine etc. maps input_dim to output_dim 
# and broadcasts the rest
# for example:
# (1, 5) -> (1, 7) gets broadcasted to (seq_len, 5) -> (seq_len, 7)
# (batch_size, seq_len, input_dim) -> (batch_size, seq_len, output_dim)

# Needs a mask and some other shit
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

    # (seq_len, input_dim) -> (seq_len, 1)
    @nn.compact
    def __call__(self, input, output):
        """
        input: the input to the encoder model (before embedded)
        possible change: add MLP into this class and only call encoder with embedded input
        output: the output
        """
        # setup
        cfg = self.config
        embed = MLP(cfg)
        pos_enc = PositionalEncoding(cfg)
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        dense = nn.Dense(features=cfg.num_classes)
        
        # input = input + pos_enc(input)
        input_emb = embed(input)
        input_emb = input_emb + pos_enc
        encoder_out = encoder(input_emb)
        decoder_out = decoder(encoder_out, output)
        logits = dense(decoder_out)

        return logits
        