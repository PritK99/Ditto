import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from model import Model1, Encoder, Decoder
from utils import generate_square_subsequent_mask


if __name__ == "__main__":
    # Model parameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    nhead = 8
    num_layers = 6

    # Create model
    model = Model1(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
    )

    # Example input (batch_size=2, seq_len=10)
    src = torch.randint(1, src_vocab_size, (2, 10))
    tgt = torch.randint(1, tgt_vocab_size, (2, 8))

    # Create causal mask for decoder
    tgt_mask = generate_square_subsequent_mask(tgt.size(1))

    flag = 0
    # Forward pass
    with torch.no_grad():
        output = model(src, tgt, flag, tgt_mask=tgt_mask)
        print(f"Input shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        print(f"Output shape: {output.shape}")  # (batch_size, tgt_seq_len, tgt_vocab_size)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Example of encoder-only usage
    encoder = Encoder(vocab_size=src_vocab_size)
    encoded = encoder(src)
    print(f"Encoder output shape: {encoded.shape}")  # (batch_size, seq_len, d_model)

    # Example of decoder-only usage
    decoder = Decoder(vocab_size=tgt_vocab_size)
    decoded = decoder(tgt, encoded, tgt_mask=tgt_mask)
    print(f"Decoder output shape: {decoded.shape}")  # (batch_size, tgt_seq_len, vocab_size)