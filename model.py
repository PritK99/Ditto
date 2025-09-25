import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import PositionalEncoding

class Encoder(nn.Module):
    """Transformer Encoder using PyTorch's built-in TransformerEncoder."""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Using PyTorch's built-in TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (batch_size, seq_len)
            src_mask: (seq_len, seq_len) - attention mask
            src_key_padding_mask: (batch_size, seq_len) - padding mask
        """
        # Embedding with scaling
        src = self.embedding(src) * math.sqrt(self.d_model)

        # Add positional encoding
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        src = self.dropout(src)

        # Pass through transformer encoder
        output = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        return output

class Decoder(nn.Module):
    """Transformer Decoder using PyTorch's built-in TransformerDecoder."""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Using PyTorch's built-in TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: (batch_size, tgt_seq_len)
            memory: (batch_size, src_seq_len, d_model) - encoder output
            tgt_mask: (tgt_seq_len, tgt_seq_len) - causal mask
            memory_mask: (tgt_seq_len, src_seq_len) - cross attention mask
            tgt_key_padding_mask: (batch_size, tgt_seq_len) - target padding mask
            memory_key_padding_mask: (batch_size, src_seq_len) - source padding mask
        """
        # Embedding with scaling
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        tgt = tgt.transpose(0, 1)  # (seq_len, batch_size, d_model)
        tgt = self.pos_encoder(tgt)
        tgt = tgt.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        tgt = self.dropout(tgt)

        # Pass through transformer decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Project to vocabulary size
        output = self.output_projection(output)

        return output


class Model1(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_len=5000):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )

        self.decoder1 = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )


        self.decoder2 = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )

    def forward(self, src, tgt, flag, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # Encode source sequence
        memory = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        if flag:    # for switching between 2 languages
            output = self.decoder1(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        else:
            output = self.decoder2(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return output