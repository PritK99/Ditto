import torch
import torch.nn as nn

# This class addresses both embeddings and postional encodings
class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
