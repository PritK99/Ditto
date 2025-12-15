import torch
import math
import torch.nn as nn

def triu_to_full_matrix(triu, max_dist):
    triu = torch.tensor(triu)
    n = int((1 + math.isqrt(1 + 8*len(triu))) // 2)

    dist_matrix = torch.zeros((n, n), dtype=triu.dtype)

    rows, cols = [], []
    for i in range(n):
        for j in range(i+1, n):
            rows.append(i)
            cols.append(j)
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)

    dist_matrix[rows, cols] = triu
    dist_matrix[cols, rows] = -triu

    dist_matrix = dist_matrix + max_dist

    return dist_matrix

def apply_rope(Q, K):
    B, H, L, D = Q.shape
    assert D % 2 == 0, "RoPE requires even head dimension"

    freqs = torch.arange(0, D, 2, device=Q.device).float() / D
    freqs = 10000 ** (-freqs)  

    pos = torch.arange(L, device=Q.device).float() 
    angles = torch.einsum('l,d->ld', pos, freqs) 

    sin = angles.sin()[None, None, :, :] 
    cos = angles.cos()[None, None, :, :]

    Q1, Q2 = Q[..., ::2], Q[..., 1::2]
    K1, K2 = K[..., ::2], K[..., 1::2]

    Q = torch.cat([Q1 * cos - Q2 * sin, Q1 * sin + Q2 * cos], dim=-1)
    K = torch.cat([K1 * cos - K2 * sin, K1 * sin + K2 * cos], dim=-1)

    return Q, K

class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, pad_idx):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model, pad_idx)
    
    def forward(self, x):
        token_embedding = self.embedding_layer(x)
        return token_embedding

class RelativeBiasPE(nn.Module):
    def __init__(self, d_model: int, pos_vocab_size: int):
        super().__init__()
        self.pos_encoding_layer = nn.Embedding(pos_vocab_size, d_model)
    
    def forward(self, dist):
        token_pos_encodings = self.pos_encoding_layer(dist)
        return token_pos_encodings

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, relative_bias_layer: RelativeBiasPE, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "embedding dimension should be divisible by num heads" 

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model / num_heads

        # Defining Q, K, V matrices
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out= nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.relative_bias_layer = relative_bias_layer

    def forward(self, x, rel_bias):
        Q = self.q_linear(x).view(x.shape[2], x.shape[1], self.num_heads, self.d_head).transpose(1, 2)  
        K = self.k_linear(x).view(x.shape[2], x.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_linear(x).view(x.shape[2], x.shape[1], self.num_heads, self.d_head).transpose(1, 2)

        # Apply RoPE (absolute positional embedding)
        Q, K = apply_rope(Q, K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        bias = rel_bias.sum(-1).unsqueeze(1)  # (B, 1, L, L)
        scores = scores + bias