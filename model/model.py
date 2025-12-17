import torch
import math
import torch.nn as nn
from utils import triu_to_full_matrix

class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, pad_idx):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model, pad_idx)
    
    def forward(self, x):
        token_embedding = self.embedding_layer(x)
        return token_embedding

class RelativeBiasPE(nn.Module):
    def __init__(self, num_heads: int, pos_vocab_size: int):
        super().__init__()
        self.relative_bias_pe_layer = nn.Embedding(pos_vocab_size, num_heads)
    
    def forward(self, dist_matrix):
        pad_mask = dist_matrix < 0    # Because we assign PAD positions as -1
        dist_matrix = dist_matrix.clamp(min=0)
        rel_bias_matrix = self.relative_bias_pe_layer(dist_matrix)
        rel_bias_matrix = rel_bias_matrix.masked_fill(pad_mask.unsqueeze(-1), 0)    # PAD masked embeddings will become 0
        return rel_bias_matrix
    
class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, max_seq_len, base=10000):
        super().__init__()
        assert d_head % 2 == 0

        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        pos = torch.arange(max_seq_len)    # We compute sine and cosine once for max_seq_len and use them directly for any sequence
        freqs = torch.einsum("i,j->ij", pos, inv_freq)

        sin = freqs.sin().repeat_interleave(2, dim=-1)
        cos = freqs.cos().repeat_interleave(2, dim=-1)

        self.register_buffer("sin", sin[None, None, :, :], persistent=False)
        self.register_buffer("cos", cos[None, None, :, :], persistent=False)
    
    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    
    def apply_rope(self, q, k, sin, cos):
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k

    def forward(self, q, k):
        L = q.size(-2)
        sin = self.sin[..., :L, :]
        cos = self.cos[..., :L, :]
        return self.apply_rope(q, k, sin, cos)

class LayerNorm(nn.Module):
    """
    Normalizes the data across dimensions.
    """
    def __init__(self, d_model: int) -> None:
        """
        Args:
            d_model (int): dimension of each embedding vector.
        """
        super().__init__()
        # We need two trainable param vector alpha and beta for each dimension
        self.alpha = nn.Parameter(torch.ones(d_model).unsqueeze(0).unsqueeze(0))    
        self.beta = nn.Parameter(torch.zeros(d_model).unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (batch_size, seq_len, d_model).
        
        Returns:
            x (torch.Tensor): layer normalized matrix of shape (batch_size, seq_len, d_model).
        """
        epsilon = 1e-5    # Using epsilon to avoid 0 in denominator

        # The input is (batch_size, seq_len, d_model)
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        x = self.alpha*((x - mean)/(std + epsilon)) + self.beta

        return x

class FeedForwardNetwork(nn.Module):
    """
    Introduces non-linearity in each layer.
    """
    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        """
        Args:
            d_model (int): dimension of each embedding vector.
            hidden_size (int): size of hidden layer in feedforward neural network.
            dropout (float): dropout value.
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size

        # Defining the weights 
        self.w1 = nn.Linear(d_model, hidden_size)
        self.gelu = nn.GELU()    # nn.ReLU()    # GELU is considered more stable for training than ReLU
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden_size, d_model)
    
    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (batch_size, seq_len, d_model).
        
        Returns:
            x (torch.Tensor): transformed matrix of shape (batch_size, seq_len, d_model).
        """
        x = self.gelu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x

class ResidualConnection(nn.Module):
    """
    Adds a residual connection around a sublayer.
    """
    def __init__(self, d_model: int, dropout: float) -> None:
        """
        Args:
            d_model (int): dimension of each embedding vector.
            dropout (float): dropout value.
        """
        super().__init__()
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape (batch_size, seq_len, d_model).
            sublayer (callable): function like self attention or feed forward network.
        
        Returns:
            x (torch.Tensor): resultant matrix of shape (batch_size, seq_len, d_model).
        """
        sublayer_output = sublayer(self.layernorm(x))    # Pre-Norm
        return x + self.dropout(sublayer_output)    # The residual path does not get dropout 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, pos_vocab_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "embedding dimension should be divisible by num heads" 

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Defining Q, K, V matrices
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out= nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rope_pe = RotaryEmbedding(self.d_head, max_seq_len)  
        self.rel_bias_pe = RelativeBiasPE(num_heads, pos_vocab_size)
    
    def forward(self, x, dist_matrix):
        Q = self.q_linear(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_linear(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_linear(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        
        Q, K = self.rope_pe(Q, K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        rel_bias_matrix = self.rel_bias_pe(dist_matrix)    # This is (batch_size, max_seq_len, max_seq_len, num_heads)
        rel_bias_matrix = rel_bias_matrix.permute(0,3,1,2)    # This is (batch_size, num_heads, max_seq_len, max_seq_len)
        scores += rel_bias_matrix

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, V) 
        out = out.transpose(1,2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        out = self.out(out)

        return out