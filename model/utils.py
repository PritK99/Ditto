import torch
import math

def get_dist_from_triu(i, j, triu, num_tokens, max_pos):
    """
    This function behaves like an abstraction for distance matrix, i.e., you can use indices i and j and get corresponding value for respective idx in triu
    """
    num_tokens = int((1 + math.isqrt(1 + 8*len(triu))) // 2)    # This is formula for num_tokens from triu length
    idx = (i * (2*num_tokens - i - 1)) // 2 + (j - i - 1)
    return triu[idx] + max_pos


def triu_to_full_matrix(triu, max_seq_len, max_pos):
    """
    This function constructs a max_seq_len by max_seq_len matrix from triu vector
    For encoder, we need to skip first row and first col of dist matrix
    For decoder, we need to skip Nth row and Nth column of dist matrix, where N corresponds to [EOS] position in original dist matrix
    """
    num_tokens = int((1 + math.isqrt(1 + 8*len(triu))) // 2)
    # dist_matrix = torch.zeros((max_seq_len, max_seq_len), dtype=triu.dtype)    # Our distance matrix needs to be of (1002, 1002)
    dist_matrix = torch.full((max_seq_len+1, max_seq_len+1), -1, dtype=torch.int64)    # We fill with -1 for [PAD]. +1 is because we are constructing the positional matrix for [SOS] + tokens + [EOS] first, and then trimming [SOS] row for encoder input and [EOS] for decoder input

    for i in range(1, num_tokens+1):
        for j in range(i, num_tokens+1):
            if (i == j):
                dist_matrix[i][j] = max_pos 
            else:
                dist_matrix[i][j] = get_dist_from_triu(i-1, j-1, triu, num_tokens, max_pos)
                dist_matrix[j][i] = 2*max_pos - dist_matrix[i][j] 
    
    dist_matrix[0][0] = max_pos

    # This is distance of [SOS] from all token. We assume [SOS] is connected by edge to first token in AST
    for i in range(1, num_tokens+1):
        dist_matrix[i][0] = dist_matrix[i][1] - 1 
    
    for j in range(1, num_tokens+1):
        dist_matrix[0][j] = dist_matrix[1][j] + 1 
    
    # The matrix built till here is positional matrix for decoder input
    decoder_dist_matrix = dist_matrix.clone()
    
    dist_matrix[num_tokens + 1][num_tokens + 1] = max_pos
    
    # This is distance of [EOS] from all token. We assume [EOS] is connected by edge to last token in AST
    for i in range(1, num_tokens+1):
        dist_matrix[i][num_tokens + 1] = dist_matrix[i][num_tokens] + 1 
    
    for j in range(1, num_tokens+1):
        dist_matrix[num_tokens + 1][j] = dist_matrix[num_tokens][j] - 1 
    
    dist_matrix[0][num_tokens + 1] = dist_matrix[0][num_tokens] + 1
    dist_matrix[num_tokens + 1][0] = dist_matrix[num_tokens][0] + 1

    decoder_dist_matrix = decoder_dist_matrix[:-1, :-1]
    encoder_dist_matrix = dist_matrix[1:, 1:]

    return encoder_dist_matrix, decoder_dist_matrix

# def triu_to_full_matrix_vectorized(triu, max_seq_len, max_pos):
#     """
#     Vectorized CPU version that is EXACTLY equivalent to the original implementation.
#     """

#     device = triu.device
#     dtype = triu.dtype

#     max_seq_len += 1  # [SOS] + tokens + [EOS]

#     # Number of tokens inferred from triu length
#     L = int((1 + math.isqrt(1 + 8 * triu.numel())) // 2)

#     # Initialize with PAD
#     dist = torch.full(
#         (max_seq_len, max_seq_len),
#         -1,
#         dtype=dtype,
#         device=device
#     )

#     # Token indices (1..L)
#     idx = torch.arange(1, L + 1, device=device)

#     # Diagonal
#     dist[idx, idx] = max_pos

#     # Upper triangle indices
#     iu, ju = torch.triu_indices(L, L, offset=1, device=device)
#     iu += 1
#     ju += 1

#     # IMPORTANT: apply +max_pos here (matches get_dist_from_triu)
#     upper = triu + max_pos

#     dist[iu, ju] = upper
#     dist[ju, iu] = 2 * max_pos - upper

#     # [SOS]
#     dist[0, 0] = max_pos
#     dist[idx, 0] = dist[idx, 1] - 1
#     dist[0, idx] = dist[1, idx] + 1

#     # Decoder matrix (before EOS)
#     decoder_dist = dist[:-1, :-1].clone()

#     # [EOS]
#     eos = L + 1
#     dist[eos, eos] = max_pos
#     dist[idx, eos] = dist[idx, L] + 1
#     dist[eos, idx] = dist[L, idx] - 1
#     dist[0, eos] = dist[0, L] + 1
#     dist[eos, 0] = dist[L, 0] + 1

#     # Encoder matrix (remove SOS)
#     encoder_dist = dist[1:, 1:]

#     return encoder_dist, decoder_dist