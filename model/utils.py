import torch
import math

def get_dist_from_triu(i, j, triu, max_dist = 543):
    # Base Case
    if (i == j):
        return 0    
    
    sign = 1

    if i > j:
        i, j = j, i    # Only using the upper triangular matrix
        sign = -1

    num_tokens = int((1 + math.isqrt(1 + 8*len(triu))) // 2)    # This is formula for num_tokens from triu length

    idx = (i * (2*num_tokens - i - 1)) // 2 + (j - i - 1)

    return sign*(triu[idx]) + max_dist


def triu_to_full_matrix(triu, max_seq_len, max_dist = 543):
    num_tokens = int((1 + math.isqrt(1 + 8*len(triu))) // 2)

    dist_matrix = torch.zeros((num_tokens, num_tokens), dtype=triu.dtype)    # Our distance matrix needs to be of (1002, 1002)

    for i in range(num_tokens):
        for j in range(num_tokens):
            dist_matrix[i][j] = get_dist_from_triu(i, j, triu, max_dist)

    return dist_matrix