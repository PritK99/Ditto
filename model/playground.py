import torch
from config import Config
from dataset import TranspilerDataset
from model import Embeddings, MultiHeadAttention

if __name__ == "__main__":
    config = Config()
    mha = MultiHeadAttention(config.d_model, config.num_heads, config.max_seq_len, config.pos_vocab_size, dropout = 0.1)

    train_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.max_pos, config.use_lca_distance, mode="train", val_ratio=config.val_ratio, test_ratio=config.test_ratio, verbose=True)
    val_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.max_pos, config.use_lca_distance, mode="val", val_ratio=config.val_ratio, test_ratio=config.test_ratio)
    test_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.max_pos, config.use_lca_distance, mode="test", val_ratio=config.val_ratio, test_ratio=config.test_ratio)

    c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_matrix, cpp_encoder_matrix, c_encoder_mask, cpp_encoder_mask = train_data[99]
    embedding_layer = Embeddings(config.d_model, config.vocab_size, 0)
    x = c_encoder_token_ids.unsqueeze(0)
    x = embedding_layer(x)
    c_encoder_matrix = c_encoder_matrix.unsqueeze(0)
    out = mha(x, c_encoder_matrix)
    print(out.shape)
