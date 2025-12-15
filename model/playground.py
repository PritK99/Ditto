from config import Config
from dataset import TranspilerDataset
from utils import triu_to_full_matrix

if __name__ == "__main__":
    config = Config()

    train_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.use_lca_distance, mode="train", val_ratio=config.val_ratio, test_ratio=config.test_ratio, verbose=True)
    val_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.use_lca_distance, mode="val", val_ratio=config.val_ratio, test_ratio=config.test_ratio)
    test_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.use_lca_distance, mode="test", val_ratio=config.val_ratio, test_ratio=config.test_ratio)

    c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_dist, cpp_encoder_dist, c_encoder_mask, cpp_encoder_mask, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_dist, cpp_decoder_dist, c_decoder_mask, cpp_decoder_mask = train_data[99]
    
    c_matrix = triu_to_full_matrix(c_encoder_dist, 1002)

    with open("results.txt", "w") as f:
        f.write(f"C encoder dist: {c_encoder_dist.tolist()}\n")
        f.write(f"C Matrix: {c_matrix}\n")
