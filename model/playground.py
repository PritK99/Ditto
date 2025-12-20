import torch
from config import Config
from tqdm import tqdm
from dataset import get_dataloaders
from model import Ditto

if __name__ == "__main__":
    config = Config()
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.c_data_path, config.cpp_data_path, config.vocab_path, config.batch_size, config.max_seq_len, config.max_pos, config.use_lca_distance, config.val_ratio, config.test_ratio)

    # Initializing ditto
    ditto = Ditto(config.d_model, config.vocab_size, config.num_encoders, config.num_decoders, config.num_heads, config.ffn_hidden_size, config.max_seq_len, config.pos_vocab_size, config.dropout)
    
    for c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask in tqdm(train_dataloader):
        
        c_out, cpp_out = ditto(c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask)

        print(c_out.shape, cpp_out.shape)
