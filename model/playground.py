import torch
from config import Config
from dataset import get_dataloaders
from model import Embeddings, Encoder

if __name__ == "__main__":
    config = Config()
    encoder = Encoder(config.num_encoders, config.d_model, config.num_heads, config.ffn_hidden_size, config.max_seq_len, config.pos_vocab_size, dropout = 0.1)
    embedding_layer = Embeddings(config.d_model, config.vocab_size, 0)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config)

    for c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_matrix, cpp_encoder_matrix, c_encoder_mask, cpp_encoder_mask in train_dataloader:
        c_embeddings = embedding_layer(c_encoder_token_ids)    # Convert (batch_size, max_seq_len) to (batch_size, max_seq_len, d_model)
        c_encoder_mask = c_encoder_mask.unsqueeze(1).unsqueeze(1)

        out = encoder(c_embeddings, c_encoder_matrix, c_encoder_mask)
        print(out.shape)
