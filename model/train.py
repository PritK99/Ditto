import torch
import torch.nn as nn
from config import Config
from tqdm import tqdm
from dataset import get_dataloaders
from model import Ditto

class JointReconstructionLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, c_logits, cpp_logits, c_target, cpp_target):
        c_logits_flat = c_logits.view(-1, c_logits.size(-1))
        cpp_logits_flat = cpp_logits.view(-1, cpp_logits.size(-1))

        c_target_flat = c_target.view(-1)
        cpp_target_flat = cpp_target.view(-1)

        c_loss = self.ce_loss(c_logits_flat, c_target_flat)
        cpp_loss = self.ce_loss(cpp_logits_flat, cpp_target_flat)

        joint_loss = c_loss + cpp_loss
        return joint_loss

def train_one_epoch(model, train_dataloader, val_dataloader, loss_func, optimizer, device):

    model.train()
    train_loss = 0
    for c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask in tqdm(train_dataloader, desc="Training"):
        # Moving everything to device
        c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask = c_encoder_token_ids.to(device), cpp_encoder_token_ids.to(device), c_encoder_mask.to(device), cpp_encoder_mask.to(device), c_encoder_dist_matrix.to(device), c_decoder_dist_matrix.to(device), cpp_encoder_dist_matrix.to(device), cpp_decoder_dist_matrix.to(device), c_decoder_token_ids.to(device), cpp_decoder_token_ids.to(device), c_decoder_mask.to(device), cpp_decoder_mask.to(device)

        optimizer.zero_grad()
        c_out, cpp_out = ditto(c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask)
        loss = loss_func(c_out, cpp_out, c_encoder_token_ids, cpp_encoder_token_ids)
        loss.backward()
        train_loss += loss
        optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask in tqdm(val_dataloader, desc="Validating"):
            # Moving everything to device
            c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask = c_encoder_token_ids.to(device), cpp_encoder_token_ids.to(device), c_encoder_mask.to(device), cpp_encoder_mask.to(device), c_encoder_dist_matrix.to(device), c_decoder_dist_matrix.to(device), cpp_encoder_dist_matrix.to(device), cpp_decoder_dist_matrix.to(device), c_decoder_token_ids.to(device), cpp_decoder_token_ids.to(device), c_decoder_mask.to(device), cpp_decoder_mask.to(device)

            c_out, cpp_out = ditto(c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask)
            loss = loss_func(c_out, cpp_out, c_encoder_token_ids, cpp_encoder_token_ids)
            val_loss += loss
        
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)

    return train_loss, val_loss

def train(model, num_epochs, train_dataloader, val_dataloader, loss_func, optimizer, device):
    train_losses = []
    val_losses = []
    for epochs in num_epochs:
        train_loss, val_loss = train_one_epoch(model, train_dataloader, val_dataloader, loss_func, optimizer, device)
        train_losses.append(train_loss)
        val_losses.append(val_losses)
    
    return train_loss, val_loss

if __name__ == "__main__":
    config = Config()

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.c_data_path, config.cpp_data_path, config.vocab_path, config.batch_size, config.max_seq_len, config.max_pos, config.use_lca_distance, config.val_ratio, config.test_ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = JointReconstructionLoss(config.pad_idx)
    ditto = Ditto(config.d_model, config.vocab_size, config.num_encoders, config.num_decoders, config.num_heads, config.ffn_hidden_size, config.max_seq_len, config.pos_vocab_size, config.dropout)
    optimizer = torch.optim.Adam(ditto.parameters(), lr=config.lr)
    
    train(ditto, config.num_epochs, train_dataloader, val_dataloader, loss_func, optimizer, device)