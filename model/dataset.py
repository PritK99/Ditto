import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from utils import triu_to_full_matrix    #, triu_to_full_matrix_vectorized

class TranspilerDataset(Dataset):
    def __init__(self, c_data_path, cpp_data_path, vocab_path, max_seq_len, max_pos, use_lca_distance=False, mode="train", val_ratio=0.05, test_ratio=0.05, verbose=False):
        super().__init__()
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.max_pos = max_pos
        self.use_lca_distance = use_lca_distance

        self.c_file = pq.ParquetFile(c_data_path)
        self.cpp_file = pq.ParquetFile(cpp_data_path)

        self.c_len = self.c_file.metadata.num_rows
        self.cpp_len = self.cpp_file.metadata.num_rows
        self.dataset_len = min(self.c_len, self.cpp_len)    # We need to make both datasets of same size

        if verbose:
            print(f"Original length of C data: {self.c_len}", flush = False)
            print(f"Original length of C++ data: {self.cpp_len}", flush = False)
            print(f"Transpiler dataset length: {self.dataset_len}", flush = False)

        self.val_start = int((1 - val_ratio - test_ratio) * self.dataset_len)
        self.test_start = int((1 - test_ratio) * self.dataset_len)

        if verbose:
            print(f"Train data length: {self.val_start}", flush = False)
            print(f"Validation data length: {self.test_start - self.val_start}", flush = False)
            print(f"Train data length: {self.dataset_len - self.test_start}", flush = False)

        # Creating vocabulary and token_to_idx and idx_to_token dictionary
        self.token_to_idx, self.idx_to_token = self.build_vocab(vocab_path)

    def build_vocab(self, vocab_path):
        token_to_idx = {}
        idx_to_token = {}
        special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        for i, tok in enumerate(special_tokens):
            token_to_idx[tok] = i
            idx_to_token[i] = tok
        idx = len(special_tokens)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                tok = line.strip()
                if tok == "" or tok in token_to_idx:
                    continue
                token_to_idx[tok] = idx
                idx_to_token[idx] = tok
                idx += 1
        return token_to_idx, idx_to_token

    def get_vanilla_relative_bias_triu(self, n):
        return [j - i for i in range(n) for j in range(i+1, n)]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.mode == "train":
            idx = index
        elif self.mode == "val":
            idx = self.val_start + index
        else: 
            idx = self.test_start + index

        c_row_group, c_local_idx = divmod(idx, self.c_file.metadata.row_group(0).num_rows)
        cpp_row_group, cpp_local_idx = divmod(idx, self.cpp_file.metadata.row_group(0).num_rows)

        c_row = self.c_file.read_row_group(c_row_group, columns=["transformed_tokens","dist"]).to_pandas().iloc[c_local_idx]
        cpp_row = self.cpp_file.read_row_group(cpp_row_group, columns=["transformed_tokens","dist"]).to_pandas().iloc[cpp_local_idx]

        c_tokens = c_row["transformed_tokens"]
        cpp_tokens = cpp_row["transformed_tokens"]

        if self.use_lca_distance:
            c_dist = c_row["dist"]
            cpp_dist = cpp_row["dist"]
        else:    # This is the baseline approach
            c_dist = self.get_vanilla_relative_bias_triu(len(c_tokens))
            cpp_dist = self.get_vanilla_relative_bias_triu(len(cpp_tokens))
        
        c_dist = torch.tensor(c_dist, dtype=torch.long)
        cpp_dist = torch.tensor(cpp_dist, dtype=torch.long)
        
        def get_encoder_input(tokens, max_len):
            num_pad = max_len - len(tokens) - 1  # only [EOS] is appended to encoder input
            return [self.token_to_idx.get(t, self.token_to_idx["[UNK]"]) for t in tokens] + [self.token_to_idx["[EOS]"]] + [self.token_to_idx["[PAD]"]] * num_pad
            # For example, [Hi, from, Ditto, [EOS]]
        
        def get_decoder_input(tokens, max_len):
            num_pad = max_len - (len(tokens) -1) - 1  # only [SOS] is appended to decoder input. Also, len(tokens) - 1 is done because we perform right shift
            return [self.token_to_idx["[SOS]"]] + [self.token_to_idx.get(t, self.token_to_idx["[UNK]"]) for t in tokens][:-1] + [self.token_to_idx["[PAD]"]] * num_pad
            # For example, [[SOS], Hi, from, Ditto]
        
        # Preparing distance matrix for encoder and decoder
        c_encoder_dist_matrix, c_decoder_dist_matrix = triu_to_full_matrix(c_dist, self.max_seq_len, self.max_pos)
        cpp_encoder_dist_matrix, cpp_decoder_dist_matrix = triu_to_full_matrix(cpp_dist, self.max_seq_len, self.max_pos)

        # # Trying some optimizations
        # c_encoder_dist_matrix_vec, c_decoder_dist_matrix_vec = triu_to_full_matrix_vectorized(c_dist, self.max_seq_len, self.max_pos)
        # cpp_encoder_dist_matrix_vec, cpp_decoder_dist_matrix_vec = triu_to_full_matrix_vectorized(cpp_dist, self.max_seq_len, self.max_pos)

        # if not torch.equal(c_encoder_dist_matrix, c_encoder_dist_matrix_vec):
        #     diff = (c_encoder_dist_matrix != c_encoder_dist_matrix_vec).nonzero(as_tuple=False)
        #     i, j = diff[0].tolist()
        #     print(
        #         "[ENCODER MISMATCH]",
        #         f"at ({i}, {j}):",
        #         f"ref={c_encoder_dist_matrix_vec[i, j].item()},",
        #         f"vec={c_encoder_dist_matrix[i, j].item()}",
        #         flush=True
        #     )
        # if not torch.equal(c_decoder_dist_matrix, c_decoder_dist_matrix_vec):
        #     diff = (c_decoder_dist_matrix != c_decoder_dist_matrix_vec).nonzero(as_tuple=False)
        #     i, j = diff[0].tolist()
        #     print(
        #         "[DECODER MISMATCH]",
        #         f"at ({i}, {j}):",
        #         f"ref={c_decoder_dist_matrix[i, j].item()},",
        #         f"vec={c_decoder_dist_matrix_vec[i, j].item()}",
        #         flush=True
        #     )

        # Preparing encoder and decoder inputs
        # Decoder target is nothing but encoder input
        c_encoder_token_ids = get_encoder_input(c_tokens, self.max_seq_len)
        cpp_encoder_token_ids = get_encoder_input(cpp_tokens, self.max_seq_len)
        c_encoder_token_ids = torch.tensor(c_encoder_token_ids, dtype=torch.long)
        cpp_encoder_token_ids = torch.tensor(cpp_encoder_token_ids, dtype=torch.long)

        c_decoder_token_ids = get_decoder_input(c_tokens, self.max_seq_len)
        cpp_decoder_token_ids = get_decoder_input(cpp_tokens, self.max_seq_len)
        c_decoder_token_ids = torch.tensor(c_decoder_token_ids, dtype=torch.long)
        cpp_decoder_token_ids = torch.tensor(cpp_decoder_token_ids, dtype=torch.long)

        # Creating masks
        pad_id = self.token_to_idx["[PAD]"]
        c_encoder_mask = (c_encoder_token_ids != pad_id).long()
        cpp_encoder_mask = (cpp_encoder_token_ids != pad_id).long()

        def causal_mask(seq_len, device):
            """
            Returns a boolean causal mask of shape (seq_len, seq_len) where True means attention is allowed.
            """
            return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        c_decoder_pad_mask = (c_decoder_token_ids != pad_id).long()
        cpp_decoder_pad_mask = (cpp_decoder_token_ids != pad_id).long()
        causal = causal_mask(self.max_seq_len, device=c_decoder_token_ids.device)
        
        c_pad = c_decoder_pad_mask.unsqueeze(0)
        cpp_pad = cpp_decoder_pad_mask.unsqueeze(0)

        c_decoder_mask = (causal.unsqueeze(0) & c_pad.unsqueeze(1))
        cpp_decoder_mask = (causal.unsqueeze(0) & cpp_pad.unsqueeze(1))                           

        return c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_mask, cpp_encoder_mask, c_encoder_dist_matrix, c_decoder_dist_matrix, cpp_encoder_dist_matrix, cpp_decoder_dist_matrix, c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_mask, cpp_decoder_mask
    
def get_dataloaders(c_data_path, cpp_data_path, vocab_path, batch_size, max_seq_len, max_pos, use_lca_distance, val_ratio, test_ratio):
    train_data = TranspilerDataset(c_data_path, cpp_data_path, vocab_path, max_seq_len, max_pos, use_lca_distance, mode="train", val_ratio=val_ratio, test_ratio=test_ratio, verbose=True)
    val_data = TranspilerDataset(c_data_path, cpp_data_path, vocab_path, max_seq_len, max_pos, use_lca_distance, mode="val", val_ratio=val_ratio, test_ratio=test_ratio)
    test_data = TranspilerDataset(c_data_path, cpp_data_path, vocab_path, max_seq_len, max_pos, use_lca_distance, mode="test", val_ratio=val_ratio, test_ratio=test_ratio)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader