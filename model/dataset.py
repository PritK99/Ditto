import torch
from config import Config  
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from utils import triu_to_full_matrix

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

        def pad_seq(tokens, max_len):
            num_pad = max_len - len(tokens) - 2    # -2 comes as we need [SOS] and [EOS] as well
            return [self.token_to_idx["[SOS]"]] + [self.token_to_idx.get(t, self.token_to_idx["[UNK]"]) for t in tokens] + [self.token_to_idx["[EOS]"]] + [self.token_to_idx["[PAD]"]] * num_pad 

        c_tokens_ids = pad_seq(c_tokens, self.max_seq_len)
        cpp_tokens_ids = pad_seq(cpp_tokens, self.max_seq_len)

        c_encoder_token_ids = torch.tensor(c_tokens_ids, dtype=torch.long)
        cpp_encoder_token_ids = torch.tensor(cpp_tokens_ids, dtype=torch.long)
        c_encoder_dist = torch.tensor(c_dist, dtype=torch.long)
        cpp_encoder_dist = torch.tensor(cpp_dist, dtype=torch.long)

        # We also require masks
        pad_id = self.token_to_idx["[PAD]"]
        c_encoder_mask = (c_encoder_token_ids != pad_id).long()
        cpp_encoder_mask = (cpp_encoder_token_ids != pad_id).long()

        c_encoder_matrix = triu_to_full_matrix(c_encoder_dist, self.max_seq_len, self.max_pos)
        cpp_encoder_matrix = triu_to_full_matrix(cpp_encoder_dist, self.max_seq_len, self.max_pos)

        # # The same process has to be repeated for decoder side
        # def shift_right(seq_ids):
        #     seq_ids = torch.tensor(seq_ids, dtype=torch.long) if not isinstance(seq_ids, torch.Tensor) else seq_ids
        #     return torch.cat([torch.tensor([self.token_to_idx["[SOS]"]]), seq_ids[:-1]]) 

        # c_decoder_token_ids = shift_right(c_encoder_token_ids)
        # cpp_decoder_token_ids = shift_right(cpp_encoder_token_ids)

        # c_decoder_mask = (c_decoder_token_ids != pad_id).long()
        # cpp_decoder_mask = (cpp_decoder_token_ids != pad_id).long()

        # c_decoder_dist = c_encoder_dist.clone()
        # cpp_decoder_dist = cpp_encoder_dist.clone()

        return c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_matrix, cpp_encoder_matrix, c_encoder_mask, cpp_encoder_mask    # , c_decoder_token_ids, cpp_decoder_token_ids, c_decoder_dist, cpp_decoder_dist, c_decoder_mask, cpp_decoder_mask

if __name__ == "__main__":
    config = Config()

    train_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.max_pos, config.use_lca_distance, mode="train", val_ratio=config.val_ratio, test_ratio=config.test_ratio, verbose=True)
    val_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.max_pos, config.use_lca_distance, mode="val", val_ratio=config.val_ratio, test_ratio=config.test_ratio)
    test_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.max_pos, config.use_lca_distance, mode="test", val_ratio=config.val_ratio, test_ratio=config.test_ratio)

    c_encoder_token_ids, cpp_encoder_token_ids, c_encoder_matrix, cpp_encoder_matrix, c_encoder_mask, cpp_encoder_mask = train_data[99]