import csv
import ast
import torch
import numpy as np
import pandas as pd
from config import Config
from torch.utils.data import Dataset, DataLoader

class TranspilerDataset(Dataset):
    def __init__(self, c_data_path: str, cpp_data_path: str, vocab_path: str, max_seq_len: int = 1000, use_lca_distance: bool = False, mode: str = "train", val_ratio: float = 0.05, test_ratio: float = 0.05, verbose = False):
        super().__init__()
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.use_lca_distance = use_lca_distance
        self.c_data_path = c_data_path
        self.cpp_data_path = cpp_data_path
        self.header = None

        # Calculating length of c_data and cpp_data by reading chunks 
        # This is because we can not load the complete dataset at once
        c_len = 0
        cpp_len = 0

        chunksize = 1000
        for chunk in pd.read_csv(c_data_path, chunksize=chunksize):
            c_len += len(chunk)
        for chunk in pd.read_csv(cpp_data_path, chunksize=chunksize):
            cpp_len += len(chunk)

        if (verbose):
            print(f"Original length of C data: {c_len}", flush = False)
            print(f"Original length of C++ data: {cpp_len}", flush = False)
        
        # We need to make both datasets of same size
        self.dataset_len = min(c_len, cpp_len)
        if (verbose):
            print(f"Transpiler dataset length: {self.dataset_len}", flush = False)

        # Calculating split indexes for val and test
        self.val_start = int((1 - val_ratio - test_ratio)*self.dataset_len)    # val_start will also act as train_end
        self.test_start = int((1 - test_ratio)*self.dataset_len)    # test_start will also act as test_end

        if (verbose):
            print(f"Train data length: {self.val_start}", flush = False)
            print(f"Validation data length: {self.test_start - self.val_start}", flush = False)
            print(f"Train data length: {self.dataset_len - self.test_start}", flush = False)
        
        # Computing offsets for easy access later
        self.c_offsets = self.build_offsets(c_data_path)
        self.cpp_offsets = self.build_offsets(cpp_data_path)

        # Creating vocabulary and token_to_idx and idx_to_token dictionary
        self.token_to_idx, self.idx_to_token = self.build_vocab(vocab_path)
    
    def build_offsets(self, file_path):
        offsets = []
        offset = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                offsets.append(offset)
                offset += len(line.encode('utf-8'))
        return offsets

    def build_vocab(self, vocab_path):
        token_to_idx = {}
        idx_to_token = {}

        special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]    # Reserving indices for special tokens 

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
    
    def __len__(self):
        return self.dataset_len
    
    def get_vanilla_relative_bias_triu(self, n: int):
        triu = []
        for i in range(n):
            for j in range(i+1, n):
                triu.append(j - i)

        return triu

    def __getitem__(self, index):
        if (self.header == None):
            with open(self.c_data_path, 'r', encoding='utf-8') as f:
                self.header = f.readline().strip().split(",")

        if (self.mode == "train"):
            offset = index + 1    # Because we use linecache, we need to add 1 to skip header 
        elif (self.mode == "val"):
            offset = self.val_start + index + 1  
        elif (self.mode == "test"):
            offset = self.test_start + index + 1 
        
        with open(self.c_data_path, 'r', encoding='utf-8') as f:
            f.seek(self.c_offsets[offset])
            c_line = f.readline().strip()

        with open(self.cpp_data_path, 'r', encoding='utf-8') as f:
            f.seek(self.cpp_offsets[offset])
            cpp_line = f.readline().strip()

        c_row = next(csv.reader([c_line]))
        cpp_row = next(csv.reader([cpp_line]))

        # Extracting all required columns from row
        # Note that for training we do not require obfuscation dictionaries
        # These dictionaries will only be used for inference in real time to provide accuracte output
        c_tokens = ast.literal_eval(c_row[1])
        # c_var_dict = ast.literal_eval(c_row[2])
        # c_func_dict = ast.literal_eval(c_row[3])
        # c_lit_dict = ast.literal_eval(c_row[4])
        # c_struct_dict = ast.literal_eval(c_row[5])
        # c_class_dict = ast.literal_eval(c_row[6])

        if (self.use_lca_distance):
            c_dist_vector = ast.literal_eval(c_row[7])
        else:
            c_dist_vector = self.get_vanilla_relative_bias_triu(len(c_tokens))

        cpp_tokens = ast.literal_eval(cpp_row[1])
        # cpp_var_dict = ast.literal_eval(cpp_row[2])
        # cpp_func_dict = ast.literal_eval(cpp_row[3])
        # cpp_lit_dict = ast.literal_eval(cpp_row[4])
        # cpp_struct_dict = ast.literal_eval(cpp_row[5])
        # cpp_class_dict = ast.literal_eval(cpp_row[6])

        if (self.use_lca_distance):
            cpp_dist_vector = ast.literal_eval(cpp_row[7])
        else:
            cpp_dist_vector = self.get_vanilla_relative_bias_triu(len(cpp_tokens))

        c_tokens_ids = [self.token_to_idx[t] for t in c_tokens]
        cpp_tokens_ids = [self.token_to_idx[t] for t in cpp_tokens]

        # For now, we dont consider max_len, PAD, EOS, SOS etc.
        # Pending

        return torch.tensor(c_tokens_ids, dtype=torch.long), torch.tensor(cpp_tokens_ids, dtype=torch.long), torch.tensor(c_dist_vector, dtype=torch.long), torch.tensor(cpp_dist_vector, dtype=torch.long)

if __name__ == "__main__":
    config = Config()

    train_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.use_lca_distance, "train", config.val_ratio, config.test_ratio, verbose = True)
    val_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.use_lca_distance, "val", config.val_ratio, config.test_ratio)
    test_data = TranspilerDataset(config.c_data_path, config.cpp_data_path, config.vocab_path, config.max_seq_len, config.use_lca_distance, "test", config.val_ratio, config.test_ratio)

    c_tokens, cpp_tokens, c_dist_vector, cpp_dist_vector  = train_data.__getitem__(0)
    print(c_tokens)
    print(c_dist_vector)
    print(cpp_tokens)
    print(cpp_dist_vector)