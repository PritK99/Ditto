import math
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import triu_to_full_matrix
from config import Config

def build_vocab(vocab_path):
    token_to_idx = {}
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    for i, tok in enumerate(special_tokens):
        token_to_idx[tok] = i

    idx = len(special_tokens)
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok and tok not in token_to_idx:
                token_to_idx[tok] = idx
                idx += 1
    return token_to_idx


def process_index(args):
    idx, c_path, cpp_path, vocab, max_seq_len, max_pos, use_lca = args

    c_file = pq.ParquetFile(c_path)
    cpp_file = pq.ParquetFile(cpp_path)

    rg_size = c_file.metadata.row_group(0).num_rows
    c_rg, c_li = divmod(idx, rg_size)
    cpp_rg, cpp_li = divmod(idx, rg_size)

    c_row = c_file.read_row_group(c_rg, columns=["transformed_tokens", "dist"]).to_pandas().iloc[c_li]
    cpp_row = cpp_file.read_row_group(cpp_rg, columns=["transformed_tokens", "dist"]).to_pandas().iloc[cpp_li]

    c_tokens = c_row["transformed_tokens"]
    cpp_tokens = cpp_row["transformed_tokens"]

    if use_lca:
        c_dist = c_row["dist"]
        cpp_dist = cpp_row["dist"]
    else:
        c_dist = [j - i for i in range(len(c_tokens)) for j in range(i+1, len(c_tokens))]
        cpp_dist = [j - i for i in range(len(cpp_tokens)) for j in range(i+1, len(cpp_tokens))]

    num_c = int((1 + math.isqrt(1 + 8*len(c_dist))) // 2)
    num_cpp = int((1 + math.isqrt(1 + 8*len(cpp_dist))) // 2)

    if len(c_tokens) != num_c or len(cpp_tokens) != num_cpp:
        return ("BAD", idx)

    # tensors
    c_dist = torch.tensor(c_dist)
    cpp_dist = torch.tensor(cpp_dist)

    c_enc_dm, c_dec_dm = triu_to_full_matrix(c_dist, max_seq_len, max_pos)
    cpp_enc_dm, cpp_dec_dm = triu_to_full_matrix(cpp_dist, max_seq_len, max_pos)

    pad = vocab["[PAD]"]
    eos = vocab["[EOS]"]
    sos = vocab["[SOS]"]

    def enc(tokens):
        return [vocab.get(t, vocab["[UNK]"]) for t in tokens] + [eos] + [pad]*(max_seq_len-len(tokens)-1)

    def dec(tokens):
        return [sos] + [vocab.get(t, vocab["[UNK]"]) for t in tokens[:-1]] + [pad]*(max_seq_len-len(tokens))

    c_enc_ids = torch.tensor(enc(c_tokens))
    cpp_enc_ids = torch.tensor(enc(cpp_tokens))
    c_dec_ids = torch.tensor(dec(c_tokens))
    cpp_dec_ids = torch.tensor(dec(cpp_tokens))

    c_enc_mask = (c_enc_ids != pad).long()
    cpp_enc_mask = (cpp_enc_ids != pad).long()

    causal = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
    c_dec_mask = (causal.unsqueeze(0) & (c_dec_ids != pad).unsqueeze(0).unsqueeze(1))
    cpp_dec_mask = (causal.unsqueeze(0) & (cpp_dec_ids != pad).unsqueeze(0).unsqueeze(1))

    return ("OK", {
        "c_encoder_token_ids": c_enc_ids.tolist(),
        "cpp_encoder_token_ids": cpp_enc_ids.tolist(),
        "c_encoder_mask": c_enc_mask.tolist(),
        "cpp_encoder_mask": cpp_enc_mask.tolist(),
        "c_encoder_dist_matrix": c_enc_dm.tolist(),
        "c_decoder_dist_matrix": c_dec_dm.tolist(),
        "cpp_encoder_dist_matrix": cpp_enc_dm.tolist(),
        "cpp_decoder_dist_matrix": cpp_dec_dm.tolist(),
        "c_decoder_token_ids": c_dec_ids.tolist(),
        "cpp_decoder_token_ids": cpp_dec_ids.tolist(),
        "c_decoder_mask": c_dec_mask.tolist(),
        "cpp_decoder_mask": cpp_dec_mask.tolist(),
    })


def build_new_parquet(cfg):
    c_file = pq.ParquetFile(cfg.c_data_path)
    dataset_len = c_file.metadata.num_rows
    vocab = build_vocab(cfg.vocab_path)

    args = [
        (i, cfg.c_data_path, cfg.cpp_data_path, vocab,
         cfg.max_seq_len, cfg.max_pos, cfg.use_lca_distance)
        for i in range(dataset_len)
    ]

    good_rows = []
    bad = []

    with Pool(cpu_count()) as p:
        for status, result in tqdm(p.imap(process_index, args), total=dataset_len):
            if status == "OK":
                good_rows.append(result)
            else:
                bad.append(result)

    if bad:
        with open("bad_indices.txt", "w") as f:
            for i in bad:
                f.write(f"{i}\n")

    table = pa.Table.from_pylist(good_rows)
    pq.write_table(table, "precomputed_dataset.parquet")


if __name__ == "__main__":
    cfg = Config()
    build_new_parquet(cfg)
