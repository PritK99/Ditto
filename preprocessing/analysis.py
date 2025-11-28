import pandas as pd
import ast
from collections import Counter, defaultdict
import numpy as np
import os

# csv_file = "../data/c_tokens.csv" 
# base_folder = "../assets/preprocessing_analysis_results/c" 
# log_path = os.path.join("../assets/preprocessing_analysis_results/c/c_analysis_log.txt") 
csv_file = "../data/cpp_tokens.csv" 
base_folder = "../assets/preprocessing_analysis_results/cpp" 
log_path = os.path.join("../assets/preprocessing_analysis_results/cpp/cpp_analysis_log.txt")

os.makedirs(base_folder, exist_ok=True)

# For C, the chunk size can be kept high. (~25K-50K for a system with 16 GB RAM) 
# # For C++, the chunk size has to be low due to the massive size of cpp_tokens.csv file. (~500 for a system with 16 GB RAM)
chunk_size = 500  

list_col = "transformed_tokens"

token_freq = Counter()
token_rows = defaultdict(int)  
token_lengths = []

i = 0
min_token_list_len = float("inf")
max_token_list_len = float("inf")
total_token_list_len = 0
for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
    for idx, row in chunk.iterrows():
        tokens = ast.literal_eval(row[list_col])

        if (len(tokens) > max_token_list_len):
            max_token_list_len = len(tokens)
        if (len(tokens) < min_token_list_len):
            min_token_list_len = len(tokens)

        total_token_list_len += len(tokens)

        seen_in_row = set()
        for t in tokens:
            token_freq[t] += 1
            token_lengths.append(len(t))
            seen_in_row.add(t)
        for t in seen_in_row:
            token_rows[t] += 1

    i += len(chunk)
    print(f"Processed {i} rows")

vocab_df = pd.DataFrame({
    "token": list(token_freq.keys()),
    "num_rows": [token_rows[t] for t in token_freq.keys()],
    "count": list(token_freq.values())
}).sort_values("num_rows", ascending=False)

vocab_csv_path = os.path.join(base_folder, "vocab_counts.csv")
vocab_df.to_csv(vocab_csv_path, index=False)
print(f"Saved vocabulary CSV to {vocab_csv_path}")

with open(log_path, "w", encoding="utf-8") as log:
    log.write(f"Max token list length: {max_token_list_len}\n")
    log.write(f"Min token list length: {min_token_list_len}\n")
    log.write(f"Avg token list length: {total_token_list_len / i}\n\n")

    log.write(f"Max token length: {max(token_lengths)}\n")
    log.write(f"Min token length: {min(token_lengths)}\n")
    log.write(f"Avg token length: {np.mean(token_lengths):.2f}\n")

    log.write(f"Total vocabulary size: {len(token_freq)}\n\n")

print(f"Saved logs to {log_path}")
