"""
This script takes the obfuscted tokens (c_tokens.csv and cpp_tokens.csv) and converts them to analyzed tokens (c_tokens_analyzed.csv and cpp_tokens_analyzed.csv).
Analysis eliminates the code lines which are too lengthy.
"""
import pandas as pd
import ast
from collections import Counter, defaultdict
import os

csv_file = "../data/c_tokens.csv" 
base_folder = "../assets/preprocessing_logs/" 
log_path = os.path.join(base_folder, "c_analysis_log.txt")
analyzed_csv_path = "../data/c_tokens_analyzed.csv"
# csv_file = "../data/cpp_tokens.csv" 
# base_folder = "../assets/preprocessing_logs/" 
# log_path = os.path.join(base_folder, "cpp_analysis_log.txt")
# analyzed_csv_path = "../data/cpp_tokens_analyzed.csv"

os.makedirs(base_folder, exist_ok=True)
token_threshold = 1000
list_col = "transformed_tokens"

# For C, the chunk size can be kept high. (~25K for a system with 16 GB RAM) 
# # For C++, the chunk size has to be low due to the massive size of cpp_tokens.csv file. (~100 for a system with 16 GB RAM)
chunk_size = 1000 

i = 0
min_token_list_len = float("inf")
max_token_list_len = float("-inf")
total_token_list_len = 0
token_freq = Counter()
token_rows = defaultdict(int)  
discarded = []
first_chunk = True
for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
    good_rows = []    # These rows will be part of data/c_tokens_analyzed.csv or data/cpp_tokens_analyzed.csv
    for idx, row in chunk.iterrows():
        tokens = ast.literal_eval(row[list_col])

        # Discarding all lengthy rows
        if (len(tokens) > token_threshold):
            discarded.append(i)   

        else:
            # Computing min, max and mean for valid rows
            if (len(tokens) > max_token_list_len):
                max_token_list_len = len(tokens)
            if (len(tokens) < min_token_list_len):
                min_token_list_len = len(tokens)

            total_token_list_len += len(tokens)

            seen_in_row = set()
            for t in tokens:
                token_freq[t] += 1
                seen_in_row.add(t)
            for t in seen_in_row:
                token_rows[t] += 1
            
            good_rows.append(row)

        i += 1
        if (i % 10000 == 0):
            print(f"Processed {i} rows with {len(discarded)} discarded.")
    
    if good_rows:
        df_out = pd.DataFrame(good_rows)
        df_out.to_csv(
            analyzed_csv_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )
        first_chunk = False

# Resetting index of analyzed csv
df = pd.read_csv(analyzed_csv_path)
df.reset_index(drop=True).to_csv(analyzed_csv_path, index=False)
print(f"Saved analyzed CSV to {analyzed_csv_path}")
          
vocab_df = pd.DataFrame({
    "token": list(token_freq.keys()),
    "num_rows": [token_rows[t] for t in token_freq.keys()],
    "count": list(token_freq.values())
}).sort_values("num_rows", ascending=False)

# vocab_csv_path = os.path.join("../data/", "cpp_vocab_raw.csv")
vocab_csv_path = os.path.join("../data/", "c_vocab_raw.csv")
vocab_df.to_csv(vocab_csv_path, index=False)
print(f"Saved vocabulary CSV to {vocab_csv_path}")

with open(log_path, "w", encoding="utf-8") as log:
    log.write(f"Max token list length: {max_token_list_len}\n")
    log.write(f"Min token list length: {min_token_list_len}\n")
    log.write(f"Avg token list length: {total_token_list_len / i}\n\n")

    log.write(f"Total vocabulary size: {len(token_freq)}\n\n")

    # Add discarded row details
    log.write(f"Total discarded rows: {len(discarded)}\n")
    log.write("Discarded row indices:\n")
    for d in discarded:
        log.write(f"{d}\n")

print(f"Saved logs to {log_path}")
