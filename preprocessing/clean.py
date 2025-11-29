import pandas as pd

num_codes_threshold = 100
count_threshold = 100

# Load both vocab files
df1 = pd.read_csv("../data/c_vocab_raw.csv")
df2 = pd.read_csv("../data/cpp_vocab_raw.csv")

combined_df = pd.concat([df1, df2], axis=0)
combined_df = combined_df.groupby('token', as_index=False).sum()
print("Combined vocab size:", len(combined_df))

obfuscation_patterns = ['^func', '^var', '^lit', '^class', '^struct']
obfuscation_mask = combined_df['token'].str.contains('|'.join(obfuscation_patterns))

# To be accepted, a token must have either appeared in atleast num_codes_threshold code snippets, or atleast aqppear count_threshold times, or belong to obfuscation pattern
filtered_df = combined_df[(combined_df['num_rows'] >= num_codes_threshold) | obfuscation_mask | (combined_df['count'] >= count_threshold)]   

print(f"Combined vocab size after filtering: ",len(filtered_df))

tokens_list = filtered_df['token'].unique()
tokens_list_sorted = sorted(tokens_list)

with open("../data/final_vocab_combined.txt", "w", encoding="utf-8") as f:
    for token in tokens_list_sorted:
        f.write(f"{token}\n")

print(f"Saved final combined vocab ({len(tokens_list_sorted)} tokens) to final_vocab_combined.txt")
