import os
import ast
from tqdm import tqdm
import pandas as pd

def get_vocab(c_df_path, cpp_df_path, num_codes_threshold=100):
    c_df = pd.read_csv(c_df_path)
    cpp_df = pd.read_csv(cpp_df_path)

    combined_df = pd.concat([c_df, cpp_df], axis=0)
    combined_df = combined_df.groupby('token', as_index=False).sum()
    print("Combined df size:", len(combined_df))

    obfuscation_patterns = ['^func', '^var', '^lit', '^class', '^struct']
    obfuscation_mask = combined_df['token'].str.contains('|'.join(obfuscation_patterns))

    # To be accepted, a token must have either appeared in atleast num_codes_threshold code snippets or belong to obfuscation pattern
    filtered_df = combined_df[(combined_df['num_rows'] >= num_codes_threshold) | obfuscation_mask ]   

    tokens_list = filtered_df['token'].unique()
    print(f"Vocab size after filtering: ",len(tokens_list))

    with open("../data/final_vocab_combined.txt", "w", encoding="utf-8") as f:
        for token in tokens_list:
            f.write(f"{token}\n")
    print(f"Saved final combined vocab to ../data/final_vocab_combined.txt")

    return tokens_list

def get_cleaned_data(df_path, vocab, threshold=0.02, lang="c"):
    df = pd.read_csv(df_path)
    cleaned_df = pd.DataFrame(columns=df.columns)
    log_path = os.path.join("../data", f"{lang}_cleaning_log.txt")

    dict_list = ['var_dict', 'func_dict', 'lit_dict', 'struct_dict', 'class_dict']
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cleaned_tokens = []
        num_unks = 0

        tokens = ast.literal_eval(row["transformed_tokens"])

        obfuscation_list = []
        for dict in dict_list:
            curr_dict = ast.literal_eval(row[dict])
            for key in curr_dict.keys():
                obfuscation_list.append(curr_dict[key])
            
        for token in tokens:
            if (token in vocab) or (token in obfuscation_list):
                cleaned_tokens.append(token)
            else:
                num_unks += 1
                cleaned_tokens.append("[UNK]")
        
        # If [UNK] makes more than 2% of the code
        if ((num_unks / len(tokens)) > threshold):
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"Skipping code line {row['line_number']}\n")
        else:
            cleaned_df.loc[len(cleaned_df)] = row
            cleaned_df.loc[len(cleaned_df) - 1, "transformed_tokens"] = str(cleaned_tokens)

    cleaned_df.to_csv(f"../data/{lang}_cleaned_tokens.csv", index=False)
    print(f"Saved cleaned data to ../data/{lang}_cleaned_tokens.csv")
    print(f"Logs saved to {log_path}")


if __name__ == "__main__":
    num_codes_threshold = 100
    c_raw_vocab_df_path = "../data/c_vocab_raw.csv"
    cpp_raw_vocab_df_path = "../data/cpp_vocab_raw.csv"
    c_df_path = "../data/c_tokens_analyzed.csv"
    cpp_df_path = "../data/cpp_tokens_analyzed.csv"

    vocab = get_vocab(c_raw_vocab_df_path, cpp_raw_vocab_df_path, num_codes_threshold)
    get_cleaned_data(c_df_path, vocab, 0.02, "c")
    get_cleaned_data(cpp_df_path, vocab, 0.02, "cpp")
    