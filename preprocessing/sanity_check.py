import pandas as pd
import ast
from collections import Counter

# Pandas settings to view the content properly
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Loading C and C++ tokens
c_tokens = pd.read_csv("../data/c_tokens.csv")
cpp_tokens = pd.read_csv("../data/cpp_tokens.csv")

# Function to safely convert string representations of lists into actual lists
def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# Function to clean tokens, removing empty or trivial ones
def clean_tokens(tokens):
    # Filter out empty strings, spaces, or other trivial tokens
    return [token for token in tokens if token.strip() and len(token) > 1]

# Function to calculate token statistics (max, min, average length, top tokens, etc.)
def calculate_token_stats(df, token_column):
    token_lengths = []
    token_counter = Counter()

    # Ensure the tokens are evaluated into actual lists
    df[token_column] = df[token_column].apply(safe_eval)

    # Loop over each row's token list to collect lengths and counts
    for tokens in df[token_column]:
        if isinstance(tokens, list):
            # Clean tokens before processing
            cleaned_tokens = clean_tokens(tokens)
            token_lengths.extend([len(token) for token in cleaned_tokens])
            token_counter.update(cleaned_tokens)

    # Max, Min, Avg token lengths
    max_token_length = max(token_lengths) if token_lengths else 0
    min_token_length = min(token_lengths) if token_lengths else 0
    avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0

    # Sort tokens by their length (smallest first)
    sorted_tokens_by_length = sorted(token_counter.items(), key=lambda x: len(x[0]))

    # Top 5 smallest and largest tokens (sorted by length)
    top_5_smallest = sorted_tokens_by_length[:5]  # First 5 smallest tokens by length
    top_5_largest = sorted_tokens_by_length[-5:]  # Last 5 largest tokens by length

    # Vocabulary and Average appearance count
    vocab_size = len(token_counter)
    avg_token_count = sum(token_counter.values()) / vocab_size if vocab_size > 0 else 0

    return {
        "max_token_length": max_token_length,
        "min_token_length": min_token_length,
        "avg_token_length": avg_token_length,
        "top_5_smallest": top_5_smallest,
        "top_5_largest": top_5_largest,
        "vocab_size": vocab_size,
        "avg_token_count": avg_token_count,
        "token_counter": token_counter
    }

# Function to log detailed DataFrame info and additional analysis to a log file
def log_to_file(df, indices, token_column, log_filename="data_log.txt"):
    with open(log_filename, "w", encoding="utf-8") as log_file:
        # General DataFrame Info
        log_file.write("General DataFrame Info:\n")
        log_file.write(f"Number of rows: {df.shape[0]}\n")
        log_file.write(f"Number of columns: {df.shape[1]}\n")
        log_file.write(f"Column names: {', '.join(df.columns)}\n")
        log_file.write(f"NULLs:\n{df.isnull().sum()}\n")
        log_file.write(f"Unique values:\n{df.nunique()}\n")
        log_file.write("\n" + "="*50 + "\n\n")

        # Token Analysis (max, min, avg length, top tokens, vocab size, avg appearance)
        token_stats = calculate_token_stats(df, token_column)

        log_file.write("Token Analysis:\n")
        log_file.write(f"  - Max Token Length: {token_stats['max_token_length']}\n")
        log_file.write(f"  - Min Token Length: {token_stats['min_token_length']}\n")
        log_file.write(f"  - Average Token Length: {token_stats['avg_token_length']:.2f}\n")
        
        # Log Top 5 Smallest Tokens
        log_file.write(f"  - Top 5 Smallest Tokens (by length):\n")
        for token, _ in token_stats['top_5_smallest']:
            log_file.write(f"    - {token} (Length: {len(token)})\n")
        
        # Log Top 5 Largest Tokens
        log_file.write(f"  - Top 5 Largest Tokens (by length):\n")
        for token, _ in token_stats['top_5_largest']:
            log_file.write(f"    - {token} (Length: {len(token)})\n")

        log_file.write(f"  - Vocabulary Size: {token_stats['vocab_size']}\n")
        log_file.write(f"  - Average Token Appearance Count: {token_stats['avg_token_count']:.2f}\n")

        # Log all tokens and their frequency (vocabulary)
        log_file.write("\nVocabulary (Token Frequencies):\n")
        for token, count in token_stats['token_counter'].items():
            log_file.write(f"  - {token}: {count}\n")

        log_file.write("\n" + "="*50 + "\n\n")

        # For each sample index, we will log all the details
        for index in indices:
            if index in df.index:
                log_file.write(f"Index {index}:\n")
                for col in df.columns:
                    log_file.write(f"  {col}: {df.at[index, col]}\n")
                log_file.write("\n" + "-"*50 + "\n")
            else:
                log_file.write(f"Index {index} not found in the DataFrame.\n")
                log_file.write("-"*50 + "\n")

    print(f"Log written to {log_filename}")

indices_to_check = [0, 10, 200, 1000]

token_column_c = "Obfuscated_Tokens"  # For c_tokens
token_column_cpp = "Obfuscated_Tokens"  # For cpp_tokens

# Call the function to log info to file for both C and C++ datasets
log_to_file(c_tokens, indices_to_check, token_column_c, log_filename="c_tokens_log.txt")
log_to_file(cpp_tokens, indices_to_check, token_column_cpp, log_filename="cpp_tokens_log.txt")
