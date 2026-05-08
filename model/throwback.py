import pyarrow.parquet as pq

def log_token_lengths(parquet_path, output_path, token_column="transformed_tokens"):
    """
    Logs token lengths from a Parquet file.

    Each line in output file:
    code_index -> num_tokens
    """
    pq_file = pq.ParquetFile(parquet_path)
    code_idx = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for rg in range(pq_file.num_row_groups):
            table = pq_file.read_row_group(rg, columns=[token_column]).to_pandas()
            for tokens in table[token_column]:
                if (len(tokens) > 1000):
                    f.write(f"{code_idx} -> {len(tokens)}\n")
                code_idx += 1


if __name__ == "__main__":
    c_data_path = "../data/c_tokens_with_lca_dist.parquet"
    cpp_data_path = "../data/cpp_tokens_with_lca_dist.parquet"

    log_token_lengths(
        parquet_path=c_data_path,
        output_path="c_token_lengths.txt"
    )

    log_token_lengths(
        parquet_path=cpp_data_path,
        output_path="cpp_token_lengths.txt"
    )
