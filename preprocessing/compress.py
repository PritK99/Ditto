# For model, converting csv to parquet can lead to increase in speed
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ast
from tqdm import tqdm
import subprocess

CSV_PATH = "../data/cpp_tokens_with_lca_dist.csv"    # "../data/c_tokens_with_lca_dist.csv"
PARQUET_PATH = "../data/cpp_tokens_with_lca_dist.parquet"    # "../data/c_tokens_with_lca_dist.parquet"

CHUNK_SIZE = 500     
COMPRESSION = "zstd"    

total_rows = int(subprocess.check_output(["wc", "-l", CSV_PATH]).split()[0]) - 1
total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE 

parquet_writer = None

# Note that for training we do not require obfuscation dictionaries
# These dictionaries will only be used for inference in real time to provide accuracte output
for chunk in tqdm(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, usecols=["transformed_tokens", "dist"]), total=total_chunks,desc="Converting CSV to Parquet"):
    chunk["transformed_tokens"] = chunk["transformed_tokens"].apply(ast.literal_eval)
    chunk["dist"] = chunk["dist"].apply(ast.literal_eval)

    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(
            PARQUET_PATH,
            table.schema,
            compression=COMPRESSION
        )

    parquet_writer.write_table(table)

parquet_writer.close()
