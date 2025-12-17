# This script tells us the max positional distance in final dataset
# We require this in model to define learned relative bias 
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

c_file_path = "../data/c_tokens_with_lca_dist.parquet"
cpp_file_path = "../data/cpp_tokens_with_lca_dist.parquet"

def max_lca_distance(parquet_file):
    pf = pq.ParquetFile(parquet_file)
    max_dist = None
    total_rows_processed = 0

    print(f"Processing {parquet_file} ...")
    
    # Iterate over row groups with tqdm
    for rg in tqdm(range(pf.num_row_groups), desc="Row Groups"):
        table = pf.read_row_group(rg, columns=["dist"]).to_pandas()
        for i, dist_list in enumerate(table["dist"]):
            row_max = max(dist_list) if len(dist_list) > 0 else float('-inf')
            if max_dist is None:
                max_dist = row_max
            else:
                max_dist = max(max_dist, row_max)
            
            total_rows_processed += 1
            # if total_rows_processed % 10000 == 0:
            #     print(f"Processed {total_rows_processed} rows, current max_dist = {max_dist}")

    return max_dist

max_c = max_lca_distance(c_file_path)
max_cpp = max_lca_distance(cpp_file_path)

overall_max = max(max_c, max_cpp)
print("\nMaximum LCA distance in dataset:", overall_max)    # 271

# Hence, relative bias will take [-273, 273] i.e. 545 distinct values.
# Note, 271 becomes 273 because of the way we deal with SOS and EOS
# 271 does not account for SOS and EOS