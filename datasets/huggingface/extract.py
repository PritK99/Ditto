import pandas as pd
import os
from datasets import load_dataset

def process_solutions_from_df(df: pd.DataFrame):
    """
    Reads a Pandas DataFrame, extracts the content from the specified solution column,
    and writes each solution string to a separate .cpp file in the specified directory,
    using the ID column for file naming.
    
    """
    print(f"\n--- Processing {len(df)} solutions from DataFrame ---")

    # Define the common standard headers to inject
    # This expanded list includes the most popular containers and utilities 
    # to maximize compilation success after removing <bits/stdc++.h>.
    COMMON_HEADERS = """
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include <queue>
#include <stack>
#include <deque>
#include <utility>
#include <functional>
#include <tuple>
#include <sstream> 
#include <iomanip> 
"""

    for index, row in df.iterrows():
        try:
            solution_content = row["text"]

            solution_content = solution_content.replace("#include <bits/stdc++.h>", "")
            
            solution_content = COMMON_HEADERS.strip() + "\n\n" + solution_content
            
            filename = f"solution_{index}.cpp"
            output_path = os.path.join(filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(solution_content)
            print(f"Extracted and saved: {output_path}")

        except KeyError as e:
            print(f"FATAL ERROR: Column '{e}' not found in DataFrame. Check your column names and the rename logic.")
            return
        except Exception as e:
            print(f"Error writing file for row {index} (ID: {filename}): {e}")


if __name__ == "__main__":
    try:
        # Extract the stack_edu_cpp dataset.
        print("--- Loading Hugging Face Dataset 'hongliu9903/stack_edu_cpp' ---")
        dataset = load_dataset("hongliu9903/stack_edu_cpp")
        
        hf_dataset_split = dataset["train"]
        
        df_to_process = hf_dataset_split.to_pandas()


        process_solutions_from_df(df_to_process)

        # Extract the cpp_200k dataset.
        print("--- Loading Hugging Face Dataset 'kloodia/cpp_200k' ---")
        dataset = load_dataset("kloodia/cpp_200k")
        
        hf_dataset_split = dataset["train"]
        
        df_to_process = hf_dataset_split.to_pandas()

        process_solutions_from_df(df_to_process)


    except Exception as e:
        print(f"\nFATAL ERROR: Could not load or process the dataset.")
        print(f"Make sure you have the 'datasets' library installed: pip install datasets")
        print(f"Underlying error: {e}")

    print("\nProcess finished.")
    print(f"Check the current directory for your generated C++ files.")
