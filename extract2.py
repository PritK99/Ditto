import os
import random
import re

def remove_comments(code: str) -> str:
    """
    Remove C/C++ single-line (//...) and multi-line (/* ... */) comments from code.
    """
    # Remove single-line comments
    code_no_single = re.sub(r'//.*', '', code)
    # Remove multi-line comments
    code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_single, flags=re.DOTALL)
    return code_no_comments

def escape_code(code: str) -> str:
    """Escape newlines and tabs so they appear as \\n and \\t in output."""
    return code.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")

def create_dataset(folder_path: str, n: int, output_file: str = "dataset.txt"):
    # Get all .c and .cpp files in the folder
    all_files = [f for f in os.listdir(folder_path) if f.endswith((".c", ".cpp"))]
    
    if not all_files:
        raise ValueError("No C/C++ files found in the given folder.")

    # If N > available files, take all
    chosen_files = random.sample(all_files, min(n, len(all_files)))

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in chosen_files:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
                code_no_comments = remove_comments(code)
                escaped = escape_code(code_no_comments)
                out_f.write(escaped + "\n")  # one file = one line

    print(f"✅ Dataset created: {output_file}")
    print(f"Included {len(chosen_files)} files.")

# ------------------ Example usage ------------------ #
if __name__ == "__main__":
    raw_dataset_path = "./raw_data/DATASET-v2_without_clones"
    N = 100000
    output_file = "./sample.txt"
    create_dataset(raw_dataset_path, N, output_file)
