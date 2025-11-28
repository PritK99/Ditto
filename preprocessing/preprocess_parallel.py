import re
import sys
import subprocess
import os
import tempfile
import csv
import logging
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tokenizer import obfuscate_and_tokenize  

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="preprocessing_cpp_logs.log",
    filemode="w"
)

# ---------------------- Utility Functions ----------------------
def separate_includes_and_code_from_text(code_text):
    include_pattern = re.compile(r'^\s*#\s*include\s+[<"].+[>"].*$', re.MULTILINE)
    includes = "\n".join(include_pattern.findall(code_text)).strip()
    code = include_pattern.sub("", code_text).strip()
    return includes, code

def preprocess_code_only(code_text, idx, is_cpp=False):
    try:
        suffix = ".cpp" if is_cpp else ".c"
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
            tmp.write(code_text)
            tmp_name = tmp.name

        out_name = tmp_name + ".out"
        compiler = "g++" if is_cpp else "gcc"

        subprocess.run(
            [compiler, "-E", "-P", tmp_name, "-o", out_name],
            check=True, capture_output=True
        )

        with open(out_name) as f:
            result = f.read().strip()

        os.remove(tmp_name)
        os.remove(out_name)
        return result

    except subprocess.CalledProcessError:
        logging.info(f"Preprocessing failed for code {idx+1}")
        return ""

def safe_clean_code(line):
    line = line.encode("ascii", errors="ignore").decode("ascii")
    line = line.replace("\\n", "\n").replace("\\t", "\t")
    return line

# ---------------------- Worker Function (runs in each process) ----------------------
def process_single_line(args):
    """
    Runs in parallel workers.
    Returns None or a tuple that the master process writes to CSV.
    """
    i, raw_line, is_cpp = args

    try:
        line = safe_clean_code(raw_line.strip())
        if not line:
            return None

        includes, code = separate_includes_and_code_from_text(line)
        processed = preprocess_code_only(code, i, is_cpp)

        if not processed:
            logging.info(f"Skipping code {i+1}")
            return None

        transformed_tokens, var_dict, func_dict, lit_dict, struct_dict, class_dict = \
            obfuscate_and_tokenize(processed, is_cpp)

        return (
            i + 1,
            transformed_tokens,
            var_dict,
            func_dict,
            lit_dict,
            struct_dict,
            class_dict
        )

    except Exception as e:
        print(f"Error processing code {i+1}: {e}")
        line = safe_clean_code(raw_line.strip())
        print(line)
        logging.info(f"Error processing code {i+1}: {e}")
        return None

# ---------------------- Main Parallel Processing Function ----------------------
def process_txt_file_parallel(input_path, output_csv="output.csv", workers=None):
    is_cpp = "_cpp" in input_path.lower()

    if workers is None:
        workers = max(cpu_count() - 1, 1)

    logging.info(f"Using {workers} worker processes.")

    # Load the input file lines (cheap: only 200k short strings)
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # CSV Setup
    fieldnames = [
        "line_number",
        "transformed_tokens",
        "var_dict",
        "func_dict",
        "lit_dict",
        "struct_dict",
        "class_dict"
    ]

    out = open(output_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()

    # Prepare multiprocessing
    pool = Pool(processes=workers)
    tasks = ((i, line, is_cpp) for i, line in enumerate(lines))

    # imap preserves order, so index stays correct
    for result in tqdm(pool.imap(process_single_line, tasks),
                       total=len(lines), desc="Processing", unit="line"):
        if result is None:
            continue

        line_num, transformed_tokens, var_dict, func_dict, lit_dict, struct_dict, class_dict = result

        writer.writerow({
            "line_number": line_num,
            "transformed_tokens": transformed_tokens,
            "var_dict": str(var_dict),
            "func_dict": str(func_dict),
            "lit_dict": str(lit_dict),
            "struct_dict": str(struct_dict),
            "class_dict": str(class_dict)
        })

    pool.close()
    pool.join()
    out.close()

    logging.info(f"Processing completed. Results saved to {output_csv}")
    print(f"\nProcessing completed. Results saved to {output_csv}")


if __name__ == "__main__":
    input_path = "../data/unpaired_cpp.txt"
    process_txt_file_parallel(input_path, "../data/cpp_tokens.csv", 15)
