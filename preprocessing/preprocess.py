import re
import subprocess
import os
import tempfile
import csv
import logging
from tqdm import tqdm
from tokenizer import obfuscate_and_tokenize  
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,          
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="../assets/preprocessing_logs/preprocessing_cpp_logs.log",          
    filemode="w"                 
)

def separate_includes_and_code_from_text(code_text):
    """
    Separate include directives and the rest of the code from the source code

    Args:
        code_text: code in text format

    Returns:
        includes: all include directives
        code: code content other than include directives
    """
    include_pattern = re.compile(r'^\s*#\s*include\s+[<"].+[>"].*$', re.MULTILINE)
    includes = "\n".join(include_pattern.findall(code_text)).strip()
    code = include_pattern.sub("", code_text).strip()
    return includes, code

def preprocess_code_only(code_text, idx, is_cpp=False):
    """
    Preprocess code using gcc/g++ with -E -P to resolve macros and comments

    Args:
        code_text: code in text format
        idx: idx of code in csv file
        is_cpp (bool): False if C, true if C++
    
    Returns:
        result: preprocessed code
    """
    try:
        suffix = ".cpp" if is_cpp else ".c"
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
            tmp.write(code_text)
            tmp_name = tmp.name

        out_name = tmp_name + ".out"
        compiler = "g++" if is_cpp else "gcc"

        subprocess.run([compiler, "-E", "-P", tmp_name, "-o", out_name],
                       check=True, capture_output=True)

        with open(out_name) as f:
            result = f.read().strip()

        os.remove(tmp_name)
        os.remove(out_name)
        return result
    
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed for code {idx+1}")
        logging.info(f"Preprocessing failed for code {idx+1}")
        return ""

def safe_clean_code(line):
    """
    Remove non-ASCII or problematic Unicode/escape characters.
    """
    line = line.encode("ascii", errors="ignore").decode("ascii")
    line = line.replace("\\n", "\n").replace("\\t", "\t")
    return line

def process_txt_file(input_path, output_csv="output.csv"):
    """
    read file line by line, process, and write CSV.
    """
    is_cpp = "_cpp" in input_path.lower()

    # Preparing the CSV output
    fieldnames = [
        "line_number",
        "transformed_tokens",
        "var_dict",
        "func_dict",
        "lit_dict",
        "struct_dict",
        "class_dict"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for i, raw_line in enumerate(tqdm(lines, desc="Processing lines", unit="line")):
            line = safe_clean_code(raw_line.strip())
            if not line:
                continue

            try:
                includes, code = separate_includes_and_code_from_text(line)
                processed = preprocess_code_only(code, i, is_cpp)

                if not processed:
                    print(f"Skipping code {i+1}")
                    logging.info(f"Skipping code {i+1}")
                    continue
                
                transformed_tokens, var_dict, func_dict, lit_dict, struct_dict, class_dict = obfuscate_and_tokenize(
                    processed, is_cpp
                )

                writer.writerow({
                    "line_number": i + 1,
                    "transformed_tokens": transformed_tokens,
                    "var_dict": str(var_dict),
                    "func_dict": str(func_dict),
                    "lit_dict": str(lit_dict),
                    "struct_dict": str(struct_dict),
                    "class_dict": str(class_dict)
                })

            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                print(f"Skipping code {i+1}")
                logging.info(f"Error processing line {i+1}: {e}")
                logging.info(f"Skipping code {i+1}")
                continue

    print(f"\nProcessing completed. Results written to {output_csv}")
    logging.info(f"\nProcessing completed. Results written to {output_csv}")

# Process C and C++ raw data files
# input_path = "../data/unpaired_c.txt"
# process_txt_file(input_path, "../data/c_tokens.csv")

input_path = "../data/unpaired_cpp.txt"
process_txt_file(input_path, "../data/cpp_tokens.csv")