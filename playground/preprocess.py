import re
import subprocess
import os
import tempfile
import csv
from tqdm import tqdm
from tokenize import obfuscate_and_tokenize  

def extract_includes_and_code_from_text(code_text):
    """Extract include directives and the rest of the code."""
    include_pattern = re.compile(r'^\s*#\s*include\s+[<"].+[>"].*$', re.MULTILINE)
    includes = "\n".join(include_pattern.findall(code_text)).strip()
    code = include_pattern.sub("", code_text).strip()
    return includes, code

def preprocess_code_only(code_text, is_cpp=False):
    """Preprocess code using gcc/g++ with -E -P."""
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
        print(f"[!] Preprocessing failed: {e}")
        return ""

def safe_clean_code(line):
    """Remove non-ASCII or problematic Unicode/escape characters."""
    line = line.encode("ascii", errors="ignore").decode("ascii")
    line = line.replace("\\n", "\n").replace("\\t", "\t")
    return line

def process_txt_file(input_path, output_csv="output.csv"):
    """
    read file line by line, process, and write CSV.
    """
    is_cpp = "_cpp" in input_path.lower()

    # Prepare CSV output
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
                includes, code = extract_includes_and_code_from_text(line)
                processed = preprocess_code_only(code, is_cpp)

                if not processed:
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
                print(f"Error processing line {i + 1}: {e}")
                continue

    print(f"\nProcessing complete! Results written to {output_csv}")


if __name__ == "__main__":
    input_path = "../data/unpaired_c.txt"
    process_txt_file(input_path, "c_tokens.csv")
