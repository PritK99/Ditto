import os
import warnings
import subprocess
from collections import defaultdict

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Compile status log
def log_stats(processed, compiled, skipped, rejected, step=100):
    """
    Log stats for every 1000 files processed and at the end.
    """
    if processed % step == 0:
        print(f"Processed {processed} files: {compiled} compiled, {rejected} rejected, {skipped} skipped.")

def clear_output_files(files: list):
    """
    Clears the content of all files (/clean_data)

    Args:
        files (list): list of all the files which need to be cleared
    """
    for file_path in files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open(file_path, 'w', encoding='utf-8').close()

def compiles(code: str, is_cpp: bool) -> bool:
    """
    Return True if code compiles

    Args:
        code (str): code in text format
        is_cpp (bool): True for C++ code compilation, False for C code compilation
    """
    compiler = "clang++" if is_cpp else "clang"
    try:
        proc = subprocess.run(
            [compiler, "-x", "c++" if is_cpp else "c",
             "-std=c++17" if is_cpp else "-std=c11", "-fsyntax-only", "-"],
            input=code.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return proc.returncode == 0
    except Exception as e:
        print(f"Error compiling: {e}")
        return False

def clean_file(infile: str, outfile: str, is_cpp: bool, is_paired: bool):
    """
    Processes the input file and writes only compilable snippets to the output file.

    Args:
        infile (str): path to infile
        outfile (str): path to outfile
        is_cpp (bool): True for C++ code compilation, False for C code compilation
        is_paired (bool): True if working with paired data, False otherwise
    
    Returns:
        compiled_indexes (list): list of index of all compilable code snippets
    """
    kept, skipped, rejected, processed = 0, 0, 0, 0
    compiled_indexes = []

    with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
        print(f"Cleaning {infile} started.")
        for i, raw_line in enumerate(fin, 1):
            snippet = raw_line.rstrip("\n")
            if not snippet.strip():
                skipped += 1
                continue

            # Restore literal \n to real newlines
            code = snippet.encode("utf-8").decode("unicode_escape", errors='replace')

            # Compile the code
            if compiles(code, is_cpp=is_cpp):
                fout.write(raw_line + "\n")
                kept += 1
                if not is_paired:    # Only append for unpaired data
                    compiled_indexes.append(i)
            else:
                rejected += 1

            processed += 1
            log_stats(processed, kept, skipped, rejected)

        print(f"Cleaning {infile} completed. Kept {kept} snippets, skipped {skipped}, rejected {rejected}.\n")
    return compiled_indexes

def write_selected_lines(infile: str, outfile: str, selected_indexes: list):
    """
    Writing the compilable code snippets for paired data

    Args:
        infile (str): path to infile
        outfile (str): path to outfile
        selected_indexes (list): list of index of all compilable code snippets
    """
    selected_set = set(selected_indexes)
    with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            if i in selected_set:
                fout.write(line)

def process_multiple_files(input_files, output_dir):
    """
    Process a list of input files, clean and compile the code, then write results.
    
    Args:
        input_files (list): List of paths to input files
        output_dir (str): Directory to save the processed output
    """
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_path = os.path.join(output_dir, filename)
        is_cpp = 'cpp' in filename.lower()

        print(f"Processing {input_file} -> {output_path}")
        clear_output_files([output_path])
        
        compiled_indexes = clean_file(input_file, output_path, is_cpp, is_paired=False)

        print(f"Completed {filename}: {len(compiled_indexes)} snippets compiled successfully.")
        
# List of input files to be processed
input_files = ["./datasets/transcodeocean/data/unpaired_c.txt", 
               "./datasets/transcodeocean/data/unpaired_cpp.txt"]

# Ensure output directory exists
os.makedirs("final_data", exist_ok=True)

# Process all files
process_multiple_files(input_files, "final_data")
