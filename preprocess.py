"""
This script performs data preprocessing by checking if the codes in the data are compilable or not.
We only keep those snippets that compile successfully with clang/clang++ and store them in /clean_data.
"""
import os
import warnings
import subprocess
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    except Exception:
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
    kept, skipped = 0, 0
    compiled_indexes = []

    with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
        print(f"Cleaning {infile} started.")

        for i, raw_line in enumerate(fin, 1):
            snippet = raw_line.rstrip("\n")
            if not snippet.strip():
                continue

            # Restore literal \n to real newlines
            code = snippet.encode("utf-8").decode("unicode_escape", errors='replace')

            if compiles(code, is_cpp=is_cpp):
                fout.write(raw_line + "\n")
                kept += 1
                if (is_paired == False):    # Only append for unpaired data, as for paired data we need to take intersection of C and C++ code
                    compiled_indexes.append(i)
            else:
                skipped += 1

            if i % 100 == 0:
                print(f"Processed {i} snippets with kept: {kept}, skipped: {skipped}")

    print(f"Cleaning {infile} completed. Kept {kept} snippets, skipped {skipped}.\n")
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

"""
This portion is for unpaired C / C++ data files only
"""
infile = "./data/unpaired_cpp.txt"
outfile = "./clean_data/unpaired_cpp.txt"
clear_output_files([outfile])
cpp_compiled_indexes = clean_file(infile, outfile, is_cpp=True, is_paired=False)


infile = "./data/unpaired_c.txt"
outfile = "./clean_data/unpaired_c.txt"
clear_output_files([outfile])
c_compiled_indexes = clean_file(infile, outfile, is_cpp=False, is_paired=False)

"""
This portion is for paired C / C++ data files only
"""
infile_cpp = "./data/paired_cpp.txt"
outfile_cpp = "./clean_data/paired_cpp.txt"
infile_c = "./data/paired_c.txt"
outfile_c = "./clean_data/paired_c.txt"
clear_output_files([outfile_cpp, outfile_c])

cpp_compiled_indexes = clean_file(infile_cpp, outfile_cpp, is_cpp=True, is_paired=True)
c_compiled_indexes = clean_file(infile_c, outfile_c, is_cpp=False, is_paired=True)

common_indexes = sorted(set(cpp_compiled_indexes) & set(c_compiled_indexes))
print(f"Total snippets that compiled in both C++ and C: {len(common_indexes)}")

# Write only the intersection snippets for paired case
write_selected_lines(infile_cpp, outfile_cpp, common_indexes)
write_selected_lines(infile_c, outfile_c, common_indexes)