from clang import cindex
import json

# Set path to your libclang
cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def fix_escaped_newlines(line: str) -> str:
    """Convert escaped sequences like \\n, \\t to actual characters."""
    return line.encode('utf-8').decode('unicode_escape', errors='replace')

def tokenize_code(code: str, is_cpp: bool=False):
    """
    Tokenize code using libclang, keeping everything as in the original.
    Returns a list of token strings.
    """
    tu = cindex.Index.create().parse(
        path="snippet.cpp" if is_cpp else "snippet.c",
        args=["-std=c++17"] if is_cpp else ["-std=c11"],
        unsaved_files=[("snippet.cpp" if is_cpp else "snippet.c", code)],
        options=0
    )

    tokens_out = [token.spelling for token in tu.get_tokens(extent=tu.cursor.extent)]
    return tokens_out

def process_file(input_path: str, output_path: str, is_cpp: bool=False):
    """
    Read a TXT file (one snippet per line), tokenize each snippet,
    and write tokens as Python list (JSON format) to output file.
    """
    with open(input_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out_f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            code = fix_escaped_newlines(line)
            try:
                tokens = tokenize_code(code, is_cpp)
                out_f.write(json.dumps(tokens) + "\n")  # Write as Python list
            except Exception as e:
                print(f"Error processing snippet #{i}: {e}", file=sys.stderr)

    print(f"✅ Tokenized dataset written to {output_path}")


# ------------------ Example usage ------------------ #
if __name__ == "__main__":
    input_file = "./clean_data/unpaired_c.txt"   # each line = one code snippet
    output_file = "./tokens.txt"                  # output tokens per line as list
    process_file(input_file, output_file, is_cpp=False)
