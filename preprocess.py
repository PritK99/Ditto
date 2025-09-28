import os
import json
import pandas as pd
from clang import cindex

cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def tokenize_with_mapping(snippet, index, is_cpp):
    """
    Tokenizes a snippet using libclang, returns:
    - tokens: list of tokens
    - mapping: dict of variable/literal mappings
    - ast_dict: recursive AST as nested dict
    Resets var/val counters per snippet.
    """
    # Preprocess snippet: remove unwanted escape chars for tokenization
    snippet_clean = snippet.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")

    tmp_path = "tmp.c" if not is_cpp else "tmp.cpp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(snippet_clean)

    try:
        tu = index.parse(tmp_path, args=["-std=c++17"] if is_cpp else ["-std=c11"])
    except Exception:
        os.remove(tmp_path)
        return [], {}, {}
    os.remove(tmp_path)

    var_map = {}
    val_count = 1
    var_count = 1

    def visit(node, scope_vars):
        nonlocal var_count, val_count
        tokens = []
        ast_node = {"kind": node.kind.name, "spelling": node.spelling, "children": []}

        if node.kind in [cindex.CursorKind.VAR_DECL, cindex.CursorKind.PARM_DECL]:
            name = node.spelling
            if name not in scope_vars:
                scope_vars[name] = f"var{var_count}"
                var_count += 1

        for c in node.get_children():
            child_tokens, child_ast = visit(c, dict(scope_vars))
            tokens.extend(child_tokens)
            ast_node["children"].append(child_ast)

        # Generate tokens
        for t in node.get_tokens():
            tok = t.spelling.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
            if t.kind == cindex.TokenKind.IDENTIFIER:
                mapped = scope_vars.get(tok, tok)
                tokens.append(mapped)
            elif t.kind == cindex.TokenKind.LITERAL:
                val_name = f"val{val_count}"
                tokens.append(val_name)
                var_map[tok] = val_name
                val_count += 1
            else:
                tokens.append(tok)
        return tokens, ast_node

    tokens, ast_dict = visit(tu.cursor, {})
    return tokens, var_map, ast_dict


def check_compilable(snippet, index, is_cpp):
    tmp_path = "tmp.c" if not is_cpp else "tmp.cpp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(snippet)
    try:
        tu = index.parse(tmp_path, args=["-std=c++17"] if is_cpp else ["-std=c11"])
        compilable = len(list(tu.diagnostics)) == 0
    except Exception:
        compilable = False
    finally:
        os.remove(tmp_path)
    return compilable


def process(file_path):
    """
    Processes a file line by line, returns dict with:
    original_code, is_compilable, tokens, mapping, ast
    Stores original_code as a single line (escape chars removed)
    """
    index = cindex.Index.create()
    is_cpp = file_path.lower().endswith(".cpp")
    rows = []

    with open(file_path, "r", encoding="utf-8") as f_obj:
        for line in f_obj:
            snippet = line.strip()  # remove leading/trailing whitespace
            if not snippet:
                continue

            # Treat snippet literally; remove escape chars for storage
            snippet_one_line = snippet.replace("\\n", " ").replace("\\t", " ")

            compilable = check_compilable(snippet, index, is_cpp)
            tokens, mapping, ast_dict = tokenize_with_mapping(snippet, index, is_cpp)

            rows.append({
                "original_code": snippet_one_line,
                "is_compilable": compilable,
                "tokens": tokens,
                "mapping": mapping,
                "ast": ast_dict
            })
    return rows


def serialize_for_csv(df, cols_to_serialize):
    df_copy = df.copy()
    for col in cols_to_serialize:
        df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x))
    return df_copy


# Directory to store final data
directory = "final_data"
os.makedirs(directory, exist_ok=True)

# Columns and DataFrames
cols = ["original_code", "is_compilable", "tokens", "mapping", "ast"]
unpaired_c_df = pd.DataFrame(columns=cols)
unpaired_cpp_df = pd.DataFrame(columns=cols)
paired_c_df = pd.DataFrame(columns=cols)
paired_cpp_df = pd.DataFrame(columns=cols)

datasets = ["transcodeocean"]

# Process datasets
for dataset in datasets:
    path = f"./datasets/{dataset}/data/"
    if not os.path.exists(path):
        print(f"path {path} does not exist")
        continue

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue

        print(f"Processing file {file_path}...")
        df_result = pd.DataFrame(process(file_path))

        # Append to correct DataFrame
        if "_c." in file_name.lower() and "unpaired" in file_name.lower():
            unpaired_c_df = pd.concat([unpaired_c_df, df_result], ignore_index=True)
        elif "_cpp." in file_name.lower() and "unpaired" in file_name.lower():
            unpaired_cpp_df = pd.concat([unpaired_cpp_df, df_result], ignore_index=True)
        elif "_c." in file_name.lower() and "paired" in file_name.lower():
            paired_c_df = pd.concat([paired_c_df, df_result], ignore_index=True)
        elif "_cpp." in file_name.lower() and "paired" in file_name.lower():
            paired_cpp_df = pd.concat([paired_cpp_df, df_result], ignore_index=True)

# Print shapes
print("Shapes of DataFrames:")
print("unpaired_c_df:", unpaired_c_df.shape)
print("unpaired_cpp_df:", unpaired_cpp_df.shape)
print("paired_c_df:", paired_c_df.shape)
print("paired_cpp_df:", paired_cpp_df.shape)

# Columns to serialize
cols_to_serialize = ["tokens", "mapping", "ast"]

# Serialize DataFrames
unpaired_c_csv = serialize_for_csv(unpaired_c_df, cols_to_serialize)
unpaired_cpp_csv = serialize_for_csv(unpaired_cpp_df, cols_to_serialize)
paired_c_csv = serialize_for_csv(paired_c_df, cols_to_serialize)
paired_cpp_csv = serialize_for_csv(paired_cpp_df, cols_to_serialize)

# Save to CSV
unpaired_c_csv.to_csv("final_data/unpaired_c.csv", index=False)
unpaired_cpp_csv.to_csv("final_data/unpaired_cpp.csv", index=False)
paired_c_csv.to_csv("final_data/paired_c.csv", index=False)
paired_cpp_csv.to_csv("final_data/paired_cpp.csv", index=False)

print("All DataFrames saved as CSV in final_data/")
