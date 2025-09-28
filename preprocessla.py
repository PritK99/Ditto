import os
import json
import pandas as pd
from clang import cindex

cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def extract_tokens_and_ast(snippet: str, index: cindex.Index, is_cpp: bool):
    """
    Use libclang to extract tokens (no comments) and AST.
    Returns tokens (list of dicts), ast_dict.
    """
    # Create a temporary unsaved file for clang to parse
    filename = "temp.cpp" if is_cpp else "temp.c"
    args = ["-std=c++17"] if is_cpp else ["-std=c11"]

    tu = index.parse(
        path=filename,
        args=args,
        unsaved_files=[(filename, snippet)],
        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )

    # --- Tokens ---
    tokens = []
    for token in tu.get_tokens(extent=tu.cursor.extent):
        # skip comments
        if token.kind == cindex.TokenKind.COMMENT:
            continue
        tokens.append({
            "kind": str(token.kind),
            "spelling": token.spelling,
            "location": {
                "line": token.location.line,
                "column": token.location.column
            }
        })

    # --- AST ---
    def cursor_to_dict(node):
        result = {
            "kind": str(node.kind),
            "spelling": node.spelling,
            "type": str(node.type.spelling),
            "location": {
                "line": node.location.line,
                "column": node.location.column
            },
            "children": []
        }
        for child in node.get_children():
            result["children"].append(cursor_to_dict(child))
        return result

    ast_dict = cursor_to_dict(tu.cursor)

    return tokens, ast_dict

def clean_snippet(snippet: str) -> str:
    # replace literal escapes with space
    cleaned = snippet.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    # replace actual newlines with space
    cleaned = cleaned.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    return cleaned.strip()

def merge_preprocessor(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok["spelling"] == "#" and i + 1 < len(tokens):
            directive = ["#"]
            j = i + 1
            while j < len(tokens) and tokens[j]["spelling"] not in {"#", ";"}:
                directive.append(tokens[j]["spelling"])
                j += 1
            merged.append({
                "kind": "PreprocessorDirective",
                "spelling": " ".join(directive),
                "location": tok["location"]
            })
            i = j
        else:
            merged.append(tok)
            i += 1
    return merged

def process(file_path):
    index = cindex.Index.create()
    is_cpp = file_path.lower().endswith(".cpp")
    rows = []

    with open(file_path, "r", encoding="utf-8") as f_obj:
        for line in f_obj:
            snippet = line.strip()
            if not snippet:
                continue

            snippet = clean_snippet(snippet)

            try:
                tokens, ast_dict = extract_tokens_and_ast(snippet, index, is_cpp)
                tokens = merge_preprocessor(tokens)
                compilable = True
            except Exception as e:
                tokens, ast_dict, compilable = (
                    [{"kind": "Error", "spelling": str(e)}],
                    {"kind": "Error", "spelling": str(e)},
                    False
                )

            rows.append({
                "original_code": snippet,
                "is_compilable": compilable,
                "tokens": tokens,
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
cols = ["original_code", "is_compilable", "tokens", "ast"]
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
cols_to_serialize = ["tokens", "ast"]

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

# Printing samples of results
indices = [0, 1, 2]
for index in indices:
    row = unpaired_c_csv.loc[index]

    with open("log.txt", "a", encoding="utf-8") as logf:
        logf.write("original_code:\n")
        logf.write(row["original_code"] + "\n\n")

        logf.write("is_compilable:\n")
        logf.write(str(row["is_compilable"]) + "\n\n")

        logf.write("tokens:\n")
        try:
            tokens = json.loads(row["tokens"])
            logf.write(json.dumps(tokens, indent=2) + "\n\n")
        except Exception as e:
            logf.write(f"Error parsing tokens: {e}\n\n")

        logf.write("ast:\n")
        try:
            ast = json.loads(row["ast"])
            logf.write(json.dumps(ast, indent=2) + "\n\n")
        except Exception as e:
            logf.write(f"Error parsing AST: {e}\n\n")
        
        logf.write("="*80 + "\n\n")

