import os
import re
import json
import csv
from clang import cindex

# Path for clang library
cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def remove_escape_characters(code: str) -> str:
    """
    Remove escape characters like \n, \t, etc., and handle them properly.
    """
    # First, handle literal '\n' and other escape characters
    code = bytes(code, 'utf-8').decode('unicode_escape')
    
    # Replacing //n with the literal string /n
    code = re.sub(r'//n', '/n', code)    # This is for case where original code has \n which becomes \\n in data
    
    return code

def remove_comments(code: str) -> str:
    """
    Remove both single-line and multi-line comments from the code.
    """
    # Remove single-line comments (//)
    code = re.sub(r'//.*', '', code)
    # Remove multi-line comments (/* */)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def identify_token_type(cursor):
    """
    Identify the type of token (function, class, variable) based on the cursor.
    """
    if cursor.kind == cindex.CursorKind.FUNCTION_DECL or cursor.kind == cindex.CursorKind.CXX_METHOD:
        return 'function'
    elif cursor.kind == cindex.CursorKind.CLASS_DECL:
        return 'class'
    elif cursor.kind == cindex.CursorKind.VAR_DECL or cursor.kind == cindex.CursorKind.PARM_DECL:
        return 'variable'
    else:
        return 'unknown'
    
def extract_tokens_from_code(code: str, is_cpp: bool):
    """
    Extract tokens from the given code string using clang.
    """
    # Create a temporary file with the code content
    temp_code_file = 'temp_code.cpp'
    with open(temp_code_file, 'w', encoding='utf-8') as temp_file:
        temp_file.write(code)

    # Use clang to extract tokens
    index = cindex.Index.create()
    args = ['-x', 'c++', '-std=c++17'] if is_cpp else ['-x', 'c', '-std=c11']
    tu = index.parse(temp_code_file, args=args)
    tokens = tu.get_tokens(extent=tu.cursor.extent)

    # Extract token details
    token_list = []
    for token in tokens:

        token_info = {
            'spelling': token.spelling,
            'kind': token.kind,
            'location': (token.location.line, token.location.column),
            'type': 'unknown'
        }

        if (token.kind == cindex.TokenKind.IDENTIFIER):
            cursor = token.cursor
            if cursor:
                token_info['type'] = identify_token_type(cursor)

                if (token_info['type'] == 'unknown'):
                    token_info['type'] = token.spelling
        
        token_list.append(token_info)

    # Clean up the temporary code file
    os.remove(temp_code_file)
    
    return token_list

def process_code_file(code_block: str, output_file, is_cpp: bool):
    """
    Process a single code file by removing escape characters, comments, and extracting tokens.
    """
    # Step 1: Remove escape characters
    clean_code = remove_escape_characters(code_block)
    
    # Step 2: Remove comments
    clean_code = remove_comments(clean_code)
    
    # Step 3: Extract tokens from the cleaned code block
    tokens = extract_tokens_from_code(clean_code, is_cpp)
 
    # Step 4: Code obfuscation
    var_count = 0
    class_count = 0
    func_count = 0
    obfuscated_dict = {}
    for token in tokens:
            if (token["kind"] == cindex.TokenKind.IDENTIFIER and token["type"] != token["spelling"]):
                if (token["spelling"] in obfuscated_dict.keys()):
                    token["spelling"] = obfuscated_dict[token["spelling"]]
                else:
                    if (token["type"] == "variable"):
                        obfuscated_dict[token["spelling"]] = f"var{var_count}"
                        token["spelling"] = obfuscated_dict[token["spelling"]]
                        var_count += 1
                    elif (token["type"] == "function"):
                        obfuscated_dict[token["spelling"]] = f"func{func_count}"
                        token["spelling"] = obfuscated_dict[token["spelling"]]
                        func_count += 1
                    elif (token["type"] == "class"):
                        obfuscated_dict[token["spelling"]] = f"class{class_count}"
                        token["spelling"] = obfuscated_dict[token["spelling"]]
                        class_count += 1
    
    # Step 5: Write the tokens to the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['spelling', 'kind', 'location', 'type'])
        if f.tell() == 0:
            writer.writeheader()  # Write the header only once
        for token in tokens:
            writer.writerow(token)
    
    return obfuscated_dict

def process_txt_file(input_txt_file, output_file, is_cpp: bool):
    """
    Process the input txt file line by line (each line is a code block).
    """
    with open(input_txt_file, 'r', encoding='utf-8') as f:
        code_files = f.readlines()

    obfuscated_dict_list = []
    for idx, code_block in enumerate(code_files):
        if (idx % 1000 == 0):
            print(f"Completed processing code {idx + 1} / {len(code_files)}")
        obfuscated_dict = process_code_file(code_block.strip(), output_file, is_cpp)
        obfuscated_dict_list.append(obfuscated_dict)
    
    return obfuscated_dict_list


def save_tokens_to_json(obfuscated_dict_list, output_file):
    indexed_dict = {i: obfuscated_dict_list[i] for i in range(len(obfuscated_dict_list))}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(indexed_dict, f, indent=4)

def main():
    # Directory where the output files will be stored
    dir_name = "../clean_data"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    input_txt_file = "../final_data/test_cpp.txt" 

    base_name = os.path.splitext(os.path.basename(input_txt_file))[0]
    output_file = os.path.join(dir_name, f"{base_name}_tokens.csv")

    is_cpp = False
    if "cpp" in input_txt_file:
        is_cpp = True

    # Process the file
    print(f"Processing the input file: {input_txt_file}")
    obfuscated_dict_list = process_txt_file(input_txt_file, output_file, is_cpp)

    print(f"Tokens for {input_txt_file} saved at {output_file}")
    dict_path = os.path.join(dir_name, f"{base_name}_obfuscated_tokens.json")
    save_tokens_to_json(obfuscated_dict_list, dict_path)

if __name__ == "__main__":
    main()