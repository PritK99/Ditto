import os
import subprocess
import re
import json
from collections import defaultdict
from clang import cindex
import tempfile

# Setup clang
cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

# Helper: tokenize line by non-alphanumeric separators (simplistic)
def tokenize_line(line):
    tokens = re.findall(r'\w+|\S', line)
    return tokens

def is_cpp_file(filename):
    return 'cpp' in filename.lower()

def get_output_path(input_path, suffix=''):
    filename = os.path.basename(input_path)
    return os.path.join('final_data', f"{filename.replace('.txt', '')}{suffix}")

# Use clang AST to collect variables and their scopes
def parse_variables_with_scope(code, is_cpp):
    """
    Returns dict with keys = (scope_id, var_name) -> generic_var_name (var1, var2...)
    scope_id = 'global' or function name for locals
    """
    index = cindex.Index.create()
    args = ['-std=c++17'] if is_cpp else ['-std=c11']

    # clang requires files so write temp
    with tempfile.NamedTemporaryFile(suffix='.cpp' if is_cpp else '.c', mode='w', delete=False) as tmp:
        tmp.write(code)
        tmpfile = tmp.name

    tu = index.parse(tmpfile, args=args)
    os.unlink(tmpfile)

    var_mapping = {}
    counters = defaultdict(int)

    def visit(node, scope='global'):
        if node.kind == cindex.CursorKind.VAR_DECL:
            name = node.spelling
            key = (scope, name)
            if key not in var_mapping:
                counters[scope] += 1
                var_mapping[key] = f'var{counters[scope]}'
        elif node.kind == cindex.CursorKind.FUNCTION_DECL:
            for c in node.get_children():
                visit(c, scope=node.spelling)
            return
        for c in node.get_children():
            visit(c, scope)

    visit(tu.cursor)
    return var_mapping

# Map numbers similarly
def map_numbers(tokens, num_mapping, num_counter):
    new_tokens = []
    for t in tokens:
        if t.isdigit():
            if t not in num_mapping:
                num_counter[0] += 1
                num_mapping[t] = f'num{num_counter[0]}'
            new_tokens.append(num_mapping[t])
        else:
            new_tokens.append(t)
    return new_tokens

def replace_vars_in_tokens(tokens, var_mapping, current_scope):
    replaced = []
    for t in tokens:
        key_local = (current_scope, t)
        key_global = ('global', t)
        if key_local in var_mapping:
            replaced.append(var_mapping[key_local])
        elif key_global in var_mapping:
            replaced.append(var_mapping[key_global])
        else:
            replaced.append(t)
    return replaced

def get_scope_from_line(line, current_scope):
    m = re.match(r'\s*fun\s+(\w+)', line)
    if m:
        return m.group(1)
    if line.strip() == 'end':
        return 'global'
    return current_scope

# Compile the code
def compiles(code: str, is_cpp: bool) -> bool:
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
        return False

# Function to process code and return tokenized output, mappings, and syntax tree
def process_file(filepath, save_original_code=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    is_cpp = is_cpp_file(filepath)
    var_mapping = parse_variables_with_scope(code, is_cpp)

    num_mapping = {}
    num_counter = [0]

    output_lines = []
    current_scope = 'global'
    var_mappings = {}
    syntax_trees = []
    original_code = []  # To store original code snippets

    lines = code.split('\n')
    
    for line in lines:
        # First, check if the line of code compiles
        if not compiles(line, is_cpp):
            continue  # Skip non-compilable lines
        
        current_scope = get_scope_from_line(line, current_scope)
        tokens = tokenize_line(line)
        
        tokens = map_numbers(tokens, num_mapping, num_counter)
        tokens = replace_vars_in_tokens(tokens, var_mapping, current_scope)
        
        # Convert tuple keys to string keys for JSON compatibility
        if current_scope not in var_mappings:
            var_mappings[current_scope] = {}
        
        # Convert tuple keys (scope, var_name) to string "scope_var_name"
        for (scope, var_name), generic_var_name in var_mapping.items():
            var_mappings[current_scope][f"{scope}_{var_name}"] = generic_var_name
        
        output_lines.append(','.join(tokens))
        syntax_trees.append(str(var_mapping))  # You can replace this with actual AST extraction if needed
        
        # Save original code if required
        if save_original_code:
            original_code.append(line)

    return output_lines, var_mappings, syntax_trees, original_code

# Process files and save output
def process_multiple_files(input_files, output_dir):
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_path = os.path.join(output_dir, filename)
        tokenized_path = get_output_path(input_file, "_tokenized.txt")
        mapping_path = get_output_path(input_file, "_mapping.json")
        st_path = get_output_path(input_file, "_ast.txt")
        original_code_path = get_output_path(input_file, ".txt")

        output_lines, var_mappings, syntax_trees, original_code = process_file(input_file, save_original_code=True)

        # Writing tokenized output
        with open(tokenized_path, 'w', encoding='utf-8') as fout:
            for line in output_lines:
                fout.write(line + '\n')

        # Writing variable mappings
        with open(mapping_path, 'w', encoding='utf-8') as fout:
            json.dump(var_mappings, fout, indent=4)

        # Writing syntax trees
        with open(st_path, 'w', encoding='utf-8') as fout:
            for tree in syntax_trees:
                fout.write(tree + '\n')

        # Writing original compilable code
        with open(original_code_path, 'w', encoding='utf-8') as fout:
            for line in original_code:
                fout.write(line + '\n')

        print(f"Processed {input_file} -> {output_path}, tokenized -> {tokenized_path}, mapping -> {mapping_path}, ST -> {st_path}, original code -> {original_code_path}")

# List of input files to be processed
input_files = ["./datasets/transcodeocean/data/unpaired_c.txt"]

# Ensure output directory exists
os.makedirs("final_data", exist_ok=True)

# Process all files
process_multiple_files(input_files, "final_data")
