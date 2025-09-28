import os
import sys
import re
from collections import defaultdict
from clang import cindex  

cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def is_cpp_file(filename):
    return 'cpp' in filename.lower()

def get_output_path(input_path):
    filename = os.path.basename(input_path)
    return os.path.join('final_output', filename)

# Helper: tokenize line by non-alphanumeric separators (simplistic)
def tokenize_line(line):
    # This is naive; you can replace with a proper lexer if you want
    tokens = re.findall(r'\w+|\S', line)
    return tokens

# Use clang AST to collect variables and their scopes
def parse_variables_with_scope(code, is_cpp):
    """
    Returns dict with keys = (scope_id, var_name) -> generic_var_name (var1, var2...)
    scope_id = 'global' or function name for locals
    """

    index = cindex.Index.create()
    args = ['-std=c++17'] if is_cpp else ['-std=c11']

    # clang requires files so write temp
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.cpp' if is_cpp else '.c', mode='w', delete=False) as tmp:
        tmp.write(code)
        tmpfile = tmp.name

    tu = index.parse(tmpfile, args=args)

    os.unlink(tmpfile)

    var_mapping = {}
    counters = defaultdict(int)

    def visit(node, scope='global'):
        # Interested in VarDecl nodes for variables
        if node.kind == cindex.CursorKind.VAR_DECL:
            name = node.spelling
            # Create key with scope + name
            key = (scope, name)
            if key not in var_mapping:
                counters[scope] += 1
                var_mapping[key] = f'var{counters[scope]}'
        elif node.kind == cindex.CursorKind.FUNCTION_DECL:
            # For function body, recurse with scope = function name
            for c in node.get_children():
                visit(c, scope=node.spelling)
            return
        # Recurse for children in the same scope
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
    """
    Replace variables in tokens with mapped varX
    If variable defined in current_scope, use that mapping
    Else if defined globally, use global mapping
    Else keep as is (possibly not a variable)
    """
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
    """
    Detect function scope from line if function def starts.
    This is simplistic and assumes function definition lines start with e.g. "int main()"
    or "void foo()"
    For demo, we look for lines ending with '{' or 'fun name' like in your sample.

    Returns new scope or current_scope
    """
    # Your sample code looks like 'fun main' or 'fun printnum' for functions
    m = re.match(r'\s*fun\s+(\w+)', line)
    if m:
        return m.group(1)
    # If line == 'end' means end of function
    if line.strip() == 'end':
        return 'global'
    return current_scope

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    is_cpp = is_cpp_file(filepath)
    var_mapping = parse_variables_with_scope(code, is_cpp)

    num_mapping = {}
    num_counter = [0]

    output_lines = []
    current_scope = 'global'

    # Split code into lines
    lines = code.split('\n')

    for line in lines:
        current_scope = get_scope_from_line(line, current_scope)
        tokens = tokenize_line(line)
        # Replace numbers
        tokens = map_numbers(tokens, num_mapping, num_counter)
        # Replace variables with varX
        tokens = replace_vars_in_tokens(tokens, var_mapping, current_scope)
        # Join tokens as comma separated
        output_lines.append(','.join(tokens))

    return output_lines


input_files = ["./datasets/transcodeocean/data/unpaired_c.txt"]
os.makedirs(os.path.dirname("./final_data"), exist_ok=True)

for input_file in input_files:
    filename = os.path.basename(input_file)
    output_path = os.path.join('final_data', filename)

    output_lines = process_file(input_file)
    with open(output_path, 'a', encoding='utf-8') as fout:
        for line in output_lines:
            fout.write(line + '\n')

    print(f"Processed {input_file} -> {output_path}")

