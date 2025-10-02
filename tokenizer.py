import os
import re
import tempfile
from clang import cindex

cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def clean_source(source_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'\\.', ' ', content)
    return content

def process_file(source_file, output_csv, is_cpp: bool):
    source_file = os.path.abspath(source_file)
    cleaned_content = clean_source(source_file)
    suffix = '.cpp' if is_cpp else '.c'
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(cleaned_content)
        tmp_file.flush()
        tmp_filename = tmp_file.name

    index = cindex.Index.create()
    args = ['-x', 'c++', '-std=c++17'] if is_cpp else ['-x', 'c', '-std=c11']
    tu = index.parse(tmp_filename, args=args)
    tokens = list(tu.get_tokens(extent=tu.cursor.extent))

    # Maps: spelling -> anonymized name
    identifier_map = {}
    counts = {'var': 1, 'func': 1, 'class': 1, 'num': 1, 'str': 1}

    def from_source_file(cur):
        try:
            return cur and cur.location.file and os.path.abspath(cur.location.file.name) == source_file
        except:
            return False

    def get_identifier_type(cursor_kind):
        if cursor_kind in (cindex.CursorKind.CLASS_DECL, cindex.CursorKind.STRUCT_DECL, cindex.CursorKind.ENUM_DECL):
            return 'class'
        elif cursor_kind == cindex.CursorKind.FUNCTION_DECL:
            return 'func'
        elif cursor_kind in (cindex.CursorKind.VAR_DECL, cindex.CursorKind.PARM_DECL, cindex.CursorKind.FIELD_DECL, cindex.CursorKind.CONSTANT_DECL):
            return 'var'
        elif cursor_kind in (cindex.CursorKind.DECL_REF_EXPR, cindex.CursorKind.MEMBER_REF_EXPR):
            # These are references, will resolve from map
            return None
        else:
            return None

    def anonymize_identifier(spelling, id_type):
        if spelling not in identifier_map:
            # assign new anonymized name
            anonymized_name = f"{id_type}{counts[id_type]}"
            counts[id_type] += 1
            identifier_map[spelling] = anonymized_name
        return identifier_map[spelling]

    anonymized_tokens = []

    for token in tokens:
        spelling = token.spelling
        kind = token.kind
        cursor = token.cursor if hasattr(token, 'cursor') else None

        if kind == cindex.TokenKind.KEYWORD:
            anonymized_tokens.append(spelling)

        elif kind == cindex.TokenKind.IDENTIFIER:
            if from_source_file(cursor):
                ckind = cursor.kind
                id_type = get_identifier_type(ckind)
                if id_type is None:
                    # It's a reference to an identifier
                    # Try to find in map, else keep as is (e.g. external or undeclared)
                    if spelling in identifier_map:
                        anonymized_tokens.append(identifier_map[spelling])
                    else:
                        anonymized_tokens.append(spelling)
                else:
                    # Declaration — assign or reuse anonymized name
                    anon_name = anonymize_identifier(spelling, id_type)
                    anonymized_tokens.append(anon_name)
            else:
                # Not from source file — external identifier, leave as is
                anonymized_tokens.append(spelling)

        elif kind == cindex.TokenKind.LITERAL:
            # Literals: string or numeric
            if cursor:
                ckind = cursor.kind
                if ckind in (cindex.CursorKind.INTEGER_LITERAL, cindex.CursorKind.FLOATING_LITERAL):
                    if spelling not in identifier_map:
                        anon_name = f"num{counts['num']}"
                        counts['num'] += 1
                        identifier_map[spelling] = anon_name
                    anonymized_tokens.append(identifier_map[spelling])
                elif ckind == cindex.CursorKind.STRING_LITERAL:
                    if spelling not in identifier_map:
                        anon_name = f"str{counts['str']}"
                        counts['str'] += 1
                        identifier_map[spelling] = anon_name
                    anonymized_tokens.append(identifier_map[spelling])
                else:
                    anonymized_tokens.append(spelling)
            else:
                # fallback for literals with no cursor info
                try:
                    float(spelling)
                    if spelling not in identifier_map:
                        anon_name = f"num{counts['num']}"
                        counts['num'] += 1
                        identifier_map[spelling] = anon_name
                    anonymized_tokens.append(identifier_map[spelling])
                except ValueError:
                    anonymized_tokens.append(spelling)

        else:
            anonymized_tokens.append(spelling)

    with open(output_csv, 'a', encoding='utf-8') as f:
        f.write(' '.join(anonymized_tokens) + '\n')

    # os.remove(tmp_filename)  # optionally cleanup temp file
    print(anonymized_tokens)
    return anonymized_tokens

# Batch processing code

dir_name = "final_data"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

files = ["unpaired_cpp.csv"]

# Ensure CSV files exist (empty if new)
for file_name in files:
    file_path = os.path.join(dir_name, file_name)
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass

datasets = ["transcodeocean"]

for dataset in datasets:
    base_path = os.path.join("datasets", dataset, "data")
    for file_name in files:
        src_base = file_name.replace(".csv", ".txt")
        source_file = os.path.join(base_path, src_base)
        if not os.path.isfile(source_file):
            print(f"Source file does not exist: {source_file}")
            continue
        output_csv = os.path.join(dir_name, file_name)
        if "unpaired" in file_name and "cpp" in file_name:
            print(f"Processing C++ unpaired file: {source_file}")
            process_file(source_file, output_csv, is_cpp=True)
        elif "unpaired" in file_name and "c" in file_name:
            print(f"Processing C unpaired file: {source_file}")
            process_file(source_file, output_csv, is_cpp=False)
        else:
            print(f"Skipping file {file_name}")
