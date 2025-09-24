"""
This script perform tokenization on data using libclang.
It converts code to tokens (lexing), and constructing an abstract syntax tree out of the tokens (parsing).
"""
import sys
import json
import subprocess
import shutil
from typing import List, Dict, Any

from clang import cindex
cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")  

def fix_escaped_newlines(line: str) -> str:
    """
    Dealing with escape characters.

    Args:
        line (str): line of the code snippet
    
    Returns 
        line (str): modified line after dealing with escape characters
    """
    if "\\n" in line or "\\t" in line or "\\\\" in line:
        return line.encode('utf-8').decode('unicode_escape', errors='replace')
    return line




# ------------------ libclang implementation ------------------ #
def _get_tokens_libclang(code: str, is_cpp: bool = False) -> List[Dict[str, Any]]:
    index = cindex.Index.create()
    filename = 'snippet.c' if not is_cpp else 'snippet.cpp'
    args = ['-std=c11'] if not is_cpp else ['-std=c++17']
    tu = index.parse(path=filename, args=args, unsaved_files=[(filename, code)], options=0)
    tokens = list(tu.get_tokens(extent=tu.cursor.extent))
    out = []
    for t in tokens:
        out.append({
            'spelling': t.spelling,
            'kind': str(t.kind).split('.')[-1],
            'location': {
                'file': t.location.file.name if t.location.file else None,
                'line': t.location.line,
                'column': t.location.column
            }
        })
    return out

def _cursor_to_dict(cursor: 'cindex.Cursor') -> Dict[str, Any]:
    """
    Convert a clang Cursor node to a serializable dict.
    """
    node = {
        'kind': cursor.kind.name,
        'spelling': cursor.spelling,
        'displayname': cursor.displayname,
        'location': {
            'file': cursor.location.file.name if cursor.location.file else None,
            'line': cursor.location.line,
            'column': cursor.location.column
        },
        'type': str(cursor.type.spelling) if cursor.type else '',
        'children': []
    }
    for c in cursor.get_children():
        node['children'].append(_cursor_to_dict(c))
    return node

def _get_ast_libclang(code: str, is_cpp: bool = False) -> Dict[str, Any]:
    filename = 'snippet.c' if not is_cpp else 'snippet.cpp'
    args = ['-std=c11'] if not is_cpp else ['-std=c++17']
    index = cindex.Index.create()
    tu = index.parse(path=filename, args=args, unsaved_files=[(filename, code)], options=0)
    # top-level AST node is tu.cursor
    return _cursor_to_dict(tu.cursor)

# ------------------ subprocess clang fallback ------------------ #
def _run_clang_subprocess(code: str, is_cpp: bool = False):
    """
    Return (tokens_str, ast_json_str) by invoking clang/clang++ subprocess.
    - tokens_str: textual dump from -Xclang -dump-tokens
    - ast_json_str: JSON ast from -Xclang -ast-dump=json
    """
    clang_bin = 'clang++' if is_cpp else 'clang'
    if not shutil.which(clang_bin):
        raise FileNotFoundError(f"{clang_bin} not found in PATH. Install clang or libclang.")
    # tokens
    try:
        p1 = subprocess.run(
            [clang_bin, '-fsyntax-only', '-x', 'c++' if is_cpp else 'c', '-std=c++17' if is_cpp else '-std=c11',
             '-Xclang', '-dump-tokens', '-'],
            input=code.encode('utf-8'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        tokens_out = (p1.stdout + p1.stderr).decode('utf-8', errors='ignore')
    except Exception as e:
        tokens_out = f'ERROR running clang dump-tokens: {e}'

    # ast json
    try:
        p2 = subprocess.run(
            [clang_bin, '-fsyntax-only', '-x', 'c++' if is_cpp else 'c', '-std=c++17' if is_cpp else '-std=c11',
             '-Xclang', '-ast-dump=json', '-'],
            input=code.encode('utf-8'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        ast_out = (p2.stdout + p2.stderr).decode('utf-8', errors='ignore')
    except Exception as e:
        ast_out = f'ERROR running clang ast-dump=json: {e}'

    return tokens_out, ast_out

def _parse_clang_tokens_dump(tokens_dump: str) -> List[Dict[str,Any]]:
    """
    Parse clang -dump-tokens output. Format lines like:
      token: identifier "int" at [0:0 - 0:3]
    We try to extract token kind and spelling and location when available.
    """
    out = []
    for line in tokens_dump.splitlines():
        line = line.strip()
        if not line:
            continue
        # naive parse
        # e.g. token: identifier "int" at [1:1 - 1:4]
        try:
            if line.startswith('token:'):
                # split token: and rest
                rest = line[len('token:'):].strip()
                # kind is first word
                parts = rest.split(None, 1)
                kind = parts[0]
                remainder = parts[1] if len(parts)>1 else ''
                # find quoted spelling
                spelling = None
                if '"' in remainder:
                    first = remainder.find('"')
                    last = remainder.rfind('"')
                    if first != last:
                        spelling = remainder[first+1:last]
                # location bracket
                loc = None
                if '[' in remainder and ']' in remainder:
                    try:
                        bracket = remainder[remainder.find('[')+1:remainder.rfind(']')]
                        # bracket like 1:1 - 1:4
                        loc = bracket
                    except:
                        loc = None
                out.append({'kind': kind, 'spelling': spelling, 'loc': loc, 'raw': line})
            else:
                # include raw line
                out.append({'raw': line})
        except Exception as e:
            out.append({'raw': line, 'parse_error': str(e)})
    return out

def _parse_clang_ast_json(ast_json_str: str) -> Any:
    """
    clang -Xclang -ast-dump=json prints JSON; try to parse it.
    If parsing fails, return the raw string.
    """
    try:
        # sometimes clang may print multiple JSON objects or warnings before JSON.
        # Try to find first { ... } block start.
        first = ast_json_str.find('{')
        if first == -1:
            return ast_json_str
        candidate = ast_json_str[first:]
        return json.loads(candidate)
    except Exception:
        # return raw as fallback
        return ast_json_str

# ------------------ Public API ------------------ #
def get_tokens(code: str, is_cpp: bool = False) -> List[Dict[str, Any]]:
    """
    Return a list of token dicts for the given code snippet.
    Uses libclang when available, otherwise falls back to clang subprocess parsing.
    """
    code = _fix_escaped_newlines(code)
    if use_libclang:
        try:
            return _get_tokens_libclang(code, is_cpp=is_cpp)
        except Exception as e:
            # fallback to subprocess
            print(f"libclang tokenization failed ({e}), falling back to clang subprocess.", file=sys.stderr)

    tokens_dump, _ = _run_clang_subprocess(code, is_cpp=is_cpp)
    return _parse_clang_tokens_dump(tokens_dump)

def get_ast(code: str, is_cpp: bool = False) -> Any:
    """
    Return a nested dict representing AST for the code.
    Prefer libclang (returns a structured dict). Otherwise return clang's JSON AST if available.
    """
    code = _fix_escaped_newlines(code)
    if use_libclang:
        try:
            return _get_ast_libclang(code, is_cpp=is_cpp)
        except Exception as e:
            print(f"libclang AST parse failed ({e}), falling back to clang subprocess.", file=sys.stderr)

    _, ast_out = _run_clang_subprocess(code, is_cpp=is_cpp)
    return _parse_clang_ast_json(ast_out)

# ------------------ Helper: process file ------------------ #
def process_snippets_file(path: str, is_cpp: bool = True):
    with open(path, 'r', encoding='utf-8') as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip('\n')
            if not line.strip():
                continue
            code = _fix_escaped_newlines(line)
            print("="*80)
            print(f"Snippet #{i} (length {len(code)} chars):")
            print("-" * 40)
            try:
                tokens = get_tokens(code, is_cpp=is_cpp)
                ast = get_ast(code, is_cpp=is_cpp)
                print("Tokens (first 80 items):")
                print(json.dumps(tokens[:80], indent=2))
                print("\nAST (top-level):")
                # pretty print large AST carefully
                if isinstance(ast, (dict, list)):
                    print(json.dumps(ast, indent=2)[:20000])  # cap to avoid huge outputs
                else:
                    print(str(ast)[:20000])
            except Exception as e:
                print(f"Error processing snippet #{i}: {e}", file=sys.stderr)
            print("\n")


path = "./clean_data/sample.txt"
process_snippets_file(path)
