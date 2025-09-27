"""
This script performs tokenization and AST parsing on C/C++ code snippets using libclang.
It converts code to tokens (lexing) and constructs an abstract syntax tree (AST).
"""
import sys
import json
from typing import List, Dict, Any
from clang import cindex

# Set the libclang shared library path
cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def fix_escaped_newlines(line: str) -> str:
    """
    Converts escaped sequences in a line to actual characters.
    E.g., '\\n' -> newline, '\\t' -> tab, '\\\\' -> backslash.

    Args:
        line (str): code snippet

    Returns:
        str: Modified line with escape sequences replaced.
    """
    if "\\n" in line or "\\t" in line or "\\\\" in line:
        return line.encode('utf-8').decode('unicode_escape', errors='replace')
    return line

def get_tokens_libclang(code: str, is_cpp: bool = False) -> List[Dict[str, Any]]:
    """
    Tokenizes C/C++ code using libclang.

    Args:
        code (str): Full code snippet.
        is_cpp (bool): True for C++, False for C.

    Returns:
        List[Dict[str, Any]]: List of tokens with kind, spelling, and location.
    """
    index = cindex.Index.create()    # cindex is the python interface for libclang and index is the object
    filename = 'snippet.cpp' if is_cpp else 'snippet.c'
    args = ['-std=c++17'] if is_cpp else ['-std=c11']
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
    Recursively convert a libclang Cursor node to a serializable dict.

    Args:
        cursor (cindex.Cursor): libclang AST cursor.

    Returns:
        Dict[str, Any]: Nested dictionary representation of the AST node.
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


def get_ast_libclang(code: str, is_cpp: bool = False) -> Dict[str, Any]:
    """
    Parses C/C++ code into an AST using libclang.

    Args:
        code (str): Full code snippet.
        is_cpp (bool): True for C++, False for C.

    Returns:
        Dict[str, Any]: Nested dictionary representing the AST.
    """
    index = cindex.Index.create()
    filename = 'snippet.cpp' if is_cpp else 'snippet.c'
    args = ['-std=c++17'] if is_cpp else ['-std=c11']
    tu = index.parse(path=filename, args=args, unsaved_files=[(filename, code)], options=0)
    return _cursor_to_dict(tu.cursor)


# ------------------ Public API ------------------ #
def get_tokens(code: str, is_cpp: bool = False) -> List[Dict[str, Any]]:
    """
    Return a list of token dicts for a given code snippet using libclang.

    Args:
        code (str): Full code snippet.
        is_cpp (bool): True for C++, False for C.

    Returns:
        List[Dict[str, Any]]: Tokenized representation.
    """
    code = fix_escaped_newlines(code)
    return get_tokens_libclang(code, is_cpp=is_cpp)


def get_ast(code: str, is_cpp: bool = False) -> Dict[str, Any]:
    """
    Return a nested dict representing the AST for a code snippet using libclang.

    Args:
        code (str): Full code snippet.
        is_cpp (bool): True for C++, False for C.

    Returns:
        Dict[str, Any]: AST as nested dictionary.
    """
    code = fix_escaped_newlines(code)
    return get_ast_libclang(code, is_cpp=is_cpp)


# ------------------ Helper: process file ------------------ #
def process_snippets_file(path: str, is_cpp: bool = True):
    """
    Process a file of code snippets, printing tokens and AST for each snippet.

    Args:
        path (str): Path to input file (one snippet per line).
        is_cpp (bool): True if C++ code, False for C code.
    """
    with open(path, 'r', encoding='utf-8') as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip('\n')
            if not line.strip():    # Skipping blank lines 
                continue

            code = fix_escaped_newlines(line)    # Fixing the escape characters

            try:
                tokens = get_tokens(code, is_cpp=is_cpp)
                print(len(tokens))
                # ast = get_ast(code, is_cpp=is_cpp)
                # print("Tokens (first 80 items):")
                # print(json.dumps(tokens[:80], indent=2))
                # print("\nAST (top-level):")
                # print(json.dumps(ast, indent=2)[:20000])  # cap for large ASTs
            except Exception as e:
                print(f"Error processing snippet #{i}: {e}", file=sys.stderr)
            print("\n")


# Example usage
path = "./sample.txt"
process_snippets_file(path)
