import re
import json
import ctypes
import os
import pandas as pd
import sys
import csv
import unicodedata
from tree_sitter import Language, Parser, Node
from typing import List, Tuple, Optional, Dict, Any

C_LIB_PATH   = os.path.expanduser('~/tree-sitter-c/c_language_lib.so')
CPP_LIB_PATH = os.path.expanduser('~/tree-sitter-cpp/cpp_language_lib.so')

def clean_code(code: str) -> str:
    code = code.replace("\\n", "\n")
    code = bytes(code, "utf-8").decode("unicode_escape")
    code = re.sub(r'#\s*include\s*(<[^>]*>|"[^"]*")', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//[^\n]*', '', code)
    code = "\n".join(line for line in code.splitlines() if line.strip())
    return code

def load_language(lib_path: str, symbol: str) -> Language:
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Missing: {lib_path}")
    lib = ctypes.CDLL(lib_path)
    fn = getattr(lib, symbol)
    fn.restype = ctypes.c_void_p
    return Language(fn())

def make_parser(is_cpp: bool) -> Parser:
    if is_cpp:
        lang = load_language(CPP_LIB_PATH, "tree_sitter_cpp")
    else:
        lang = load_language(C_LIB_PATH, "tree_sitter_c")
    parser = Parser()
    parser.language = lang
    return parser

def print_tree(node, code_bytes, indent="", is_last=True):
    prefix = indent + ("└── " if is_last else "├── ")
    snippet = code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="ignore")
    print(f"{prefix}{node.type}: {snippet!r}")
    new_indent = indent + ("    " if is_last else "│   ")
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        print_tree(child, code_bytes, new_indent, i == child_count - 1)

def get_ancestors(node: Node) -> List[Node]:
    ancestors = []
    cur = node
    while cur:
        ancestors.append(cur)
        cur = cur.parent
    return ancestors

def find_lca_of_two_nodes(a: Node, b: Node) -> Optional[Node]:
    if a == b:
        return a
    ancestors_a = set(get_ancestors(a))
    cur = b
    while cur:
        if cur in ancestors_a:
            return cur
        cur = cur.parent
    return None

def find_lca_of_nodes(nodes: List[Node]) -> Optional[Node]:
    if not nodes:
        return None
    lca = nodes[0]
    for n in nodes[1:]:
        lca = find_lca_of_two_nodes(lca, n)
        if not lca:
            return None
    return lca

def calculate_tree_distance_nodes(a: Node, b: Node) -> Tuple[Optional[Node], int]:
    lca = find_lca_of_two_nodes(a, b)
    if not lca:
        return None, -1

    def depth_to_ancestor(n: Node, anc: Node) -> int:
        d = 0
        cur = n
        while cur and cur != anc:
            cur = cur.parent
            d += 1
        return d if cur == anc else -1

    return lca, depth_to_ancestor(a, lca) + depth_to_ancestor(b, lca)

def collect_leaf_nodes(root: Node, code_bytes: bytes) -> List[Node]:
    leaves = []
    def walk(n):
        if not n.children:
            txt = code_bytes[n.start_byte:n.end_byte].decode("utf8")
            if txt and not txt.isspace():
                leaves.append(n)
            return
        for c in n.children:
            walk(c)
    walk(root)
    return leaves

def build_and_merge_tokens(leaf_nodes: List[Node], code_bytes: bytes) -> List[Dict[str, Any]]:
    tokens = []
    for n in leaf_nodes:
        txt = code_bytes[n.start_byte:n.end_byte].decode('utf8')
        tokens.append({
            'text': txt,
            'start': n.start_byte,
            'end': n.end_byte,
            'types': [n.type],
            'nodes': [n],
            'rep_node': n
        })

    merged = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        next_t = tokens[i+1] if i+1 < len(tokens) else None

        txt = t['text'].strip()
        nxt_txt = next_t['text'].strip() if next_t else ''

        if txt == '#' and nxt_txt in ('include', 'define', 'pragma', 'error', 'warning', 'if', 'ifdef', 'ifndef', 'elif', 'else', 'endif'):
            combined = f"#{nxt_txt}"
            nodes = t['nodes'] + next_t['nodes']
            rep = find_lca_of_nodes(nodes)
            merged.append({
                'text': combined,
                'start': t['start'],
                'end': next_t['end'],
                'types': t['types'] + next_t['types'],
                'nodes': nodes,
                'rep_node': rep
            })
            i += 2
            continue

        parent = t['nodes'][0].parent
        if parent and parent.type in {'string_literal', 'char_literal'}:
            nodes_to_merge = t['nodes']
            combined_text = t['text']
            j = i + 1
            while j < len(tokens):
                next_check = tokens[j]
                if next_check['nodes'][0].parent == parent:
                    nodes_to_merge += next_check['nodes']
                    combined_text += next_check['text']
                    j += 1
                else:
                    break

            end_byte = tokens[j - 1]['end'] if j > i else t['end']
            
            rep = find_lca_of_nodes(nodes_to_merge)
            merged.append({
                'text': combined_text,
                'start': t['start'],
                'end': end_byte,
                'types': [parent.type],
                'nodes': nodes_to_merge,
                'rep_node': rep
            })
            i = j
            continue

        merged.append(t)
        i += 1
    return merged

LITERAL_REGEX = re.compile(r'^[+-]?\d+(\.\d*)?([eE][+-]?\d+)?[uUlLlF]*$')

# never obfuscate these stream identifiers
ALWAYS_KEEP = {"cin", "cout", "cerr", "clog", "namespace", "std"}

def obfuscate_token_text(token, var_map, lit_map, func_map,
                         var_counter, lit_counter, func_counter,
                         declared_funcs: set,
                         stream_candidates: set):
    txt = token['text'].strip()
    types = set(token.get('types', []))
    node = token['nodes'][0] if token.get('nodes') else None

    if txt in ALWAYS_KEEP:
        return txt, var_counter, lit_counter, func_counter

    LITERAL_TYPES = {'number_literal', 'string_literal', 'char_literal', 'true', 'false'}

    def safe_type(n):
        try:
            return n.type
        except Exception:
            return ""

    def is_identifier_node(n):
        return n is not None and safe_type(n) in ("identifier", "field_identifier")

    def is_namespace_like_text(s: str) -> bool:
        return "::" in s or s == "std"

    # preprocessor
    if txt.startswith("#"):
        return txt, var_counter, lit_counter, func_counter

    # literals
    is_literal_type = types.intersection(LITERAL_TYPES)
    is_quoted = (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'"))
    is_numeric = LITERAL_REGEX.match(txt)
    if is_literal_type or is_quoted or is_numeric:
        if txt not in lit_map:
            lit_map[txt] = f"lit{lit_counter}"
            lit_counter += 1
        return lit_map[txt], var_counter, lit_counter, func_counter

    # identifiers
    if is_identifier_node(node):
        if is_namespace_like_text(txt):
            return txt, var_counter, lit_counter, func_counter
        if txt in stream_candidates:
            return txt, var_counter, lit_counter, func_counter

        parent = node.parent
        ptype = safe_type(parent)
        if ptype in {"function_declarator", "function_definition"}:
            if txt not in func_map:
                func_map[txt] = f"func{func_counter}"
                func_counter += 1
            declared_funcs.add(txt)
            return func_map[txt], var_counter, lit_counter, func_counter
        
        cur = node
        is_call = False
        while cur:
            if "call" in safe_type(cur):
                is_call = True
                break
            cur = cur.parent

        if is_call:
            if txt in declared_funcs:
                if txt not in func_map:
                    func_map[txt] = f"func{func_counter}"
                    func_counter += 1
                return func_map[txt], var_counter, lit_counter, func_counter
            return txt, var_counter, lit_counter, func_counter
        
        if txt not in var_map:
            var_map[txt] = f"var{var_counter}"
            var_counter += 1
        return var_map[txt], var_counter, lit_counter, func_counter
    return txt, var_counter, lit_counter, func_counter

def calculate_all_pairwise_distances(tokens: List[Dict[str, Any]]) -> Tuple[
        List[Dict[str, Any]], Dict[str, str], Dict[str, str], Dict[str, str], List[str]]:
    results = []
    var_map, lit_map, func_map = {}, {}, {}
    var_counter = lit_counter = func_counter = 0

    declared_funcs = set()
    stream_candidates = set()
    tok_texts = [t['text'].strip() for t in tokens]

    for t in tokens:
        node = t.get('nodes', [None])[0]
        if node is None:
            continue
        parent = node.parent
        if parent is not None:
            ptype = getattr(parent, "type", "")
            if ptype in ("function_declarator", "function_definition"):
                declared_funcs.add(t['text'].strip())

    for i, txt in enumerate(tok_texts):
        if txt in ("<<", ">>"):
            if i-1 >= 0:
                left = tok_texts[i-1]
                if re.match(r'^[A-Za-z_]\w*$', left):
                    stream_candidates.add(left)
            if i+1 < len(tok_texts):
                right = tok_texts[i+1]
                if re.match(r'^[A-Za-z_]\w*$', right):
                    stream_candidates.add(right)

    for i in range(len(tok_texts)-2):
        if tok_texts[i] == "using" and tok_texts[i+1] == "namespace" and tok_texts[i+2] == "std":
            stream_candidates.add("std")

    obfuscated = []
    for t in tokens:
        obf, var_counter, lit_counter, func_counter = obfuscate_token_text(
            t, var_map, lit_map, func_map,
            var_counter, lit_counter, func_counter,
            declared_funcs, stream_candidates
        )
        obfuscated.append(obf)

    for i, a in enumerate(tokens):
        for j, b in enumerate(tokens[i:], start=i):
            rep_a, rep_b = a['rep_node'], b['rep_node']
            if rep_a is None or rep_b is None:
                lca, dist = None, -1
            elif rep_a == rep_b:
                lca, dist = rep_a, 0
            else:
                lca, dist = calculate_tree_distance_nodes(rep_a, rep_b)
            A_obf = obfuscated[i]
            B_obf = obfuscated[j]
            results.append({
                'A_text': a['text'].strip(),
                'B_text': b['text'].strip(),
                'A_pos':  (a['start'], a['end']),
                'B_pos':  (b['start'], b['end']),
                'Distance': dist
            })

    return results, var_map, lit_map, func_map, obfuscated

if __name__ == "__main__":
    is_cpp = True
    parser = make_parser(is_cpp)

    file_path = "unpaired_cpp.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    for line in all_lines:
        clean_line = clean_code(line)
        print(clean_line)

        clean_line_bytes = clean_line.encode("utf8")
        tree = parser.parse(clean_line_bytes)
        root = tree.root_node

        print_tree(root, clean_line_bytes)

        leaves = collect_leaf_nodes(root, clean_line_bytes)
        merged_tokens = build_and_merge_tokens(leaves, clean_line_bytes)
        results, var_map, lit_map, func_map, obf_tokens = calculate_all_pairwise_distances(merged_tokens)

        for r in results:
            A = r['A_text']
            B = r['B_text']
            A_s, A_e = r['A_pos']
            B_s, B_e = r['B_pos']
            dist = r['Distance']

            print(f"({A}, {B}) ([{A_s} {A_e}], [{B_s} {B_e}]) {dist}")