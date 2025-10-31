#!/usr/bin/env python3
import re
import json
import ctypes
import os
import sys
import csv
import unicodedata
from tree_sitter import Language, Parser, Node
from typing import List, Tuple, Optional, Dict, Any

# -------------------------------------------------------------
# Configuration
# Replace the paths below with your actual path i.e. path to .so file after generating them from corresponding githubs.
# -------------------------------------------------------------
CPP_LIB_PATH = os.path.expanduser('~/tree-sitter-cpp/cpp_language_lib.so')
C_LIB_PATH   = os.path.expanduser('~/tree-sitter-c/c_language_lib.so')

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

# -------------------------------------------------------------
# Functions for LCA
# -------------------------------------------------------------
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

# ==========================================================
# Functions for Token Management
# ==========================================================
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

        # --- Merge preprocessor (#include etc.)
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

        # --- Merge multi-part string/char literals
        parent = t['nodes'][0].parent
        if parent and parent.type in {'string_literal', 'char_literal'}:
            nodes_to_merge = t['nodes']
            combined_text = t['text']
            j = i + 1
            while j < len(tokens):
                next_check = tokens[j]
                # Check if the next token also belongs to the *same* string/char literal parent
                if next_check['nodes'][0].parent == parent:
                    nodes_to_merge += next_check['nodes']
                    combined_text += next_check['text']
                    j += 1
                else:
                    break
            
            # Use the end byte of the last token merged
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

# ==========================================================
# Obfuscation Code 
# ==========================================================
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
        # 1) If token text looks like namespace/qualified name, keep it
        if is_namespace_like_text(txt):
            return txt, var_counter, lit_counter, func_counter

        # 2) If token was detected as a stream-object candidate (via token adjacency to << or >>), keep it
        if txt in stream_candidates:
            return txt, var_counter, lit_counter, func_counter

        parent = node.parent
        ptype = safe_type(parent)

        # 3) Function declaration / definition -> record and obfuscate
        if ptype in {"function_declarator", "function_definition"}:
            if txt not in func_map:
                func_map[txt] = f"func{func_counter}"
                func_counter += 1
            declared_funcs.add(txt)
            return func_map[txt], var_counter, lit_counter, func_counter

        # 4) Function call detection: search ancestors for a 'call' type
        cur = node
        is_call = False
        while cur:
            if "call" in safe_type(cur):
                is_call = True
                break
            cur = cur.parent

        if is_call:
            # If user-defined (declared in this snippet) -> obfuscate
            if txt in declared_funcs:
                if txt not in func_map:
                    func_map[txt] = f"func{func_counter}"
                    func_counter += 1
                return func_map[txt], var_counter, lit_counter, func_counter
            # Else: external/stdlib -> keep as-is
            return txt, var_counter, lit_counter, func_counter

        # 5) Otherwise treat as variable (obfuscate)
        if txt not in var_map:
            var_map[txt] = f"var{var_counter}"
            var_counter += 1
        return var_map[txt], var_counter, lit_counter, func_counter

    # fallback unchanged
    return txt, var_counter, lit_counter, func_counter

# ==========================================================
# Functions for calculating Pairwise Distance 
# ==========================================================
def calculate_all_pairwise_distances(tokens: List[Dict[str, Any]]) -> Tuple[
        List[Dict[str, Any]], Dict[str, str], Dict[str, str], Dict[str, str], List[str]]:
    results = []
    var_map, lit_map, func_map = {}, {}, {}
    var_counter = lit_counter = func_counter = 0

    # --- PASS A: collect declared functions and detect stream-like identifiers
    declared_funcs = set()
    stream_candidates = set()
    tok_texts = [t['text'].strip() for t in tokens]

    # Detect declared functions from AST parent types
    for t in tokens:
        node = t.get('nodes', [None])[0]
        if node is None:
            continue
        parent = node.parent
        if parent is not None:
            ptype = getattr(parent, "type", "")
            if ptype in ("function_declarator", "function_definition"):
                declared_funcs.add(t['text'].strip())

    # Detect patterns for stream-objects: identifier adjacent to << or >>
    for i, txt in enumerate(tok_texts):
        if txt in ("<<", ">>"):
            # left neighbor
            if i-1 >= 0:
                left = tok_texts[i-1]
                # crude heuristic: if left looks like an identifier (letters, underscores)
                if re.match(r'^[A-Za-z_]\w*$', left):
                    stream_candidates.add(left)
            # right neighbor
            if i+1 < len(tok_texts):
                right = tok_texts[i+1]
                if re.match(r'^[A-Za-z_]\w*$', right):
                    stream_candidates.add(right)

    # Also detect "using namespace std" pattern to keep std as namespace
    for i in range(len(tok_texts)-2):
        if tok_texts[i] == "using" and tok_texts[i+1] == "namespace" and tok_texts[i+2] == "std":
            stream_candidates.add("std")

    # --- PASS B: obfuscate tokens using declared_funcs and stream_candidates
    obfuscated = []
    for t in tokens:
        obf, var_counter, lit_counter, func_counter = obfuscate_token_text(
            t, var_map, lit_map, func_map,
            var_counter, lit_counter, func_counter,
            declared_funcs, stream_candidates
        )
        obfuscated.append(obf)

    # --- Pairwise distance computation (unchanged)
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
                'A_raw': a['text'].strip(),
                'B_raw': b['text'].strip(),
                'A_obf': A_obf,
                'B_obf': B_obf,
                'LCA_Type': lca.type if lca else 'N/A',
                'Distance': dist
            })

    return results, var_map, lit_map, func_map, obfuscated

# ==========================================================
# Driver Function
# ==========================================================
def main_worker(code_string: str, is_cpp: bool):
    try:
        parser = make_parser(is_cpp)
        code_bytes = code_string.encode('utf8')
        tree = parser.parse(code_bytes)
        root = tree.root_node

        leaves = collect_leaf_nodes(root, code_bytes)
        merged_tokens = build_and_merge_tokens(leaves, code_bytes)
        results, var_map, lit_map, func_map, obf_tokens = calculate_all_pairwise_distances(merged_tokens)

        return {
            'Code': code_string.replace('\n', '\\n'),
            'Obfuscated_Tokens': json.dumps(obf_tokens),
            'Variable_Map': json.dumps(var_map),
            'Literal_Map': json.dumps(lit_map),
            'Function_Map': json.dumps(func_map)
        }
    except Exception as e:
        print("Error:", e)
        return None

# ==========================================================
# Utils
# ==========================================================
def write_to_csv(data, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Code', 'Obfuscated_Tokens', 'Variable_Map', 'Literal_Map', 'Function_Map'])
        writer.writeheader()
        writer.writerows(data)

def remove_escape_characters(code: str) -> str:
    code = unicodedata.normalize("NFKC", code)

    try:
        code = bytes(code, "utf-8").decode("unicode_escape")
    except Exception:
        code = code.encode("utf-8", "ignore").decode("unicode_escape", "ignore")

    code = re.sub(r"\\[ntr]", lambda m: {"\\n": "\n", "\\t": "\t", "\\r": "\r"}[m.group(0)], code)
    code = "".join(ch for ch in code if ch.isprintable() or ch in "\n\t")

    code = re.sub(r"//.*?(?=\n|$)", "", code)                 # single-line //
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)    # multi-line /* ... */
    code = re.sub(r"\n{3,}", "\n\n", code)
    code = code.replace("\ufeff", "").replace("\u200b", "")

    return code.strip()

# -------------------------------------------------------------
# Driver Code
# -------------------------------------------------------------
if __name__ == "__main__":
    is_cpp = True if len(sys.argv) > 1 and sys.argv[1] == "cpp" else False
    input_file = "unpaired_cpp.txt" if is_cpp else "unpaired_c.txt"
    out_file = "cpp_tokens.csv" if is_cpp else "c_tokens.csv"

    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found.")
        sys.exit(1)

    with open(input_file) as f:
        snippets = [remove_escape_characters(line.strip()) for line in f if line.strip()]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_results = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as ex:
        futures = [ex.submit(main_worker, s, is_cpp) for s in snippets]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                all_results.append(r)

    write_to_csv(all_results, out_file)
    print(f"Wrote {len(all_results)} entries to {out_file}")