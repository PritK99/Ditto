"""
This script takes code and converts it to distance matrix obtained from AST after parsing.
"""
import re
import ast
import ctypes
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from tree_sitter import Language, Parser, Node
from typing import List, Tuple, Optional, Dict, Any

C_LIB_PATH   = os.path.expanduser('~/tree-sitter-c/c_language_lib.so')
CPP_LIB_PATH = os.path.expanduser('~/tree-sitter-cpp/cpp_language_lib.so')

def clean_code(code: str) -> str:
    code = code.replace("\\n", "\n")
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

# This function computes the entire distance matrix
def calculate_all_pairwise_distances(tokens: List[Dict[str, Any]]):
    dist_matrix = np.zeros((len(tokens), len(tokens)), dtype=int)

    for i, a in enumerate(tokens):
        for j, b in enumerate(tokens[i:], start=i):
            rep_a, rep_b = a['rep_node'], b['rep_node']
            if rep_a is None or rep_b is None:
                dist = -1
            elif rep_a == rep_b:
                dist = 0
            else:
                lca, dist = calculate_tree_distance_nodes(rep_a, rep_b)

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  

    return dist_matrix

# This function only computes triu
# We can work with triu because we know our matrix is skew symmetric i.e. O(n^2) -> O(((n)*(n+1)/2)) in terms of memory
# Further, we can optimize it as the diagonal entries are always 0 i.e. O(((n)*(n+1)/2)) -> O(((n)*(n-1)/2)) in terms of memory
def calculate_triu_pairwise_distances(tokens):
    triu = []

    n = len(tokens)
    for i in range(n):
        a = tokens[i]
        rep_a = a['rep_node']

        for j in range(i+1, n):
            b = tokens[j]
            rep_b = b['rep_node']

            if rep_a == rep_b:
                dist = 0
            else:
                lca, dist = calculate_tree_distance_nodes(rep_a, rep_b)

            triu.append(dist)

    return triu

# Given a pair of token indexes i and j, this function returns its distance using triu
# The triu vector that we have is defined as upper triangular matrix and hence it corresponds to all positive distances
# The lower triangular matrix will be same as upper, but with negative sign.
# The maximum value that our triu can take has to be determined emperically since we are dealing with a tree structure
# Our tree has abstract nodes which are parents, and hence we can't simply say max distance is 1000
# To make everything positive, we add by max_dist
# We need to do this for nn.Embeddings()
def get_dist_from_triu(i, j, triu, max_dist = 1000):
    # Base Case
    if (i == j):
        return 0    
    
    sign = 1

    if i > j:
        i, j = j, i    # Only using the upper triangular matrix
        sign = -1

    num_tokens = int((1 + math.isqrt(1 + 8*len(triu))) // 2)    # This is formula for num_tokens from triu length

    idx = (i * (2*num_tokens - i - 1)) // 2 + (j - i - 1)

    return sign*(triu[idx]) + max_dist

if __name__ == "__main__":
    is_cpp = False
    base_str = "cpp" if is_cpp else "c"
    parser = make_parser(is_cpp)

    input_file = f"../data/{base_str}_cleaned_tokens.csv"
    output_file = f"../data/{base_str}_tokens_with_lca_dist.csv"

    cols = ["line_number", "transformed_tokens", "var_dict", "func_dict", "lit_dict", "struct_dict", "class_dict", "dist"]

    pd.DataFrame(columns=cols).to_csv(
        output_file, index=False
    )

    i = 0
    # Reading and writing each row at a time because the resultant df has a size of ~15GBs 
    for chunk in tqdm(pd.read_csv(input_file, chunksize=1), total=50000):    # The tqdm used here is dummy and just for viewing progress  
        row = chunk.iloc[0]
        code_tokens = ast.literal_eval(row["transformed_tokens"])

        code = " ".join(code_tokens)
        clean_line = clean_code(code)
        clean_line_bytes = clean_line.encode("utf8")
        tree = parser.parse(clean_line_bytes)
        root = tree.root_node

        leaves = collect_leaf_nodes(root, clean_line_bytes)
        merged_tokens = build_and_merge_tokens(leaves, clean_line_bytes)
        triu = calculate_triu_pairwise_distances(merged_tokens)

        chunk["dist"] = str(triu)

        chunk.to_csv(output_file, mode="a", header=False, index=False)
        
        # We only process first 50K rows for C since we have only ~48K rows for C++
        i += 1
        if (not is_cpp) and  (i > 50000):
            break