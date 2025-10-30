#!/usr/bin/env python3
import os
import ctypes
import itertools
import re
import csv
import json
from typing import List, Tuple, Optional, Dict, Any
from tree_sitter import Language, Parser, Node
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==========================================================
# Configuration - ADJUST THESE PATHS
# ==========================================================
# NOTE: Replace with the actual path to your compiled C grammar library
GRAMMAR_LIB_PATH = os.path.expanduser('~/tree-sitter-c/c_language_lib.so') # 🎯 Changed for C
CSV_OUTPUT_PATH = 'cpp_tokens_1.csv' # Define C output file path
# Use all but one core, or a minimum of 1
MAX_WORKERS = max(1, 5)

_C_LANGUAGE: Optional[Language] = None

# Fallback C code

# ==========================================================
# Language Loader (C)
# ==========================================================
def load_c_language(lib_path: str) -> Language:
    """Loads the compiled Tree-sitter C grammar (memoized)."""
    global _C_LANGUAGE
    if _C_LANGUAGE is None:
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Compiled C grammar library not found at: {lib_path}")

        LANGUAGE_FN_NAME = 'tree_sitter_c'  # 🎯 Function exported by compiled C grammar
        lib = ctypes.CDLL(lib_path)
        lang_fn = getattr(lib, LANGUAGE_FN_NAME)
        lang_fn.restype = ctypes.c_void_p
        _C_LANGUAGE = Language(lang_fn())
    return _C_LANGUAGE

# ==========================================================
# Tree Utilities - UNCHANGED
# ==========================================================
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
# Token Collection and Merging (Adapted for C)
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
# Obfuscation (Adapted for C)
# ==========================================================
LITERAL_REGEX = re.compile(r'^[+-]?\d+(\.\d*)?([eE][+-]?\d+)?[uUlLlF]*$')

def obfuscate_token_text(token: Dict[str, Any], var_map, lit_map, var_counter, lit_counter):
    txt = token['text'].strip()
    types = set(token.get('types', []))
    LITERAL_TYPES = {
        'number_literal', 'string_literal', 'char_literal', 'true', 'false'
    }

    if 'identifier' in types or 'field_identifier' in types: # Include C field_identifier
        if txt not in var_map:
            var_map[txt] = f"var{var_counter}"
            var_counter += 1
        return var_map[txt], var_counter, lit_counter

    is_literal_type = types.intersection(LITERAL_TYPES)
    is_quoted = (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'"))
    is_numeric = LITERAL_REGEX.match(txt)
    
    if is_literal_type or is_quoted or is_numeric:
        if txt not in lit_map:
            lit_map[txt] = f"lit{lit_counter}"
            lit_counter += 1
        return lit_map[txt], var_counter, lit_counter

    if txt.startswith('#'):
        return txt, var_counter, lit_counter

    return txt, var_counter, lit_counter

# ==========================================================
# Pairwise Distance Computation - UNCHANGED Logic
# ==========================================================
def calculate_all_pairwise_distances(tokens: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str], List[str]]:
    """Calculates distances, builds maps, and returns all results."""
    results = []
    var_map, lit_map = {}, {}
    var_counter = lit_counter = 0

    obfuscated = []
    for t in tokens:
        obf, var_counter, lit_counter = obfuscate_token_text(t, var_map, lit_map, var_counter, lit_counter)
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
            
            # Use pre-calculated obfuscated strings
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
    return results, var_map, lit_map, obfuscated

# ==========================================================
# Worker Function (C)
# ==========================================================
def main_worker(c_code_string: str) -> Optional[Dict[str, Any]]:
    """Processes a single C snippet and returns the data row dictionary, including distances."""
    try:
        # Strip any leading/trailing whitespace including newlines
        c_code_string = c_code_string.strip()
        c_lang = load_c_language(GRAMMAR_LIB_PATH)
        parser = Parser()
        parser.language = c_lang
        code_bytes = c_code_string.encode('utf8')
        tree = parser.parse(code_bytes)
        root = tree.root_node

        leaves = collect_leaf_nodes(root, code_bytes)
        merged_tokens = build_and_merge_tokens(leaves, code_bytes)

        results = []
        var_map, lit_map = {}, {}
        var_counter = lit_counter = 0

        obfuscated = []
        for t in merged_tokens:
            obf, var_counter, lit_counter = obfuscate_token_text(t, var_map, lit_map, var_counter, lit_counter)
            obfuscated.append(obf)

        # 🎯 FIX: Calculate pairwise distances to get the full data
        # results, var_map, lit_map, obf_tokens = calculate_all_pairwise_distances(merged_tokens)
        # dist_values = [r['Distance'] for r in results]

        return {
            'C_Code': c_code_string.replace('\n', '\\n'), # Keep column name for output consistency
            # 'Distances_Flattened': json.dumps(dist_values),
            'Obfuscated_Tokens': json.dumps(obfuscated),
            'Literal_Map': json.dumps(lit_map),
            'Variable_Map': json.dumps(var_map)
        }
    except Exception as e:
        # It's helpful to see the error for debugging
        # print(f"[Worker error for snippet starting with '{c_code_string[:30].strip()}...'] {e}")
        return None

# ==========================================================
# CSV Writing - UNCHANGED Logic
# ==========================================================
def write_all_results_to_csv(data: List[Dict[str, Any]], filepath: str):
    fields = ['C_Code', 'Obfuscated_Tokens', 'Literal_Map', 'Variable_Map']
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

def remove_escape_characters(code: str) -> str:
    """
    Remove escape characters like \n, \t, etc., and handle them properly.
    """
    # First, handle literal '\n' and other escape characters
    code = bytes(code, 'utf-8').decode('unicode_escape')
    
    # Replacing //n with the literal string /n
    code = re.sub(r'\\n', '\n', code)    # This is for case where original code has \n which becomes \\n in data
    
    return code

# ==========================================================
# Main Execution (Parallel) - Adapted for C file
# ==========================================================
if __name__ == "__main__":
    
    INPUT_FILE = "unpaired_c.txt" # 🎯 Changed input file name
    
    try:
        if os.path.exists(INPUT_FILE):
            with open(INPUT_FILE) as f:
                snippets = [remove_escape_characters(line.strip()) for line in f if line.strip()]
        else:
            print(f"⚠️ WARNING: '{INPUT_FILE}' not found. Running default C_CODE_STRING once.")
            snippets = [C_CODE_STRING.strip()]

        if not snippets:
            print(f"⚠️ No snippets found in '{INPUT_FILE}'. Exiting.")
            exit()
            
        print(f"🧩 Loaded {len(snippets)} C snippets — starting with {MAX_WORKERS} workers...")

        all_results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Note: Using the worker function for C snippets
            futures = {executor.submit(main_worker, s): s for s in snippets}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing C Snippets"):
                res = fut.result()
                if res:
                    all_results.append(res)

        if all_results:
            write_all_results_to_csv(all_results, CSV_OUTPUT_PATH)
            print(f"✅ Wrote {len(all_results)} entries to {CSV_OUTPUT_PATH}")
        else:
            print("⚠️ No results processed successfully.")
            
    except FileNotFoundError:
        # Should be caught by the check above, but here for robustness
        pass 
    except Exception as e:
        print(f"❌ Unexpected error in main execution: {e}")
