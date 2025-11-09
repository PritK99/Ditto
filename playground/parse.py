from clang.cindex import Config, Index, TokenKind, CursorKind

# Path to libclang.so
Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

index = Index.create()
tu = index.parse('cleaned_simple.c')

def location_to_offset(loc):
    """Convert a SourceLocation to a numeric offset for comparison."""
    return loc.offset

def find_cursor_for_token(cursor, token):
    """Recursively find the AST node that covers the token."""
    start_offset = location_to_offset(cursor.extent.start)
    end_offset = location_to_offset(cursor.extent.end)
    token_offset = location_to_offset(token.location)

    if start_offset <= token_offset <= end_offset:
        for child in cursor.get_children():
            result = find_cursor_for_token(child, token)
            if result:
                return result
        return cursor
    return None

# Dictionaries to store mapping
var_dict = {}
func_dict = {}
lit_dict = {}

# Counters for generating names
var_counter = 0
func_counter = 0
lit_counter = 0

# Collect transformed code tokens
transformed_tokens = []

for token in tu.get_tokens(extent=tu.cursor.extent):
    new_token = token.spelling  # default: leave as-is

    if token.kind == TokenKind.LITERAL:
        if token.spelling not in lit_dict:
            lit_dict[token.spelling] = f"lit{lit_counter}"
            lit_counter += 1
        new_token = lit_dict[token.spelling]

    elif token.kind == TokenKind.IDENTIFIER:
        cursor = find_cursor_for_token(tu.cursor, token)
        if cursor:
            # Function declaration
            if cursor.kind == CursorKind.FUNCTION_DECL:
                if token.spelling not in func_dict:
                    func_dict[token.spelling] = f"func{func_counter}"
                    func_counter += 1
                new_token = func_dict[token.spelling]

            # Variable declaration (global or local)
            elif cursor.kind == CursorKind.VAR_DECL:
                if token.spelling not in var_dict:
                    var_dict[token.spelling] = f"var{var_counter}"
                    var_counter += 1
                new_token = var_dict[token.spelling]

            # Parameter declaration
            elif cursor.kind == CursorKind.PARM_DECL:
                if token.spelling not in var_dict:
                    var_dict[token.spelling] = f"var{var_counter}"
                    var_counter += 1
                new_token = var_dict[token.spelling]

            # Usage/reference (DECL_REF_EXPR)
            elif cursor.kind == CursorKind.DECL_REF_EXPR:
                if token.spelling in var_dict:
                    new_token = var_dict[token.spelling]
                elif token.spelling in func_dict:
                    new_token = func_dict[token.spelling]
                # else leave as-is

    transformed_tokens.append(new_token)

# Reconstruct code with spaces between tokens
final_code = transformed_tokens

print("---- Transformed Code ----")
print(final_code)
print("\n---- Variable Dict ----")
print(var_dict)
print("\n---- Function Dict ----")
print(func_dict)
print("\n---- Literal Dict ----")
print(lit_dict)
