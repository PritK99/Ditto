from clang.cindex import Config, Index, TokenKind, CursorKind
import tempfile
import os

# Path to libclang.so
Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

# Special cases
KEYWORDS = [
    "alignas", "alignof", "and", "and_eq", "asm",
    "atomic_cancel", "atomic_commit", "atomic_noexcept", "auto", "bitand",
    "bitor", "bool", "break", "case", "catch",
    "char", "char8_t", "char16_t", "char32_t", "class",
    "co_await", "co_return", "co_yield", "compl", "concept",
    "const", "consteval", "constexpr", "constinit", "const_cast",
    "continue", "decltype", "default", "delete", "do",
    "double", "dynamic_cast", "else", "enum", "explicit",
    "export", "extern", "false", "float", "for",
    "friend", "goto", "if", "inline", "int", "long",
    "mutable", "namespace", "new", "noexcept", "not",
    "not_eq", "nullptr", "operator", "or", "or_eq",
    "private", "protected", "public", "register", "reinterpret_cast",
    "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "synchronized",
    "template", "this", "thread_local", "throw", "true",
    "try", "typedef", "typeid", "typename", "union",
    "unsigned", "using", "virtual", "void", "volatile",
    "wchar_t", "while", "xor", "xor_eq",
    # C-only keywords
    "_Bool", "_Complex", "_Imaginary", "restrict"
]

def obfuscate_and_tokenize(code_text, is_cpp = False):
    """
    Transform C code text into mapped tokens.

    Args:
        code_text (str): Raw C code as a string.

    Returns:
        tuple: (transformed_code, var_dict, func_dict, lit_dict)
    """

    # Write code to temporary file for clang parsing
    index = Index.create()
    if (is_cpp):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as tmp:
            tmp.write(code_text)
            tmp_name = tmp.name

        tu = index.parse(tmp_name, args=["-std=c++17"])
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as tmp:
            tmp.write(code_text)
            tmp_name = tmp.name

        tu = index.parse(tmp_name)

    # Clean up temp file immediately after parsing
    os.remove(tmp_name)

    def location_to_offset(loc):
        """
        Convert a SourceLocation to a numeric offset for comparison.
        """
        return loc.offset

    def find_cursor_for_token(cursor, token):
        """
        Recursively find the AST node that covers the token.
        """
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
    struct_dict = {}
    class_dict = {}

    # Counters for generating names
    var_counter = 0
    func_counter = 0
    lit_counter = 0
    struct_counter = 0
    class_counter = 0

    # Collect transformed code tokens
    transformed_tokens = []

    i = 0
    raw_tokens =list(tu.get_tokens(extent=tu.cursor.extent))

    for token in tu.get_tokens(extent=tu.cursor.extent):
        new_token = token.spelling  # default: leave as-is

        # print(token.spelling, token.kind)

        # std doesn't adhere to a specific cursor type, and hence we need a special case
        # main is a reserved function in both C and C++
        # cout sometimes goes into COMPOUND_STMT
        # other special cases like override
        if token.spelling in KEYWORDS:
            new_token = token.spelling

        if (token.spelling == "std" or token.spelling == "main" or token.spelling == "cout" or token.spelling == "cin" or token.spelling == "override" or token.spelling == "vector"):
            new_token = token.spelling

        # Sometimes the datatypes after std become field declarations
        elif (token.spelling == "string" or token.spelling == "int" or token.spelling == "char" or token.spelling == "float" or token.spelling == "double"):
            new_token = token.spelling

        elif token.kind == TokenKind.LITERAL:
            if token.spelling not in lit_dict:
                lit_dict[token.spelling] = f"lit{lit_counter}"
                lit_counter += 1
            new_token = lit_dict[token.spelling]

        elif token.kind == TokenKind.IDENTIFIER:
            cursor = find_cursor_for_token(tu.cursor, token)
            if cursor:
                # print(token.spelling, cursor.kind)
                if cursor.kind == CursorKind.FUNCTION_DECL or cursor.kind == CursorKind.CXX_METHOD:
                    if token.spelling not in func_dict:
                        func_dict[token.spelling] = f"func{func_counter}"
                        func_counter += 1
                    new_token = func_dict[token.spelling]

                elif cursor.kind in (CursorKind.VAR_DECL, CursorKind.PARM_DECL):
                    if token.spelling not in var_dict:
                        var_dict[token.spelling] = f"var{var_counter}"
                        var_counter += 1
                    new_token = var_dict[token.spelling]

                elif cursor.kind == CursorKind.DECL_REF_EXPR:
                    if token.spelling in var_dict:
                        new_token = var_dict[token.spelling]
                    elif token.spelling in func_dict:
                        new_token = func_dict[token.spelling]
        
                elif cursor.kind == CursorKind.STRUCT_DECL:
                    if token.spelling not in struct_dict:
                        struct_dict[token.spelling] = f"struct{struct_counter}"
                        struct_counter += 1
                    new_token = struct_dict[token.spelling]

                elif cursor.kind == CursorKind.FIELD_DECL or cursor.kind == CursorKind.MEMBER_REF_EXPR or cursor.kind == CursorKind.MEMBER_REF:
                    if token.spelling not in var_dict:
                        var_dict[token.spelling] = f"var{var_counter}"
                        var_counter += 1
                    new_token = var_dict[token.spelling]
                
                elif cursor.kind == CursorKind.CLASS_DECL or cursor.kind == CursorKind.CONSTRUCTOR or cursor.kind == CursorKind.DESTRUCTOR:
                    if (cursor.kind == CursorKind.CONSTRUCTOR):    # The variables in constructor parameter also belong to constructor cursor
                        if token.spelling in class_dict:
                            new_token = class_dict[token.spelling]
                        else:
                            new_token = var_dict[token.spelling]

                    elif token.spelling not in class_dict:
                        class_dict[token.spelling] = f"class{class_counter}"
                        class_counter += 1
                        new_token = class_dict[token.spelling]
                    else:
                        new_token = class_dict[token.spelling]
                
                # This is the case when we use the defined struct and class
                # Since both belong to same cursorkind, we need to check fo the token spelling in both dict
                # The spelling has to exist in one dict compulsorily
                elif cursor.kind == CursorKind.TYPE_REF:
                    if token.spelling in struct_dict:
                        new_token = struct_dict[token.spelling]
                    
                    elif token.spelling in class_dict:
                        new_token = class_dict[token.spelling]

                # If nothing matches, we check all dictionaries and set the object to be variable or function if we dont find anything
                # This is meant to deal with cursor.COMPOUND_STMT type of tokens
                else:
                    # Brute force search
                    if token.spelling in var_dict:
                        new_token = var_dict[token.spelling]
                    elif token.spelling in func_dict:
                        new_token = func_dict[token.spelling]
                    elif token.spelling in struct_dict:
                        new_token = struct_dict[token.spelling]
                    elif token.spelling in lit_dict:
                        new_token = lit_dict[token.spelling]
                    elif token.spelling in class_dict:
                        new_token = class_dict[token.spelling]
                    # Make all unknown objects as variable if they dont like a function
                    else:
                        if (i+1 < len(raw_tokens) and raw_tokens[i+1].spelling == "("):
                            if token.spelling not in func_dict:
                                func_dict[token.spelling] = f"func{func_counter}"
                                func_counter += 1
                            new_token = func_dict[token.spelling]
                        elif token.spelling not in var_dict:
                            var_dict[token.spelling] = f"var{var_counter}"
                            var_counter += 1
                            new_token = var_dict[token.spelling]
                        else:
                            new_token = var_dict[token.spelling]


        # Strip newlines and escape characters
        transformed_tokens.append(new_token.replace("\n", "").replace("\r", ""))
        i += 1

    return transformed_tokens, var_dict, func_dict, lit_dict, struct_dict, class_dict


# Example usage
if __name__ == "__main__":
    code_text = ("int x = 100;\n int square(int x)\n { printf(\"%d\\n\", 5);\n"
                 " return ((x) * (x));\n }\n int main()\n { int r = square(5);\n"
                 " printf(\"%d\\n\", (r));\n return 0; \n}\n")

    transformed_tokens, var_dict, func_dict, lit_dict = obfuscate_and_tokenize(code_text)

    print("---- Transformed Code ----")
    print(transformed_tokens)
    print("\n---- Variable Dict ----")
    print(var_dict)
    print("\n---- Function Dict ----")
    print(func_dict)
    print("\n---- Literal Dict ----")
    print(lit_dict)
