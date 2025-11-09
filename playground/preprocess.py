import re
import subprocess
import sys
import os
import tempfile
from tokenize import obfuscate_and_tokenize

def extract_includes_and_code(filename):
    """
    separates includes and rest of code from the src file

    args:
        filename: input c file

    returns: 
        includes: include statements 
        code: code part
    """
    with open(filename) as f:
        content = f.read()

    include_pattern = re.compile(r'^\s*#\s*include\s+[<"].+[>"].*$', re.MULTILINE)
    includes = "\n".join(include_pattern.findall(content)).strip()
    code = include_pattern.sub("", content).strip()
    return includes, code


def preprocess_code_only(code_text, is_cpp = False):
    """
    run gcc / g++ with flags -E -P on code and return output

    args:
        code_text: only code part of the src file

    returns: 
        result: file with comments and macros resolved
    """
    if (is_cpp):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as tmp:
            tmp.write(code_text)
            tmp_name = tmp.name
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tmp:
            tmp.write(code_text)
            tmp_name = tmp.name

    out_name = tmp_name + ".out"
    subprocess.run(["gcc", "-E", "-P", tmp_name, "-o", out_name], check=True)

    with open(out_name) as f:
        result = f.read().strip()

    os.remove(tmp_name)
    os.remove(out_name)
    return result

def main():
    filename = "simple.cpp"

    is_cpp = False
    if (".cpp" in filename):
        is_cpp = True

    includes, code = extract_includes_and_code(filename)
    processed = preprocess_code_only(code, is_cpp)

    transformed_tokens, var_dict, func_dict, lit_dict, struct_dict, class_dict = obfuscate_and_tokenize(processed, is_cpp)

    print("---- Transformed Code ----")
    print(transformed_tokens)
    print("\n---- Variable Dict ----")
    print(var_dict)
    print("\n---- Function Dict ----")
    print(func_dict)
    print("\n---- Literal Dict ----")
    print(lit_dict)
    print("\n---- Struct Dict ----")
    print(struct_dict)
    print("\n---- Class Dict ----")
    print(class_dict)

    with open(f"cleaned_{filename}", "w") as f:
        f.write(f"{includes}\n\n{processed}")    # Combining includes and resolved code part back

    print(f"Final output written to cleaned_{filename}")

if __name__ == "__main__":
    main()
