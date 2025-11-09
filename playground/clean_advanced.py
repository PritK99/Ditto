import re
import subprocess
import sys
import os
import tempfile

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


def preprocess_code_only(code_text):
    """
    run gcc -E -P on code and return output

    args:
        code_text: only code part of the src file

    returns: 
        result: file with comments and macros resolved
    """
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
    filename = "simple.c"

    includes, code = extract_includes_and_code(filename)
    processed = preprocess_code_only(code)

    with open(f"cleaned_{filename}", "w") as f:
        f.write(f"{includes}\n{processed}")    # Combining includes and resolved code part back

    print(f"Final output written to {filename}")

if __name__ == "__main__":
    main()
