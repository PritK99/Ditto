import os
import re
import subprocess
import shutil
from pathlib import Path

file_path = "./simple.c"

src = Path(file_path)
base = src.stem
ext = src.suffix

# Creating a directory to store temporaries
os.makedirs("temporaries", exist_ok=True)

temp1 = f"temp1{ext}"
temp4 = f"temp4.txt"
temp5 = f"temp5.txt"
temp6 = f"temp6.txt"

# temp1 consists only the include statements from src file
include_re = re.compile(r'^\s*#\s*include\b')
with open(src) as f, open(temp1, "w") as g:
    for line in f:
        if include_re.match(line):
            g.write(line)

# temp4 contains AST for src
with open(temp4, "w") as g:
    subprocess.run(["clang", "-fsyntax-only", "-Xclang", "-ast-dump", src], stdout=g, stderr=subprocess.STDOUT, check=False)

# temp5 contains AST for temp1, which is includes only file
with open(temp5, "w") as g:
    subprocess.run(["clang", "-fsyntax-only", "-Xclang", "-ast-dump", temp1], stdout=g, stderr=subprocess.STDOUT, check=False)

# temp6 is set difference of src and temp1, i.e. it is AST for code without includes
with open(temp4) as f:
    lines4 = f.readlines()
with open(temp5) as f:
    lines5 = f.readlines()
with open(temp4) as f4, open(temp5) as f5, open(temp6, "w") as out:
    lines4 = f4.readlines()
    n_remove = len(f5.readlines())   
    out.writelines(lines4[n_remove:])

# This renames the ouput tree file as well as brings it to cwd
os.rename(temp6, f"{base}_{ext[1:]}_tree.txt")

# Cleaning the temporaries
for t in [temp1, temp4, temp5]:
    try:
        os.remove(t)
    except OSError:
        pass

# Clearing the temporary directory
if os.path.isdir("temporaries"):
    shutil.rmtree("temporaries")
