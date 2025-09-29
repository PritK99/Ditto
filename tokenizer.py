import os
from clang import cindex

cindex.Config.set_library_file("/usr/lib/llvm-18/lib/libclang.so")

def process_file(source_file, output_csv, is_cpp: bool):
    index = cindex.Index.create()
    
    if (is_cpp):
        args = ['-x', 'c++', '-std=c++17']
    
    else:
        args = ['-x', 'c', '-std=c11']
        
    tu = index.parse(source_file, args=args)
    tokens = tu.get_tokens(extent=tu.cursor.extent)

    for token in tokens:
        print(token.spelling)

    # try:
    #     tu = index.parse(source_file, args=args)
    # except cindex.TranslationUnitLoadError as e:
    #     print(f"Failed to parse {source_file}: {e}")
    #     return
    
    # tokens = tu.get_tokens(extent=tu.cursor.extent)
    # token_list = [token.spelling for token in tokens]
    
    # with open(output_csv, 'w') as f:
    #     f.write(','.join(token_list))

dir_name = "final_data"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

files = ["unpaired_cpp.csv"]

# Ensure CSV files exist (empty if new)
for file_name in files:
    file_path = os.path.join(dir_name, file_name)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass

datasets = ["transcodeocean"]

for dataset in datasets:
    base_path = os.path.join("datasets", dataset, "data")
    
    for file_name in files:
        src_base = file_name.replace(".csv", ".txt")
        source_file = os.path.join(base_path, src_base)
        
        if not os.path.isfile(source_file):
            print(f"Source file does not exist: {source_file}")
            continue
        
        output_csv = os.path.join(dir_name, file_name)
        
        if "unpaired" in file_name and "cpp" in file_name:
            print(f"Processing C++ unpaired file: {source_file}")
            process_file(source_file, output_csv, is_cpp=True)
        elif "unpaired" in file_name and "c" in file_name:
            print(f"Processing C unpaired file: {source_file}")
            process_file(source_file, output_csv, is_cpp=False)
        else:
            print(f"Skipping file {file_name}")
