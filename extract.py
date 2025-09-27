"""
This script extracts C / C++ codes from raw jsons.
"""
import os
import json

def clear_output_files():
    """
    Clears the content of all output files (/data)
    """

    files = [
        'data/paired_c.txt',
        'data/paired_cpp.txt',
        'data/unpaired_c.txt',
        'data/unpaired_cpp.txt'
    ]

    for file_path in files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Open and immediately close to clear the file
        open(file_path, 'w', encoding='utf-8').close()
    
    print("Output files cleared.")

def process_json_files(file_paths):
    """
    Extracts C / C++ code from JSON, removing duplicates
    """
    lines_processed = 0
    lines_skipped = 0
    entries_with_c_and_cpp = 0
    entries_with_only_c = 0
    entries_with_only_cpp = 0

    seen_c = set()
    seen_cpp = set()

    try:
        with open('data/paired_c.txt', 'a', encoding='utf-8') as paired_c_file, \
             open('data/paired_cpp.txt', 'a', encoding='utf-8') as paired_cpp_file, \
             open('data/unpaired_c.txt', 'a', encoding='utf-8') as unpaired_c_file, \
             open('data/unpaired_cpp.txt', 'a', encoding='utf-8') as unpaired_cpp_file:
            
            for path in file_paths:
                with open(path, 'r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        line = line.strip()
                        if not line:
                            continue 

                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"Skipping invalid JSON on line {line_num}: {e}")
                            lines_skipped += 1
                            continue

                        has_c = 'C' in entry
                        has_cpp = 'C++' in entry

                        if has_c and has_cpp:
                            c_code = str(entry['C']).replace('\n', '\\n')
                            cpp_code = str(entry['C++']).replace('\n', '\\n')

                            if c_code not in seen_c and cpp_code not in seen_cpp:
                                paired_c_file.write(c_code + '\n')
                                paired_cpp_file.write(cpp_code + '\n')
                                seen_c.add(c_code)
                                seen_cpp.add(cpp_code)
                                entries_with_c_and_cpp += 1

                        elif has_c:
                            c_code = str(entry['C']).replace('\n', '\\n')
                            if c_code not in seen_c:
                                unpaired_c_file.write(c_code + '\n')
                                seen_c.add(c_code)
                                entries_with_only_c += 1

                        elif has_cpp:
                            cpp_code = str(entry['C++']).replace('\n', '\\n')
                            if cpp_code not in seen_cpp:
                                unpaired_cpp_file.write(cpp_code + '\n')
                                seen_cpp.add(cpp_code)
                                entries_with_only_cpp += 1

                        lines_processed += 1

        print(f"Processing completed.")
        print(f"Total lines processed: {lines_processed}")
        print(f"Total lines skipped (invalid JSON): {lines_skipped}")
        print(f"Unique entries with both 'C' and 'C++': {entries_with_c_and_cpp}")
        print(f"Unique entries with only 'C': {entries_with_only_c}")
        print(f"Unique entries with only 'C++': {entries_with_only_cpp}")

    except Exception as e:
        print(f"Error processing files: {e}")

data = ["raw_data/multilingual_train.json", "raw_data/multilingual_test.json", "raw_data/multilingual_valid.json", "raw_data/LLMTrans.json", "raw_data/niche_test.json", "raw_data/niche_train.json", "raw_data/niche_valid.json"]
process_json_files(data)

# clear_output_files()