import os
import json

def clear_output_files():
    """
    Clears the content of all output files (clean_data)
    """

    files = [
        'clean_data/paired_c.txt',
        'clean_data/paired_cpp.txt',
        'clean_data/unpaired_c.txt',
        'clean_data/unpaired_cpp.txt'
    ]

    for file_path in files:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Open and immediately close to clear the file
        open(file_path, 'w', encoding='utf-8').close()
    
    print("Output files cleared.")


def process_json_files(file_paths):
    """
    Extracts C / C++ code from JSON
    
    Args:
        file_paths (list[str]): List of JSON files
    """
    lines_processed = 0
    lines_skipped = 0
    entries_with_c_and_cpp = 0
    entries_with_only_c = 0
    entries_with_only_cpp = 0

    try:
        with open('clean_data/paired_c.txt', 'a', encoding='utf-8') as paired_c_file, \
             open('clean_data/paired_cpp.txt', 'a', encoding='utf-8') as paired_cpp_file, \
             open('clean_data/unpaired_c.txt', 'a', encoding='utf-8') as unpaired_c_file, \
             open('clean_data/unpaired_cpp.txt', 'a', encoding='utf-8') as unpaired_cpp_file:
            
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
                            # Write paired entries to paired files
                            paired_c_file.write(str(entry['C']).replace('\n', '\\n') + '\n')
                            paired_cpp_file.write(str(entry['C++']).replace('\n', '\\n') + '\n')
                            entries_with_c_and_cpp += 1
                        elif has_c:
                            # Write C-only entries
                            unpaired_c_file.write(str(entry['C']).replace('\n', '\\n') + '\n')
                            entries_with_only_c += 1
                        elif has_cpp:
                            # Write C++-only entries
                            unpaired_cpp_file.write(str(entry['C++']).replace('\n', '\\n') + '\n')
                            entries_with_only_cpp += 1

                        lines_processed += 1

        print(f"Processing completed.")
        print(f"Total lines processed: {lines_processed}")
        print(f"Total lines skipped (invalid JSON): {lines_skipped}")
        print(f"Entries with both 'C' and 'C++': {entries_with_c_and_cpp}")
        print(f"Entries with only 'C': {entries_with_only_c}")
        print(f"Entries with only 'C++': {entries_with_only_cpp}")

    except Exception as e:
        print(f"Error processing files: {e}")


data = ["data/multilingual_train.json", "data/multilingual_test.json", "data/multilingual_valid.json", "data/LLMTrans.json", "data/niche_test.json", "data/niche_train.json", "data/niche_valid.json"]
process_json_files(data)

# clear_output_files()