#!/bin/bash

# Directory to store data
DATA_DIR="data"

# Create data directory if it does not exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
else
    echo "Data directory already exists: $DATA_DIR"
fi

# Base URL for download training data
BASE_URL="https://iiithydresearch-my.sharepoint.com/:f:/g/personal/prit_kanadiya_research_iiit_ac_in/IgAUZu2iso0HRqS21t70pEfnAdmpkvT8_uOQZ8vClWAwDb0?e=Vd6puv"

# For training, we require these 3 datasets
FILES=(
    "c_tokens_with_lca_dist.parquet"
    "cpp_tokens_with_lca_dist.parquet"
    "final_vocab_combined.txt"
)

# Loop over files and download if missing
for FILE in "${FILES[@]}"; do
    FILE_PATH="$DATA_DIR/$FILE"
    if [ -f "$FILE_PATH" ]; then
        echo "File already exists: $FILE_PATH"
    else
        echo "Downloading $FILE..."
        # Use wget or curl depending on availability
        if command -v wget > /dev/null 2>&1; then
            wget -O "$FILE_PATH" "$BASE_URL/$FILE"
        elif command -v curl > /dev/null 2>&1; then
            curl -L "$BASE_URL/$FILE" -o "$FILE_PATH"
        else
            echo "Error: wget or curl not found. Please install one to download files."
            exit 1
        fi
    fi
done

echo "All files are present in $DATA_DIR"
