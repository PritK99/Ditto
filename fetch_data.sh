#!/bin/bash

DATA_DIR="data"

if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
else
    echo "Data directory already exists: $DATA_DIR"
fi

# Please note that these URLs might change
# In case the script doesn't work, please download using link given in README
C_URL="https://iiithydresearch-my.sharepoint.com/:u:/g/personal/prit_kanadiya_research_iiit_ac_in/IQDDPHjeas1rSId9ZyR3ShBTAdohKu_DJatn39Qo0iiYU2E?download=1"
CPP_URL="https://iiithydresearch-my.sharepoint.com/:u:/g/personal/prit_kanadiya_research_iiit_ac_in/IQCRmkRplSrqRpRKDIglsbXAAXuSX6MouOf2K4sk3lfgH9k?download=1"
VOCAB_URL="https://iiithydresearch-my.sharepoint.com/:t:/g/personal/prit_kanadiya_research_iiit_ac_in/IQDA9yGC5y6iQ4CItz97UWUsATECSa6So2ao1cGMb6_PLnE?e=U8B7XK"

FILES=("c_tokens_with_lca_dist.parquet" "cpp_tokens_with_lca_dist.parquet" "final_vocab_combined.txt")
URLS=("$C_URL" "$CPP_URL" "$VOCAB_URL")

for i in "${!FILES[@]}"; do
    FILE_PATH="$DATA_DIR/${FILES[$i]}"
    URL="${URLS[$i]}"

    if [ -f "$FILE_PATH" ]; then
        echo "File already exists: $FILE_PATH"
    else
        echo "Downloading ${FILES[$i]}..."
        if command -v wget > /dev/null 2>&1; then
            wget -O "$FILE_PATH" "$URL"
        elif command -v curl > /dev/null 2>&1; then
            curl -L "$URL" -o "$FILE_PATH"
        else
            echo "Error: wget or curl not found. Please install one."
            exit 1
        fi
    fi

    if file "$FILE_PATH" | grep -q "HTML"; then
        echo "ERROR: ${FILES[$i]} appears to be an HTML page, not the actual data file."
        echo "Please check the SharePoint link or download manually."
        exit 1
    else
        echo "${FILES[$i]} downloaded and verified successfully."
    fi
done

echo "All files are present and verified in $DATA_DIR."
