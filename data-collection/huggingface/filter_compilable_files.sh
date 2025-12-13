#!/bin/bash

# --- Configuration ---
SUCCESS_DIR="./compiled_success_executables"
FAILURE_DIR="./compiled_failure_source"
SOURCE_ROOT="."

mkdir -p "$SUCCESS_DIR"
mkdir -p "$FAILURE_DIR"

echo "Starting compilation. Successful executables will be moved to $SUCCESS_DIR"
echo "Files that fail (e.g., due to bits/stdc++.h) will be moved to $FAILURE_DIR for review."
echo "Searching recursively for .cpp files inside the '$SOURCE_ROOT' directory."
echo "------------------------------------------------------------------"

find "$SOURCE_ROOT" -name "*.cpp" | while read FILE_PATH; do
    EXEC_NAME=$(echo "$FILE_PATH" | sed 's/^\.\///' | tr '/' '_')
    EXEC_NAME="${EXEC_NAME%.cpp}"
    
    executable="$SUCCESS_DIR/$EXEC_NAME"
    
    TEMP_EXEC_PATH="./temp_$EXEC_NAME"

    g++ -std=c++17 -Wall "$FILE_PATH" -o "$TEMP_EXEC_PATH"
    
    if [ $? -eq 0 ]; then
        mv "$FILE_PATH" "$SUCCESS_DIR"
        rm -f "$TEMP_EXEC_PATH" 
        echo "[SUCCESS] $FILE_PATH compiled. Executable moved to $executable"
    else
        echo "[FAILURE] Error compiling $FILE_PATH. Source file moved to $FAILURE_DIR/ for inspection."
        FAILURE_FILE="$FAILURE_DIR/$EXEC_NAME.cpp"
        mv "$FILE_PATH" "$FAILURE_FILE"
        rm -f "$TEMP_EXEC_PATH" 
    fi
done

echo "------------------------------------------------------------------"
echo "Compilation process finished."