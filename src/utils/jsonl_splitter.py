# src/utils/jsonl_splitter.py

import os
import logging
from typing import List

def split_jsonl_file(
    input_file: str,
    output_dir: str,
    max_size: int = 1 * 1024 * 1024
) -> List[str]:
    """
    Splits a JSONL file into multiple files, each under max_size bytes.
    
    Args:
        input_file (str): Path to the input JSONL file (e.g., chunks.json or embeddings.json).
        output_dir (str): Local directory to store the split output files.
        max_size (int): Maximum file size in bytes. Defaults to 1 MB.

    Returns:
        List[str]: List of file paths to the split JSON files.
    """
    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        return []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    split_file_paths = []
    file_count = 1
    current_size = 0
    
    # Start the first split file
    current_file_path = os.path.join(output_dir, f"part_{file_count}.json")
    current_file = open(current_file_path, 'w', encoding='utf-8')
    split_file_paths.append(current_file_path)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_size = len(line.encode('utf-8'))
            # If adding this line exceeds max_size, move to the next file
            if current_size + line_size > max_size:
                current_file.close()
                file_count += 1
                current_file_path = os.path.join(output_dir, f"part_{file_count}.json")
                current_file = open(current_file_path, 'w', encoding='utf-8')
                split_file_paths.append(current_file_path)
                current_size = 0
            
            current_file.write(line)
            current_size += line_size

    current_file.close()
    logging.info(f"Split {input_file} into {len(split_file_paths)} files in '{output_dir}'.")
    return split_file_paths