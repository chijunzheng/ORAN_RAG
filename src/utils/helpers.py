# src/utils/helpers.py

import os

def ensure_directory(path: str):
    """
    Ensures that the specified directory exists.
    
    Args:
        path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)