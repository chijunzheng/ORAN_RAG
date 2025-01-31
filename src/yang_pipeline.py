# src/yang_pipeline.py

import os
import logging
from typing import List, Dict
from src.data_processing.yang_preprocessor import (
    extract_vendor_package,
    extract_yang_version,
    extract_module_name,
    load_raw_yang_text,
    convert_yang_to_markdown
)
from src.data_processing.yang_chunker import YangChunker

def process_yang_dir(yang_dir: str) -> List[Dict]:
    """
    Goes through .yang files in the specified directory (and all subdirs),
    builds a list of chunk dicts with doc_type='yang_model'.

    We also build a 'pyang_lib_dirs' list that includes *every* subdirectory
    under 'yang_dir', so Pyang can locate all dependencies.
    """

    if not os.path.isdir(yang_dir):
        logging.warning(f"YANG directory not found: {yang_dir}")
        return []

    # 1) Collect all subdirectories as search paths
    pyang_lib_dirs = []
    for root, dirs, files in os.walk(yang_dir):
        pyang_lib_dirs.append(root)

    all_chunks = []
    chunker = YangChunker()

    # 2) Now walk again to process each .yang file
    for root, dirs, files in os.walk(yang_dir):
        for fname in files:
            if fname.endswith(".yang"):
                fpath = os.path.join(root, fname)
                logging.info(f"Processing YANG file: {fpath}")
                try:
                    # (A) parse vendor package from file path
                    vendor_pkg = extract_vendor_package(fpath)

                    # (B) read raw text for version & module
                    raw_text = load_raw_yang_text(fpath)
                    ver = extract_yang_version(raw_text)
                    mod_name = extract_module_name(raw_text)

                    # (C) Convert with Pyang → tree → markdown
                    #     pass the entire subdir set to let Pyang find all imports
                    markdown_text = convert_yang_to_markdown(
                        fpath,
                        pyang_search_paths=pyang_lib_dirs,
                        ignore_pyang_errors=False  # or True if you want to skip on errors
                    )

                    if not markdown_text.strip():
                        logging.warning(f"Skipping file {fname} because pyang returned empty text.")
                        continue

                    # (D) chunk
                    new_chunks = chunker.chunk(
                        text=markdown_text,
                        module_name=mod_name,
                        version=ver,
                        package_name=vendor_pkg,
                        filename=fname
                    )
                    all_chunks.extend(new_chunks)

                except Exception as e:
                    logging.error(f"Failed to process YANG file {fname}: {e}", exc_info=True)

    logging.info(f"Total YANG chunks from {yang_dir}: {len(all_chunks)}")
    return all_chunks