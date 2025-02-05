# src/yang_pipeline.py

import os
import logging
from typing import List, Dict
from src.data_processing.yang_preprocessor import (
    extract_vendor_package,
    extract_revision,
    extract_module_name,
    extract_namespace,
    extract_yang_version_keyword,
    extract_organization,
    extract_description_snippet,
    load_raw_yang_text,
    convert_yang_to_markdown
)
from src.data_processing.yang_chunker import YangChunker

def process_yang_dir(yang_dir: str) -> List[Dict]:
    """
    Processes all .yang files in the given directory recursively.
    For each file, it extracts metadata, converts the content to markdown,
    and then splits the content into chunks that include all relevant metadata (including chunk_index).
    
    Returns:
        List[Dict]: List of chunk dictionaries with full metadata.
    """
    if not os.path.isdir(yang_dir):
        logging.warning(f"YANG directory not found: {yang_dir}")
        return []
    
    # Build a list of directories for the pyang search path.
    pyang_lib_dirs = []
    for root, dirs, files in os.walk(yang_dir):
        pyang_lib_dirs.append(root)
    
    all_chunks = []
    dynamic_chunker = YangChunker(target_token_count=1536, chunk_overlap=256, max_token_limit=2048)
    
    for root, dirs, files in os.walk(yang_dir):
        for fname in files:
            if fname.endswith(".yang"):
                fpath = os.path.join(root, fname)
                logging.info(f"Processing YANG file: {fpath}")
                try:
                    vendor_pkg = extract_vendor_package(fpath)
                    raw_text = load_raw_yang_text(fpath)
                    revision = extract_revision(raw_text)
                    mod_name = extract_module_name(raw_text)
                    namespace = extract_namespace(raw_text)
                    yang_ver = extract_yang_version_keyword(raw_text)
                    organization = extract_organization(raw_text)
                    description_snippet = extract_description_snippet(raw_text)
                    
                    markdown_text = convert_yang_to_markdown(
                        fpath,
                        pyang_search_paths=pyang_lib_dirs,
                        ignore_pyang_errors=True
                    )
                    if not markdown_text.strip():
                        logging.warning(f"Skipping file {fname} because conversion returned empty text.")
                        continue
                    
                    new_chunks = dynamic_chunker.chunk(
                        text=markdown_text,
                        module_name=mod_name,
                        revision=revision,
                        package_name=vendor_pkg,
                        file_name=fname,
                        namespace=namespace,
                        yang_version=yang_ver,
                        organization=organization,
                        description=description_snippet
                    )
                    all_chunks.extend(new_chunks)
                except Exception as e:
                    logging.error(f"Failed to process YANG file {fname}: {e}", exc_info=True)
    logging.info(f"Total YANG chunks from {yang_dir}: {len(all_chunks)}")
    return all_chunks