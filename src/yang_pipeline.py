# src/yang_pipeline.py

import os
import logging
from typing import List, Dict

# Import the ORAN-specific preprocessor functions.
from src.data_processing.oran_yang_preprocessor import (
    extract_oran_metadata,
    convert_oran_yang_to_markdown
)
# Import vendor-specific functions as fallback.
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

def process_yang_dir(yang_dir: str) -> List[Dict]:
    """
    Processes all .yang files in the given directory recursively.
    For each file, extracts metadata, converts the content to markdown,
    and splits the content into chunks including metadata.
    
    For files whose full path indicates an ORAN package (i.e. the file path contains "O-RAN"),
    this function uses the ORAN-specific preprocessor to extract metadata including
    'oran_version' and 'workgroup'. Otherwise, it falls back to vendor-specific extraction.
    
    Returns:
        List[Dict]: List of chunk dictionaries with full metadata.
    """
    if not os.path.isdir(yang_dir):
        logging.warning(f"YANG directory not found: {yang_dir}")
        return []
    
    all_chunks = []
    # Initialize the YangChunker.
    from src.data_processing.yang_chunker import YangChunker
    dynamic_chunker = YangChunker(target_token_count=1536, chunk_overlap=256, max_token_limit=2048)
    
    for root, dirs, files in os.walk(yang_dir):
        for fname in files:
            if fname.endswith(".yang"):
                fpath = os.path.join(root, fname)
                logging.info(f"Processing YANG file: {fpath}")
                try:
                    # Check if the full file path indicates an ORAN package.
                    if "O-RAN" in fpath.upper():
                        # Use ORAN-specific extraction.
                        metadata = extract_oran_metadata(fpath)
                        markdown_text = convert_oran_yang_to_markdown(
                            fpath,
                            pyang_search_paths=[yang_dir],
                            ignore_pyang_errors=True
                        )
                        is_oran = True
                    else:
                        # Use vendor-specific extraction.
                        yang_text = load_raw_yang_text(fpath)
                        metadata = {
                            "module_name": extract_module_name(yang_text),
                            "revision": extract_revision(yang_text),
                            "namespace": extract_namespace(yang_text),
                            "yang_version": extract_yang_version_keyword(yang_text),
                            "organization": extract_organization(yang_text),
                            "description": extract_description_snippet(yang_text),
                            "vendor_package": extract_vendor_package(fpath)
                        }
                        markdown_text = convert_yang_to_markdown(
                            fpath,
                            pyang_search_paths=[yang_dir],
                            ignore_pyang_errors=True
                        )
                        is_oran = False
                    
                    if not markdown_text.strip():
                        logging.warning(f"Skipping file {fname} because conversion returned empty text.")
                        continue
                    
                    # For ORAN files, the package_name for the chunker will be the ORAN version.
                    package_name = metadata.get("oran_version") if is_oran else metadata.get("vendor_package", "unknown")
                    workgroup = metadata.get("workgroup") if is_oran else None
                    
                    new_chunks = dynamic_chunker.chunk(
                        text=markdown_text,
                        module_name=metadata.get("module_name", "unknown"),
                        revision=metadata.get("revision", "unknown"),
                        package_name=package_name,
                        file_name=fname,
                        namespace=metadata.get("namespace", "unknown"),
                        yang_version=metadata.get("yang_version", "unknown"),
                        organization=metadata.get("organization", "unknown"),
                        description=metadata.get("description", "No description available."),
                        is_oran=is_oran,
                        workgroup=workgroup
                    )
                    all_chunks.extend(new_chunks)
                except Exception as e:
                    logging.error(f"Failed to process YANG file {fname}: {e}", exc_info=True)
    logging.info(f"Total YANG chunks from {yang_dir}: {len(all_chunks)}")
    return all_chunks