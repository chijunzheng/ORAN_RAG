import os
import re
import logging

# Import common functions from your original YANG preprocessor.
from src.data_processing.yang_preprocessor import (
    extract_revision,
    extract_module_name,
    extract_namespace,
    extract_yang_version_keyword,
    extract_organization,
    extract_description_snippet,
    load_raw_yang_text,
    convert_yang_to_markdown
)

def extract_oran_version(file_path: str) -> str:
    """
    Extracts the ORAN version from the appropriate directory in the file path.
    For example, given a file path like:
      .../O-RAN.WG4.MP-YANGs-v06.00/Common Models/Operations/o-ran-trace.yang
    it extracts "v06.00" from the directory "O-RAN.WG4.MP-YANGs-v06.00".
    """
    # Normalize the path and split into parts.
    parts = os.path.normpath(file_path).split(os.sep)
    candidate = None
    # Look for the first directory that contains "O-RAN" (case-insensitive).
    for part in parts:
        if "O-RAN" in part.upper():
            candidate = part
            break
    if candidate:
        match = re.search(r'-v(\d+\.\d+)', candidate)
        if match:
            return "v" + match.group(1)
    return "unknown-ORAN-version"

def extract_workgroup(file_path: str) -> str:
    """
    Extracts the workgroup (WGx) from the appropriate directory in the file path.
    For example, given a file path like:
      .../O-RAN.WG4.MP-YANGs-v06.00/Common Models/Operations/o-ran-trace.yang
    it extracts "WG4" from the directory "O-RAN.WG4.MP-YANGs-v06.00".
    """
    parts = os.path.normpath(file_path).split(os.sep)
    candidate = None
    for part in parts:
        if "O-RAN" in part.upper():
            candidate = part
            break
    if candidate:
        match = re.search(r'(WG\d+)', candidate, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "unknown-workgroup"

def extract_oran_metadata(yang_path: str) -> dict:
    """
    Extracts metadata from an ORAN-specific YANG file.
    
    Uses common extraction methods and adds ORAN-specific fields:
      - oran_version: extracted from the appropriate directory in the file path.
      - workgroup: extracted from the appropriate directory in the file path.
    
    Returns a dictionary with keys:
      - module_name
      - revision
      - namespace
      - yang_version
      - organization
      - description
      - oran_version
      - workgroup
    """
    yang_text = load_raw_yang_text(yang_path)
    metadata = {
        "module_name": extract_module_name(yang_text),
        "revision": extract_revision(yang_text),
        "namespace": extract_namespace(yang_text),
        "yang_version": extract_yang_version_keyword(yang_text),
        "organization": extract_organization(yang_text),
        "description": extract_description_snippet(yang_text),
        "oran_version": extract_oran_version(yang_path),
        "workgroup": extract_workgroup(yang_path)
    }
    logging.debug(f"Extracted ORAN metadata for {yang_path}: {metadata}")
    return metadata

def convert_oran_yang_to_markdown(yang_path: str, pyang_search_paths=None, ignore_pyang_errors=False) -> str:
    """
    Converts an ORAN-specific YANG file to a markdown-like representation.
    
    This function reuses the standard conversion logic.
    """
    return convert_yang_to_markdown(yang_path, pyang_search_paths, ignore_pyang_errors)