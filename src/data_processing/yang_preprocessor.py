# src/data_processing/yang_preprocessor.py

import os
import re
import logging

def extract_vendor_package(file_path: str) -> str:
    """
    Extracts vendor package information from the file path.
    E.g., if the file path contains "24A" or "24B", return that.
    """
    match = re.search(r'[0-9]{2}[A-Za-z]', file_path)
    if match:
        return match.group(0).upper()
    return "unknown-package"

def extract_revision(yang_text: str) -> str:
    """
    Extracts the revision date from the YANG file.
    Matches both quoted and unquoted revision dates in the format YYYY-MM-DD,
    followed by a "{" or ";".
    """
    match = re.search(r'revision\s+["]?(\d{4}-\d{2}-\d{2})["]?\s*[\{;]', yang_text)
    if match:
        revision_date = match.group(1)
        logging.debug(f"Extracted revision date: {revision_date}")
        return revision_date
    logging.debug("No revision date found; returning 'unknown'")
    return "unknown"

def extract_module_name(yang_text: str) -> str:
    """
    Extracts the module name from the YANG file.
    """
    match = re.search(r'module\s+([\w\-]+)\s*\{', yang_text)
    if match:
        return match.group(1)
    return "unknown-module"

def extract_namespace(yang_text: str) -> str:
    """
    Extracts the namespace from the YANG file.
    """
    match = re.search(r'namespace\s+"([^"]+)"', yang_text)
    if match:
        return match.group(1)
    return "unknown-namespace"

def extract_yang_version_keyword(yang_text: str) -> str:
    """
    Extracts the yang-version (e.g., "1.1") from the YANG file.
    """
    match = re.search(r'yang-version\s+([0-9\.]+)', yang_text)
    if match:
        return match.group(1)
    return "unknown-yang-version"

def extract_organization(yang_text: str) -> str:
    """
    Extracts the organization information from the YANG file.
    """
    match = re.search(r'organization\s+"([^"]+)"', yang_text)
    if match:
        return match.group(1)
    return "unknown-organization"

def extract_description_snippet(yang_text: str, max_length: int = 200) -> str:
    """
    Extracts a short snippet from the module-level description.
    """
    match = re.search(r'description\s+"([^"]+)"', yang_text, re.DOTALL)
    if match:
        desc = match.group(1).strip().replace("\n", " ")
        if len(desc) > max_length:
            return desc[:max_length] + "..."
        else:
            return desc
    return "No description available."

def load_raw_yang_text(yang_path: str) -> str:
    """
    Loads the raw content of the YANG file.
    """
    with open(yang_path, "r", encoding="utf-8") as f:
        return f.read()

def convert_yang_to_markdown(yang_path: str, pyang_search_paths=None, ignore_pyang_errors=False) -> str:
    """
    Naively converts a YANG file to a markdown-like representation.
    This transformation adds markdown headings for top-level YANG statements.
    """
    if not os.path.isfile(yang_path):
        raise FileNotFoundError(f"YANG file not found: {yang_path}")
    raw_text = load_raw_yang_text(yang_path)
    lines = raw_text.splitlines()
    md_lines = []
    for line in lines:
        stripped = line.strip()
        statement_match = re.match(r'^(\w+)\s+([\w\-]+)\s*\{', stripped)
        if statement_match:
            keyword = statement_match.group(1)
            name = statement_match.group(2)
            if keyword == "module":
                md_lines.append(f"# module: {name}")
                continue
            elif keyword in ["grouping", "container", "rpc", "notification", "augment", "submodule"]:
                md_lines.append(f"## {keyword}: {name}")
                continue
            elif keyword in ["leaf", "leaf-list", "typedef", "list"]:
                md_lines.append(f"### {keyword}: {name}")
                continue
        md_lines.append(line)
    markdown_text = "\n".join(md_lines)
    if not markdown_text.strip():
        logging.warning(f"No recognized statements for {yang_path}.")
        return raw_text if not ignore_pyang_errors else ""
    return markdown_text