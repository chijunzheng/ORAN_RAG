# src/data_processing/yang_preprocessor.py

import os
import re
import logging

def extract_vendor_package(file_path: str) -> str:
    match = re.search(r'[0-9]{2}[A-Za-z]', file_path)  # e.g. "24A" or "24B"
    if match:
        return match.group(0)
    return "unknown-package"

def extract_yang_version(yang_text: str) -> str:
    """
    Attempts to parse the top-most revision statement or fallback to 'unknown'.
    But keep in mind some vendor YANGs might not update revision inside the file.
    """
    match = re.search(r'revision\s+"([^"]+)"\s*{', yang_text)
    if match:
        return match.group(1)
    return "unknown"

def extract_module_name(yang_text: str) -> str:
    match = re.search(r'module\s+([\w\-]+)\s*\{', yang_text)
    if match:
        return match.group(1)
    return "unknown-module"

def load_raw_yang_text(yang_path: str) -> str:
    with open(yang_path, "r", encoding="utf-8") as f:
        return f.read()


def convert_yang_to_markdown(
    yang_path: str,
    pyang_search_paths=None,
    ignore_pyang_errors=False
) -> str:
    """
    Naive approach: read the .yang file, detect top-level statements
    (module, grouping, container, rpc, notification, leaf, etc.),
    insert headings (#, ##, ###) so that 'yang_chunker.py' can chunk them.

    Ignores Pyang entirely.
    """

    if not os.path.isfile(yang_path):
        raise FileNotFoundError(f"YANG file not found: {yang_path}")

    # 1) Load raw text
    raw_text = load_raw_yang_text(yang_path)
    lines = raw_text.splitlines()

    # 2) We'll build a "markdown-like" string. 
    #    E.g. "# module: <name>", "## grouping: <name>", "### leaf: <name>" etc.
    md_lines = []
    for line in lines:
        stripped = line.strip()

        # We can do some regex to detect:
        #  - module <NAME> {
        #  - grouping <NAME> {
        #  - container <NAME> {
        #  - rpc <NAME> {
        #  - notification <NAME> {
        #  - leaf <NAME> {
        # etc.
        # Then convert them into heading lines.

        # Basic example: detect top-level statements with a pattern like:
        #   ^(\w+)\s+([\w\-]+)\s*{
        # That captures the keyword and the name, e.g. "grouping my-group {"
        statement_match = re.match(r'^(\w+)\s+([\w\-\d]+)\s*\{', stripped)
        if statement_match:
            keyword = statement_match.group(1)
            name = statement_match.group(2)

            # For module:
            if keyword == "module":
                md_lines.append(f"# module: {name}")
                continue
            # For grouping, container, rpc, notification:
            elif keyword in ["grouping", "container", "rpc", "notification", "augment", "submodule"]:
                md_lines.append(f"## {keyword}: {name}")
                continue
            # For leaf or leaf-list or typedef:
            elif keyword in ["leaf", "leaf-list", "typedef", "list"]:
                md_lines.append(f"### {keyword}: {name}")
                continue

        # If line doesn't match, keep as-is
        md_lines.append(line)

    # Join them back up
    markdown_text = "\n".join(md_lines)

    if not markdown_text.strip():
        # If the file is basically empty or
        # we found no recognized statements, we can proceed or skip
        logging.warning(f"No recognized statements for {yang_path}.")
        if ignore_pyang_errors:
            return ""
        else:
            # or just return the entire file as fallback
            return raw_text

    return markdown_text