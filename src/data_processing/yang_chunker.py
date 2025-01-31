# src/data_processing/yang_chunker.py
import uuid
import logging
from typing import List, Dict

class YangChunker:
    def chunk(self, text: str, module_name: str, version: str, package_name: str, filename: str) -> List[Dict]:
        """
        Splits the YANG markdown into chunk dicts. E.g. around lines that start with
        "## grouping:", "## container:", or "### leaf:" etc.
        We'll store doc_type="yang_model" + the vendor_package in metadata.
        """
        lines = text.splitlines()
        chunks = []
        current_chunk_lines = []
        current_path = "/" + module_name
        path_stack = [module_name]

        def flush_chunk():
            if not current_chunk_lines:
                return
            chunk_text = "\n".join(current_chunk_lines)

            chunk_id = str(uuid.uuid4())

            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    "doc_type": "yang_model",
                    "module": module_name,
                    "version": version,           # from the file (or 'unknown')
                    "vendor_package": package_name,  # e.g. "24A"
                    "yang_path": current_path,
                    "source": filename
                }
            })
            current_chunk_lines.clear()

        for line in lines:
            if line.startswith("## grouping:") or line.startswith("## container:") or "## rpc" in line or "## notification" in line:
                flush_chunk()
                # parse the block name from the line
                # e.g. "## grouping: slot-group"
                # path_stack = [module_name, block_name]
                parts = line.split(":", 1)
                if len(parts) == 2:
                    block_name = parts[1].strip()
                    path_stack = [module_name, block_name]
                    current_path = "/" + "/".join(path_stack)

            elif line.startswith("### leaf:"):
                flush_chunk()
                leaf_name = line.split(":", 1)[-1].strip()
                path_stack.append(leaf_name)
                current_path = "/" + "/".join(path_stack)

            current_chunk_lines.append(line)

        # Flush final
        flush_chunk()
        logging.info(f"Created {len(chunks)} YANG chunks for file: {filename}")
        return chunks
    


    