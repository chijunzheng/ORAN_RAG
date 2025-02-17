# src/data_processing/yang_chunker.py

import uuid
import logging
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import tiktoken
except ImportError:
    tiktoken = None

def count_tokens(text: str) -> int:
    """
    Counts tokens using tiktoken if available; otherwise falls back to splitting on whitespace.
    """
    if tiktoken:
        try:
            encoding = tiktoken.encoding_for_model("text-embedding-005")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    else:
        return len(text.split())

class YangChunker:
    def __init__(self, target_token_count: int = 1536, chunk_overlap: int = 256, max_token_limit: int = 2048):
        """
        Initializes the YangChunker.
        
        Args:
            target_token_count (int): Desired token count per chunk (default 1536).
            chunk_overlap (int): Number of overlapping tokens between chunks (default 256).
            max_token_limit (int): Maximum allowed tokens per chunk (default 2048).
        """
        self.target_token_count = target_token_count
        self.chunk_overlap = chunk_overlap
        self.max_token_limit = max_token_limit
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_token_count,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "],
            length_function=lambda text: len(text.split())
        )

    def _split_to_fit(self, text: str, header_tokens: int) -> List[str]:
        """
        Recursively splits the text so that when combined with header_tokens,
        the total token count does not exceed max_token_limit.
        """
        allowed = self.max_token_limit - header_tokens
        current_tokens = count_tokens(text)
        if current_tokens <= allowed:
            return [text]
        mid_index = len(text) // 2
        split_index = text.rfind(" ", 0, mid_index)
        if split_index == -1:
            split_index = mid_index
        first_part = text[:split_index].strip()
        second_part = text[split_index:].strip()
        return self._split_to_fit(first_part, header_tokens) + self._split_to_fit(second_part, header_tokens)

    def chunk(
        self,
        text: str,
        module_name: str,
        revision: str,
        package_name: str,
        file_name: str,
        namespace: str,
        yang_version: str,
        organization: str,
        description: str,
        is_oran: bool = False,
        workgroup: str = None
    ) -> List[Dict]:
        """
        Splits the provided markdown-converted YANG text into chunks.
        Each chunk is prefixed with a metadata header that contains information.
        For ORAN files, the header will include the ORAN version and workgroup;
        for vendor-specific files, it will include the vendor package.
        
        Returns:
            List[Dict]: List of chunk dictionaries with a unique UUID and full metadata.
        """
        # Build the base metadata header based on whether this is an ORAN file.
        if is_oran:
            base_metadata = (
                f"File: {file_name}; Module: {module_name}; Revision Data: {revision}; "
                f"ORAN Version: {package_name}; Workgroup: {workgroup}; Namespace: {namespace}; "
                f"Yang Version: {yang_version}; Organization: {organization}; Description: {description}"
            )
        else:
            base_metadata = (
                f"File: {file_name}; Module: {module_name}; Revision Data: {revision}; "
                f"Vendor Package: {package_name}; Namespace: {namespace}; Yang Version: {yang_version}; "
                f"Organization: {organization}; Description: {description}"
            )
        base_header = base_metadata + "\n\n"
        base_header_tokens = count_tokens(base_header)

        # Split the text into candidate chunks using the splitter.
        candidate_chunks = self.splitter.split_text(text)
        logging.info(f"Candidate chunks count before additional splitting: {len(candidate_chunks)}")
        
        # Process candidate chunks and further split if necessary.
        final_chunks = []
        for candidate in candidate_chunks:
            if not candidate.strip():
                continue
            pieces = self._split_to_fit(candidate, base_header_tokens)
            final_chunks.extend(pieces)
        logging.info(f"Total candidate chunks after ensuring max token limit: {len(final_chunks)}")

        # Assemble the final chunk dictionaries, including the chunk index.
        chunks = []
        for idx, chunk_text in enumerate(final_chunks, start=1):
            if not chunk_text.strip():
                continue
            header_with_index = base_header + f"Chunk Index: {idx}\n\n"
            full_content = header_with_index + chunk_text
            total_tokens = count_tokens(full_content)
            if total_tokens > self.max_token_limit:
                logging.error(f"Chunk for file {file_name} exceeds max token limit: {total_tokens} tokens.")
            chunk_id = str(uuid.uuid4())
            # Build metadata dict; include ORAN-specific keys if applicable.
            metadata_dict = {
                "doc_type": "yang_model",
                "module": module_name,
                "revision": revision,
                "file_name": file_name,
                "namespace": namespace,
                "yang_version": yang_version,
                "organization": organization,
                "description": description,
                "chunk_index": idx
            }
            if is_oran:
                metadata_dict["oran_version"] = package_name  # Here, package_name represents ORAN version.
                metadata_dict["workgroup"] = workgroup
            else:
                metadata_dict["vendor_package"] = package_name

            chunks.append({
                "id": chunk_id,
                "content": full_content,
                "metadata": metadata_dict
            })
        logging.info(f"Created {len(chunks)} YANG chunks for file: {file_name}")
        return chunks