# src/utils/rag_importer.py

import logging
from typing import List, Dict

class RagImporter:
    """
    A utility class to batch-import RAG files into the corpus,
    circumventing the 25 GCS URI limit per import operation.
    """

    @staticmethod
    def batch_import_rag_files(
        rag_corpus_mgr,
        rag_corpus_res: str,
        uri_list: List[str],
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        max_embedding_requests_per_min: int = 900
    ) -> Dict[str, int]:
        """
        Splits 'uri_list' into sub-lists of up to 25 URIs each and calls
        'import_files_from_gcs' repeatedly to import them into the RAG corpus.

        Args:
            rag_corpus_mgr: An instance of RagCorpusManager or equivalent, which
                has the 'import_files_from_gcs' method.
            rag_corpus_res (str): The resource name of the RAG corpus.
            uri_list (List[str]): All GCS URIs to import (possibly hundreds).
            chunk_size (int, optional): Number of tokens per chunk in the RAG import. Defaults to 512.
            chunk_overlap (int, optional): Overlap between chunks in tokens. Defaults to 100.
            max_embedding_requests_per_min (int, optional): QPM limit. Defaults to 900.

        Returns:
            Dict[str, int]: A dictionary with 'total_imported' and 'total_skipped' counts.
        """
        MAX_URIS = 25
        total_imported = 0
        total_skipped = 0

        # Batch the URIs into smaller sub-lists of size <= 25
        for start_idx in range(0, len(uri_list), MAX_URIS):
            sub_uris = uri_list[start_idx:start_idx + MAX_URIS]
            logging.info(
                f"Importing batch of {len(sub_uris)} URIs (from index {start_idx} to {start_idx+len(sub_uris)-1})."
            )

            # Call the existing 'import_files_from_gcs' method for each sub-list
            try:
                response = rag_corpus_mgr.import_files_from_gcs(
                    rag_corpus_resource=rag_corpus_res,
                    gcs_file_uris=sub_uris,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_embedding_requests_per_min=max_embedding_requests_per_min
                )
                # If 'import_files_from_gcs' returns a dict with import counts, parse them:
                if response:
                    imported_count = response.get('importedRagFilesCount', 0)
                    skipped_count = response.get('skippedRagFilesCount', 0)
                    total_imported += imported_count
                    total_skipped += skipped_count
            except Exception as e:
                logging.error(f"Batch import of {len(sub_uris)} URIs failed: {e}")
                # Decide whether to continue or break. Here we continue to try the next batch.
                continue

        logging.info(
            f"Finished batch-importing splitted JSON files. Summaries: "
            f"Imported={total_imported}, Skipped={total_skipped}."
        )
        return {
            'total_imported': total_imported,
            'total_skipped': total_skipped
        }