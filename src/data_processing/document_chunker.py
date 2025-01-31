# src/data_processing/document_chunker.py

import logging
import json
import uuid
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from google.cloud import storage
import os
from google.oauth2.credentials import Credentials


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1536,
        chunk_overlap: int = 256,
        separators: List[str] = None,
        gcs_bucket_name: str = None,
        gcs_embeddings_path: str = "embeddings/",
        credentials: Credentials = None,
        min_char_count: int = 100  # Added minimum character threshold
    ):
        """
        Initializes the DocumentChunker with specified chunking parameters and GCS configuration.
        
        Args:
            chunk_size (int, optional): Maximum number of characters per chunk. Defaults to 1536.
            chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 256.
            separators (List[str], optional): List of separators to use for splitting. Defaults to [". ", "? ", "! ", "\n\n"].
            gcs_bucket_name (str, optional): Name of the GCS bucket to upload chunks. Defaults to None.
            gcs_embeddings_path (str, optional): Path within the GCS bucket to save embeddings. Defaults to "embeddings/".
            credentials (Credentials, optional): GCS credentials. Defaults to None.
            min_char_count (int, optional): Minimum number of characters required for a chunk. Defaults to 100.
        """
        if separators is None:
            separators = [". ", "? ", "! ", "\n\n"]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            #length_function=lambda text: len(text)  # Character count
            length_function=lambda text: len(text.split())  # Word count
        )
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_embeddings_path = gcs_embeddings_path
        self.credentials = credentials
        self.min_char_count = min_char_count  # Store the threshold

        logging.info(
            f"DocumentChunker initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}, separators={separators}, "
            f"gcs_bucket_name={gcs_bucket_name}, gcs_embeddings_path={gcs_embeddings_path}, "
            f"min_char_count={min_char_count}"
        )
    
    def split_documents(self, documents: List[Document]) -> List[Dict]:
        """
        Splits a list of Document objects into chunks and assigns unique IDs.
        
        Args:
            documents (List[Document]): List of cleaned Document objects.
        
        Returns:
            List[Dict]: List of dictionaries containing chunk information.
        """
        logging.info("Starting to split documents into chunks.")
        split_docs = self.splitter.split_documents(documents)
        logging.info(f"Total chunks created before filtering: {len(split_docs)}")
        
        chunks_with_ids = self.assign_ids(split_docs)
        logging.info(f"Total chunks after filtering: {len(chunks_with_ids)}")
        return chunks_with_ids
    
    def assign_ids(self, split_docs: List[Document]) -> List[Dict]:
        """
        Assigns a UUID to each chunk, calculates character count,
        and filters out chunks below min_char_count.

        Args:
            split_docs (List[Document]): List of Document chunks.

        Returns:
            List[Dict]: List of chunk dicts with a 'uuid' and chunk details.
        """
        logging.info("Assigning UUIDs and filtering chunks based on character count.")
        chunks_with_ids = []
        for doc in split_docs:
            # Calculate character count
            char_count = len(doc.page_content)
            if char_count < self.min_char_count:
                logging.debug(f"Skipped chunk due to low character count: {char_count} < {self.min_char_count}.")
                continue

            chunk_uuid = str(uuid.uuid4())
            document_name = doc.metadata.get('document_name')
            page_number = doc.metadata.get('page_number')

            chunk_dict = {
                'id': chunk_uuid,            # <--- Unique UUID
                'content': doc.page_content, # The chunk text
                'document_name': document_name,
                'page_number': page_number,
                'char_count': char_count
            }
            chunks_with_ids.append(chunk_dict)

        logging.info("Completed assigning UUIDs and filtering chunks.")
        return chunks_with_ids
    
    def save_chunks_to_json(self, chunks: List[Dict], file_path: str = "chunks.json"):
        """
        Saves the chunk mappings to a JSON Lines (.jsonl) file.
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries.
            file_path (str, optional): Path to save the JSON Lines file. Defaults to "chunks.json".
        """
        logging.info(f"Saving {len(chunks)} chunks to {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    # Build a record that includes metadata
                    line_dict = {
                        'id': chunk.get('id'),
                        'content': chunk.get('content', ""),
                        'document_name': chunk.get('document_name'),
                        'page_number': chunk.get('page_number'),
                        'char_count': chunk.get('char_count'),
                        'metadata': chunk.get('metadata', {})  # Preserve all chunk-specific metadata
                    }
                    json_record = json.dumps(line_dict, ensure_ascii=False)
                    f.write(json_record + '\n')
            logging.info(f"Chunk data with IDs and metadata saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save chunks to {file_path}: {e}")
            raise
    
    def upload_to_gcs(self, file_path: str, overwrite: bool = True):
        """
        Uploads a file to the specified Google Cloud Storage bucket.
        
        Args:
            file_path (str): Local path of the file to upload.
            overwrite (bool, optional): Whether to overwrite the file if it exists in GCS. Defaults to True.
        """
        if not self.gcs_bucket_name:
            logging.error("GCS bucket name not provided. Cannot upload file.")
            raise ValueError("GCS bucket name not provided.")
        
        logging.info(f"Uploading {file_path} to GCS bucket {self.gcs_bucket_name} at {self.gcs_embeddings_path}")
        try:
            # Initialize the GCS client with credentials if provided
            if self.credentials:
                storage_client = storage.Client(credentials=self.credentials, project=self.credentials.project_id)
            else:
                storage_client = storage.Client()
            
            bucket = storage_client.bucket(self.gcs_bucket_name)
            
            # Define the destination path in GCS
            destination_blob_name = os.path.join(
                self.gcs_embeddings_path, os.path.basename(file_path)
            )
            blob = bucket.blob(destination_blob_name)
            
            if blob.exists() and not overwrite:
                logging.warning(f"Blob {destination_blob_name} already exists and overwrite is set to False.")
                return
            
            blob.upload_from_filename(file_path, content_type="application/json")
            logging.info(f"Uploaded {file_path} to GCS: {destination_blob_name}")
        except Exception as e:
            logging.error(f"Failed to upload {file_path} to GCS: {e}")
            raise