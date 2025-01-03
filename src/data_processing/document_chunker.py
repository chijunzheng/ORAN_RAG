# src/data_processing/document_chunker.py

import logging
import json
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from google.cloud import storage
import os
from google.oauth2.credentials import Credentials

from src.utils.jsonl_splitter import split_jsonl_file

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1536,
        chunk_overlap: int = 256,
        separators: List[str] = None,
        gcs_bucket_name: str = None,
        gcs_embeddings_path: str = "embeddings/",
        credentials: Credentials = None,
        min_char_count: int = 100
    ):
        if separators is None:
            separators = [". ", "? ", "! ", "\n\n"]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=lambda text: len(text)
        )
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_embeddings_path = gcs_embeddings_path
        self.credentials = credentials
        self.min_char_count = min_char_count

        self.project_id = None
        if credentials:
            self.project_id = getattr(credentials, "project_id", None)
        
        logging.info(
            f"DocumentChunker initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
            f"separators={separators}, gcs_bucket_name={gcs_bucket_name}, "
            f"gcs_embeddings_path={gcs_embeddings_path}, min_char_count={min_char_count}"
        )

    def split_documents(self, documents: List[Document]) -> List[Dict]:
        logging.info("Starting to split documents into chunks.")
        split_docs = self.splitter.split_documents(documents)
        logging.info(f"Total chunks created before filtering: {len(split_docs)}")
        
        chunks_with_ids = self.assign_ids(split_docs)
        logging.info(f"Total chunks after filtering: {len(chunks_with_ids)}")
        return chunks_with_ids
    
    def assign_ids(self, split_docs: List[Document]) -> List[Dict]:
        logging.info("Assigning unique IDs and calculating character counts for each chunk.")
        chunks_with_ids = []
        for i, doc in enumerate(split_docs):
            document_name = doc.metadata.get('document_name')
            page_number = doc.metadata.get('page_number')
            char_count = len(doc.page_content)
            
            if char_count < self.min_char_count:
                logging.debug(f"Skipped chunk_{i} due to low char count: {char_count}.")
                continue
            
            chunk_id = f"chunk_{i}"
            chunk = {
                'id': chunk_id,
                'content': doc.page_content,
                'document_name': document_name,
                'page_number': page_number,
                'char_count': char_count
            }
            chunks_with_ids.append(chunk)
        logging.info("Completed assigning IDs and filtering chunks based on character count.")
        return chunks_with_ids
    
    def save_chunks_to_jsonl(self, chunks: List[Dict], file_path: str = "chunks.jsonl"):
        """
        Saves the chunk mappings to a JSON Lines (.jsonl) file (newline-delimited JSON).
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries.
            file_path (str, optional): Path to save the .jsonl file. Defaults to "chunks.jsonl".
        """
        logging.info(f"Saving {len(chunks)} chunks to {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    json_record = json.dumps(chunk)
                    f.write(json_record + '\n')
            logging.info(f"Chunk data with IDs saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save chunks to {file_path}: {e}")
            raise

    def upload_to_gcs(self, file_path: str, overwrite: bool = True):
        """
        Uploads a file (e.g., .jsonl) to GCS without splitting. 
        """
        if not self.gcs_bucket_name:
            logging.error("GCS bucket name not provided. Cannot upload file.")
            raise ValueError("GCS bucket name not provided.")
        
        logging.info(f"Uploading {file_path} to GCS bucket {self.gcs_bucket_name} at {self.gcs_embeddings_path}")
        try:
            from google.cloud import storage
            if self.credentials:
                storage_client = storage.Client(credentials=self.credentials, project=self.project_id)
            else:
                storage_client = storage.Client()

            bucket = storage_client.bucket(self.gcs_bucket_name)
            destination_blob_name = os.path.join(
                self.gcs_embeddings_path, os.path.basename(file_path)
            )
            blob = bucket.blob(destination_blob_name)

            if blob.exists() and not overwrite:
                logging.warning(f"Blob {destination_blob_name} already exists and overwrite=False.")
                return
            
            blob.upload_from_filename(file_path, content_type="application/json")
            logging.info(f"Uploaded {file_path} to GCS: {destination_blob_name}")
        except Exception as e:
            logging.error(f"Failed to upload {file_path} to GCS: {e}")
            raise

    def split_and_upload_chunks(self, local_chunks_file: str):
        """
        Splits the final chunks.jsonl into multiple <1 MB files and uploads them to
        gs://<bucket>/<embeddings_path>/chunks_split/.
        
        Args:
            local_chunks_file (str): Path to the local chunks.jsonl file.
        """
        if not os.path.exists(local_chunks_file):
            logging.error(f"Local chunks file not found: {local_chunks_file}")
            raise FileNotFoundError(f"Local chunks file not found: {local_chunks_file}")

        if not self.gcs_bucket_name:
            logging.error("GCS bucket name not provided. Cannot split and upload chunks.")
            raise ValueError("GCS bucket name not provided.")
        
        logging.info(f"Splitting NDJSON file '{local_chunks_file}' into 1MB parts...")
        split_dir = os.path.join(os.path.dirname(local_chunks_file), "chunks_split_local")
        os.makedirs(split_dir, exist_ok=True)

        split_file_paths = split_jsonl_file(
            input_file=local_chunks_file,
            output_dir=split_dir,
            max_size=1 * 1024 * 1024  # 1 MB
        )

        try:
            from google.cloud import storage
            if self.credentials:
                storage_client = storage.Client(credentials=self.credentials, project=self.project_id)
            else:
                storage_client = storage.Client()

            bucket = storage_client.bucket(self.gcs_bucket_name)

            for split_file_path in split_file_paths:
                # rename .json -> .jsonl if you prefer
                if not split_file_path.endswith('.jsonl'):
                    # e.g. rename the file from part_1.json -> part_1.jsonl
                    new_path = split_file_path.replace('.json', '.jsonl')
                    os.rename(split_file_path, new_path)
                    split_file_path = new_path

                file_basename = os.path.basename(split_file_path)
                gcs_target_path = f"{self.gcs_embeddings_path}chunks_split/{file_basename}"
                blob = bucket.blob(gcs_target_path)
                blob.upload_from_filename(split_file_path, content_type="application/json")
                logging.info(f"Uploaded split chunk NDJSON file to GCS: {blob.name}")

        except Exception as e:
            logging.error(f"Failed to upload split chunks to GCS: {e}")
            raise