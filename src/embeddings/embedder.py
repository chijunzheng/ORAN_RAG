# src/embeddings/embedder.py

import json
import logging
import os
from google.cloud import storage
from typing import List, Dict
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.oauth2.credentials import Credentials

# New import for splitting
from src.utils.jsonl_splitter import split_jsonl_file

class Embedder:
    MAX_BATCH_SIZE = 25

    def __init__(self, config: dict, credentials: Credentials = None):
        """
        Initializes the Embedder with Google Cloud details and configurable embedding model.
        
        Args:
            config (dict): Configuration dictionary containing embedding parameters.
            credentials (Credentials, optional): GCS credentials. Defaults to None.
        """
        self.project_id = config.get('gcp', {}).get('project_id')
        self.location = config.get('gcp', {}).get('location')
        self.bucket_name = config.get('gcp', {}).get('bucket_name')
        self.embeddings_path = config.get('gcp', {}).get('embeddings_path')
        self.embedding_model_name = config.get('embedding', {}).get('embedding_model_name', "text-embedding-005")
        
        if credentials:
            self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
        else:
            self.storage_client = storage.Client(project=self.project_id)
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
        self.embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
        logging.info(f"Initialized Embedder with model '{self.embedding_model_name}'.")

    def generate_and_store_embeddings(self, chunks: List[Dict], local_jsonl_path: str = "embeddings.jsonl", batch_size: int = 10):
        """
        Generates embeddings for text chunks in batches and uploads them to GCS as split JSON files.
        
        Args:
            chunks (List[Dict]): List of text chunks with 'id' and 'text'.
            local_jsonl_path (str, optional): Local path to store the JSONL file. Defaults to "embeddings.jsonl".
            batch_size (int, optional): Number of chunks to process in each batch. Defaults to 10.
        """
        if batch_size > self.MAX_BATCH_SIZE:
            logging.warning(
                f"Requested batch_size {batch_size} exceeds MAX_BATCH_SIZE {self.MAX_BATCH_SIZE}. "
                f"Setting batch_size to {self.MAX_BATCH_SIZE}."
            )
            batch_size = self.MAX_BATCH_SIZE

        logging.info("Starting embeddings generation with batching.")
        try:
            with open(local_jsonl_path, 'w', encoding='utf-8') as f:
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    embedding_inputs = [
                        TextEmbeddingInput(task_type="RETRIEVAL_DOCUMENT", text=chunk['content']) for chunk in batch
                    ]
                    embeddings = self.embedding_model.get_embeddings(embedding_inputs)
                    
                    for chunk, embedding in zip(batch, embeddings):
                        embedding_data = {
                            "id": chunk['id'],
                            "embedding": embedding.values
                        }
                        f.write(json.dumps(embedding_data) + '\n')
                        logging.debug(f"Generated and wrote embedding for {chunk['id']}")
            logging.info(f"Embeddings saved to {local_jsonl_path}")
        except Exception as e:
            logging.error(f"Failed to save embeddings to {local_jsonl_path}: {e}")
            raise

        # 1. Split the embeddings JSONL into smaller files
        split_dir = os.path.join(os.path.dirname(local_jsonl_path), "embeddings_split_local")
        split_file_paths = split_jsonl_file(
            input_file=local_jsonl_path,
            output_dir=split_dir,
            max_size=1 * 1024 * 1024
        )

        # 2. Upload each split .jsonl file to GCS under embeddings_split/
        for split_file_path in split_file_paths:
            # rename to .jsonl if needed
            if not split_file_path.endswith('.jsonl'):
                new_path = split_file_path.replace('.json', '.jsonl')
                os.rename(split_file_path, new_path)
                split_file_path = new_path

            file_basename = os.path.basename(split_file_path)
            gcs_target_path = f"{self.embeddings_path}embeddings_split/{file_basename}"
            try:
                blob = self.bucket.blob(gcs_target_path)
                blob.upload_from_filename(split_file_path, content_type="application/json")
                logging.info(f"Uploaded split embedding file to GCS: {blob.name}")
            except Exception as e:
                logging.error(f"Failed to upload split embedding file {split_file_path} to GCS: {e}")
                raise