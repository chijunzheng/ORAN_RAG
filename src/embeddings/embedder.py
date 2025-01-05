# src/embeddings/embedder.py

import json
import logging
from google.cloud import storage
from typing import List, Dict
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.oauth2.credentials import Credentials

class Embedder:
    MAX_BATCH_SIZE = 25
    def __init__(self, project_id: str, 
                 location: str, 
                 bucket_name: str, 
                 embeddings_path: str,
                 credentials: Credentials = None):
        """
        Initializes the Embedder with Google Cloud details.
        
        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud region.
            bucket_name (str): GCS bucket name.
            embeddings_path (str): Path within the bucket to store embeddings.
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.embeddings_path = embeddings_path
        
        # Initialize the storage client with credentials if provided
        if credentials:
            self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
        else:
            self.storage_client = storage.Client(project=self.project_id)

        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    def generate_and_store_embeddings(self, chunks: List[Dict], local_jsonl_path: str = "embeddings.jsonl", batch_size: int = 10):
        """
        Generates embeddings for text chunks in batches and uploads them to GCS as a JSONL file.
        
        Args:
            chunks (List[Dict]): List of text chunks with 'id' and 'text'.
            local_jsonl_path (str, optional): Local path to store the JSONL file. Defaults to "embeddings.jsonl".
            batch_size (int, optional): Number of chunks to process in each batch. Defaults to 1000.
        """
        # Ensure batch_size does not exceed MAX_BATCH_SIZE
        if batch_size > self.MAX_BATCH_SIZE:
            logging.warning(f"Requested batch_size {batch_size} exceeds MAX_BATCH_SIZE {self.MAX_BATCH_SIZE}. Setting batch_size to {self.MAX_BATCH_SIZE}.")
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
                    
                    logging.info(f"Processed batch {i // batch_size + 1} with {len(batch)} chunks.")
            
            logging.info(f"Embeddings saved to {local_jsonl_path}")
        except Exception as e:
            logging.error(f"Failed to save embeddings to {local_jsonl_path}: {e}")
            raise

        # Upload to GCS
        try:
            embeddings_blob = self.bucket.blob(f"{self.embeddings_path}embeddings.json")
            embeddings_blob.upload_from_filename(local_jsonl_path, content_type="application/json")
            logging.info(f"Uploaded embeddings to GCS at {embeddings_blob.name}")
        except Exception as e:
            logging.error(f"Failed to upload embeddings to GCS: {e}")
            raise