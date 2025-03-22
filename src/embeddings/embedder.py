# src/embeddings/embedder.py

import json
import logging
from typing import List, Dict
from google.cloud import storage
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.oauth2.credentials import Credentials

# Import tiktoken for accurate token counting.
try:
    import tiktoken
except ImportError:
    logging.warning("tiktoken is not installed. Install it with 'pip install tiktoken' for accurate token counting.")
    tiktoken = None

def count_tokens(text: str) -> int:
    """
    Uses tiktoken to count tokens for a given text.
    If tiktoken is unavailable, falls back to a simple whitespace split.
    """
    if tiktoken:
        try:
            # Try to use the encoding specific for "text-embedding-005"
            encoding = tiktoken.encoding_for_model("text-embedding-005")
        except Exception:
            # Fallback to cl100k_base encoding
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    else:
        return len(text.split())

class Embedder:
    # Maximum number of chunks per batch (API parameter)
    MAX_BATCH_SIZE = 25
    # The API supports up to 20,000 tokens per call.
    MAX_API_TOKENS = 20000
    # We use a safe limit below that (e.g. 19,000 tokens) for our batching logic.
    SAFE_TOKEN_LIMIT = 19000  

    def __init__(self, project_id: str, location: str, bucket_name: str, embeddings_path: str, credentials: Credentials = None):
        """
        Initializes the Embedder with Google Cloud details.
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.embeddings_path = embeddings_path

        if credentials:
            self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
        else:
            self.storage_client = storage.Client(project=self.project_id)
        self.bucket = self.storage_client.bucket(self.bucket_name)

        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        logging.info("Initialized TextEmbeddingModel from 'text-embedding-005'")

    @staticmethod
    def load_json_file(file_path):
        """
        Load data from a JSON file where each line is a valid JSON object.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            list: List of parsed JSON objects
        """
        logging.info(f"Loading data from JSON file: {file_path}")
        results = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        results.append(json.loads(line))
            logging.info(f"Successfully loaded {len(results)} records from {file_path}")
            return results
        except Exception as e:
            logging.error(f"Failed to load JSON file {file_path}: {e}", exc_info=True)
            raise

    def generate_and_store_embeddings(self, chunks: List[Dict], local_json_path: str = "embeddings.json", batch_size: int = 9):
        """
        Generates embeddings for text chunks in batches using dynamic batching.
        The method ensures that the total token count in each API call does not exceed our SAFE_TOKEN_LIMIT.
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries (each with 'id' and 'content').
            local_json_path (str, optional): Path to save the JSON file.
            batch_size (int, optional): Maximum number of chunks per batch.
        """
        if batch_size > self.MAX_BATCH_SIZE:
            logging.warning(f"Requested batch_size {batch_size} exceeds MAX_BATCH_SIZE {self.MAX_BATCH_SIZE}. Using {self.MAX_BATCH_SIZE} instead.")
            batch_size = self.MAX_BATCH_SIZE

        logging.info("Starting embeddings generation with dynamic batching.")
        results = []
        total_chunks = len(chunks)
        i = 0
        batch_num = 1

        while i < total_chunks:
            current_batch = []
            current_batch_tokens = 0

            # Accumulate chunks into the current batch while ensuring the batch token count stays under SAFE_TOKEN_LIMIT.
            while i < total_chunks and len(current_batch) < batch_size:
                chunk = chunks[i]
                tokens = count_tokens(chunk['content'])

                # If a single chunk exceeds the safe limit, trim it to SAFE_TOKEN_LIMIT tokens.
                if tokens > self.SAFE_TOKEN_LIMIT:
                    logging.warning(f"Chunk {chunk['id']} has {tokens} tokens, exceeding the safe limit. Trimming to {self.SAFE_TOKEN_LIMIT} tokens.")
                    words = chunk['content'].split()
                    # Trim by word; note that this is approximate.
                    trimmed_content = " ".join(words[:self.SAFE_TOKEN_LIMIT])
                    chunk['content'] = trimmed_content
                    tokens = count_tokens(chunk['content'])

                # If the batch is non-empty and adding this chunk would exceed the safe limit, break to process the batch.
                if current_batch and (current_batch_tokens + tokens > self.SAFE_TOKEN_LIMIT):
                    break

                current_batch.append(chunk)
                current_batch_tokens += tokens
                i += 1

            logging.info(f"Processing batch {batch_num}: {len(current_batch)} chunks with total token count {current_batch_tokens}.")
            batch_num += 1

            try:
                embedding_inputs = [
                    TextEmbeddingInput(task_type="RETRIEVAL_DOCUMENT", text=chunk['content'])
                    for chunk in current_batch
                ]
                embeddings = self.embedding_model.get_embeddings(embedding_inputs)
            except Exception as e:
                logging.error(f"Failed to generate embeddings for batch starting at index {i}: {e}", exc_info=True)
                raise

            # Pair each embedding with its chunk ID.
            for chunk, embedding in zip(current_batch, embeddings):
                results.append({
                    "id": chunk['id'],
                    "embedding": embedding.values
                })

        # Save the embeddings to a JSON file.
        try:
            with open(local_json_path, 'w', encoding='utf-8') as f:
                for record in results:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logging.info(f"Embeddings saved to {local_json_path}")
        except Exception as e:
            logging.error(f"Failed to save embeddings to {local_json_path}: {e}", exc_info=True)
            raise

        # Upload the JSON file to GCS.
        try:
            destination_blob_name = f"{self.embeddings_path}embeddings.json"
            embeddings_blob = self.bucket.blob(destination_blob_name)
            embeddings_blob.upload_from_filename(local_json_path, content_type="application/json")
            logging.info(f"Uploaded embeddings to GCS at {destination_blob_name}")
        except Exception as e:
            logging.error(f"Failed to upload embeddings to GCS: {e}", exc_info=True)
            raise