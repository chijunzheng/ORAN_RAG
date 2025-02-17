# src/vector_search/searcher.py

import json
import logging
from typing import List, Dict
from google.cloud import storage, aiplatform
from google.api_core.exceptions import GoogleAPICallError, NotFound, FailedPrecondition
from src.utils.helpers import ensure_directory
from src.vector_search.indexer import VectorIndexer
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput


class VectorSearcher:
    def __init__(self, 
                 project_id: str,
                 location: str,
                 bucket_name: str,
                 embeddings_path: str,
                 bucket_uri: str,
                 credentials):
        """
        Initializes the VectorSearcher with specific configuration parameters.

        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud location (e.g., 'us-central1').
            bucket_name (str): Name of the GCS bucket.
            embeddings_path (str): Path within the GCS bucket for embeddings.
            bucket_uri (str): URI of the GCS bucket.
            credentials: Google Cloud credentials object.
        """
        try:
            self.project_id = project_id
            self.location = location
            self.bucket_name = bucket_name
            self.embeddings_path = embeddings_path
            self.bucket_uri = bucket_uri

            # Initialize AI Platform
            aiplatform.init(project=self.project_id, location=self.location, credentials=credentials)
            logging.info(f"Initialized VectorSearcher with project_id='{self.project_id}', location='{self.location}', bucket_uri='{self.bucket_uri}'")
            
            # Initialize Storage Client
            self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
            self.bucket = self.storage_client.bucket(self.bucket_name)
            
            # Initialize Embedding Model
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            logging.info("Initialized TextEmbeddingModel from 'text-embedding-005'")
            
            # Download and load chunks once during initialization
            self.chunks = self.download_chunks()
            logging.info("Chunks loaded into memory during VectorSearcher initialization.")
        
        except KeyError as ke:
            logging.error(f"Missing configuration key: {ke}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to initialize VectorSearcher: {e}", exc_info=True)
            raise

    def download_chunks(self, chunks_file_path: str = 'chunks.json') -> Dict[str, Dict]:
        """
        Downloads the chunks.json file from GCS and loads it into a dictionary.

        Args:
            chunks_file_path (str, optional): Local path to save the chunks file. Defaults to 'chunks.json'.

        Returns:
            Dict[str, Dict]: Mapping from chunk ID to chunk data.
        """
        try:
            chunks_blob = self.bucket.blob(f"{self.embeddings_path}chunks.json")
            chunks_blob.download_to_filename(chunks_file_path)
            logging.info(f"Downloaded chunks file from GCS: {chunks_blob.name}")
        except Exception as e:
            logging.error(f"Failed to download chunks file from GCS: {e}", exc_info=True)
            raise

        id_to_chunk = {}
        try:
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line.strip())
                    chunk_id = chunk.get('id')
                    text_content = chunk.get('content')
                    document_name = chunk.get('document_name')
                    page_number = chunk.get('page_number')
                    meta = chunk.get('metadata', {})
                    if chunk_id and text_content:
                        id_to_chunk[chunk_id] = {
                            'id': chunk_id,
                            'content': text_content,
                            'document_name': document_name,
                            'page_number': page_number,
                            'metadata': meta
                        }
            logging.info(f"Loaded {len(id_to_chunk)} chunks into memory.")
            return id_to_chunk
        except FileNotFoundError:
            logging.error(f"Chunks file not found at {chunks_file_path}", exc_info=True)
            raise
        except json.JSONDecodeError as jde:
            logging.error(f"JSON decode error while loading chunks: {jde}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to load chunks from {chunks_file_path}: {e}", exc_info=True)
            raise

    def vector_search(
        self,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        query_text: str,
        num_neighbors: int = 5
    ) -> List[Dict]:
        """
        Performs a vector search query against the deployed index.

        Args:
            index_endpoint_display_name (str): Display name of the index endpoint.
            deployed_index_id (str): ID of the deployed index.
            query_text (str): User query text.
            num_neighbors (int, optional): Number of nearest neighbors to retrieve. Defaults to 5.

        Returns:
            List[Dict]: List of retrieved chunks with metadata.
        """
        # Validate search parameters
        try:
            self.validate_search_params(index_endpoint_display_name, deployed_index_id, query_text, num_neighbors)
        except ValueError as ve:
            logging.error(f"Invalid search parameters: {ve}", exc_info=True)
            raise

        logging.info(f"Performing vector search for query: '{query_text}' with {num_neighbors} neighbors.")

        # Generate embedding for the query
        try:
            embedding_input = TextEmbeddingInput(
                task_type="RETRIEVAL_DOCUMENT",
                text=query_text
            )
            embedding = self.embedding_model.get_embeddings([embedding_input])[0]
            query_embedding = embedding.values
            logging.debug(f"Generated embedding for query: {query_embedding}")
        except Exception as e:
            logging.error(f"Failed to generate embedding for query: {e}", exc_info=True)
            raise

        # Retrieve the full resource name of the index endpoint
        try:
            index_endpoint_resource_name = self.get_index_endpoint_resource_name(index_endpoint_display_name)
        except Exception as e:
            logging.error(f"Failed to retrieve resource name for endpoint '{index_endpoint_display_name}': {e}", exc_info=True)
            raise

        # Initialize the index endpoint with full resource name
        try:
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_resource_name)
            logging.info(f"Initialized MatchingEngineIndexEndpoint: {index_endpoint_resource_name}")
        except Exception as e:
            logging.error(f"Failed to initialize MatchingEngineIndexEndpoint '{index_endpoint_resource_name}': {e}", exc_info=True)
            raise

        # Perform the search
        try:
            response = index_endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=[query_embedding],
                num_neighbors=num_neighbors,
            )
            logging.info("Vector search query executed successfully.")
        except Exception as e:
            logging.error(f"Failed to perform vector search: {e}", exc_info=True)
            raise

        retrieved_chunks = []
        try:
            for query_result in response:
                for neighbor in query_result:
                    datapoint_id = neighbor.id
                    distance = neighbor.distance
                    chunk_data = self.chunks.get(datapoint_id, {})
                    text_content = chunk_data.get('content', "Content not found.")
                    document_name = chunk_data.get('document_name', "Unknown Document")
                    page_number = chunk_data.get('page_number', "Unknown Page")
                    metadata = chunk_data.get('metadata', {})
                    retrieved_chunks.append({
                        'id': datapoint_id,
                        'distance': distance,
                        'content': text_content,
                        'document_name': document_name,
                        'page_number': page_number,
                        'metadata': metadata,
                    })
            logging.info(f"Retrieved {len(retrieved_chunks)} chunks for the query.")
            return retrieved_chunks
        except Exception as e:
            logging.error(f"Error processing search results: {e}", exc_info=True)
            raise

    def get_index_endpoint_resource_name(self, display_name: str) -> str:
        """
        Retrieves the full resource name of the index endpoint based on its display name.

        Args:
            display_name (str): The display name of the index endpoint.

        Returns:
            str: The full resource name of the index endpoint.

        Raises:
            ValueError: If the endpoint is not found or multiple endpoints have the same display name.
        """
        try:
            # List all existing index endpoints
            index_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
            logging.info(f"Retrieved {len(index_endpoints)} index endpoints.")

            # Filter endpoints matching the display name
            matching_endpoints = [endpoint for endpoint in index_endpoints if endpoint.display_name == display_name]
            
            if not matching_endpoints:
                raise ValueError(f"No index endpoint found with display name '{display_name}'.")
            elif len(matching_endpoints) > 1:
                raise ValueError(f"Multiple index endpoints found with display name '{display_name}'. Please ensure unique display names.")
            
            # Return the full resource name
            resource_name = matching_endpoints[0].resource_name
            logging.info(f"Found index endpoint '{display_name}' with resource name '{resource_name}'.")
            return resource_name

        except GoogleAPICallError as e:
            logging.error(f"API call error while retrieving index endpoints: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Error while retrieving index endpoint resource name: {e}", exc_info=True)
            raise

    def validate_search_params(
            self,
            index_endpoint_display_name: str,
            deployed_index_id: str,
            query_text: str,
            num_neighbors: int
        ):
        """
        Validates parameters for performing a vector search.

        Args:
            index_endpoint_display_name (str): Display name of the index endpoint.
            deployed_index_id (str): ID of the deployed index.
            query_text (str): User query text.
            num_neighbors (int): Number of nearest neighbors to retrieve.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not index_endpoint_display_name:
            raise ValueError("index_endpoint_display_name cannot be empty.")
        if not deployed_index_id:
            raise ValueError("deployed_index_id cannot be empty.")
        if not query_text:
            raise ValueError("query_text cannot be empty.")
        if not isinstance(num_neighbors, int) or num_neighbors <= 0:
            raise ValueError("num_neighbors must be a positive integer.")