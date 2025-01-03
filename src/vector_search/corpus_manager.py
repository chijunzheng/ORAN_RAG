# src/vector_search/corpus_manager.py

import logging
from typing import List
from vertexai.preview import rag
import vertexai
from src.utils.config_manager import ConfigManager  # Ensure this import does not cause circular imports
import os


class RagCorpusManager:
    def __init__(self, 
                 config: dict,
                 credentials
                 ):
        """
        Initializes the RagCorpusManager with configurations from config.yaml.

        Args:
            config (dict): Configuration dictionary containing RAG corpus parameters.
            credentials: Google Cloud credentials.
        """
        try:
            # Extract RAG corpus configuration
            self.project_id = config.get('gcp', {}).get('project_id')
            self.location = config.get('gcp', {}).get('location')
            self.rag_corpus_resource = config.get('vector_search', {}).get('rag_corpus_resource')
            self.display_name = config.get('vector_search', {}).get('rag_corpus_display_name', "O-RAN RAG Corpus")
            self.description = config.get('vector_search', {}).get('rag_corpus_description', "Corpus for O-RAN RAG Pipeline")
            self.embedding_model_config = config.get('embedding', {}).get('embedding_model_name')
            self.publisher_model = config.get('embedding', {}).get('publisher_model')
            self.vector_db_config = config.get('vector_search', {}).get('vector_db_config', {})
            self.update_method = config.get('vector_search', {}).get('update_method', 'STREAM_UPDATE')
            self.credentials = credentials
            self.credentials_path = config.get('gcp', {}).get('credentials_path')
            if not self.credentials_path:
                raise ValueError("credentials_path must be specified in config.yaml under 'gcp'")
            
            if not all([self.project_id, self.location]):
                raise ValueError("Missing GCP configuration parameters in config.yaml.")

            if not self.vector_db_config:
                raise ValueError("vector_db_config must be specified under 'vector_search' in config.yaml.")

            # Initialize Vertex AI and RAG
            vertexai.init(project=self.project_id, location=self.location, credentials=self.credentials)
            logging.info(f"Initialized RagCorpusManager with project_id='{self.project_id}', location='{self.location}'")

        except KeyError as ke:
            logging.error(f"Missing configuration key: {ke}")
            raise ValueError(f"Missing configuration key: {ke}") from ke
        except Exception as e:
            logging.error(f"Error initializing RagCorpusManager: {e}", exc_info=True)
            raise

    def initialize_corpus(self, display_name: str, description: str) -> str:
        """
        Creates a new RAG corpus if it does not already exist.

        Args:
            display_name (str): Display name for the RAG corpus.
            description (str): Description for the RAG corpus.

        Returns:
            str: The resource name of the initialized RAG Corpus.
        """
        try:
            logging.info("Checking for existing RAG corpora.")
            existing_corpora = rag.list_corpora()

            # Check if a corpus with the same display name exists
            for corpus in existing_corpora:
                if corpus.display_name == display_name:
                    logging.info(f"RAG Corpus '{display_name}' already exists: {corpus.name}")
                    return corpus.name

            # Create a new RAG corpus
            logging.info(f"Creating new RAG Corpus: {display_name}")
            embedding_model = rag.EmbeddingModelConfig(
                publisher_model=self.publisher_model
            )

            # Initialize Vertex Vector Search Configuration with both index and index_endpoint
            vertex_vector_search = rag.VertexVectorSearch(
                index=self.vector_db_config['vertex_vector_search']['index'],
                index_endpoint=self.vector_db_config['vertex_vector_search']['index_endpoint']
            )

            # Create the RAG Corpus with the vector database configuration
            corpus = rag.create_corpus(
                display_name=display_name,
                description=description,
                embedding_model_config=embedding_model,
                vector_db=vertex_vector_search,  # Use VertexVectorSearch with both index and index_endpoint
            )
            logging.info(f"Created RAG Corpus: {corpus.name}")

            # Update config.yaml with the new rag_corpus_resource
            self.update_config_with_corpus_resource(corpus.name)

            return corpus.name

        except Exception as e:
            logging.error(f"Failed to initialize RAG Corpus: {e}", exc_info=True)
            raise

    def update_config_with_corpus_resource(self, rag_corpus_resource: str):
        """
        Updates the config.yaml with the new rag_corpus_resource.

        Args:
            rag_corpus_resource (str): The resource name of the newly created RAG Corpus.
        """
        try:
            config_manager = ConfigManager(config_path='configs/config.yaml')
            config_manager.update_config({
                'vector_search.rag_corpus_resource': rag_corpus_resource
            })
            logging.info(f"Updated config.yaml with rag_corpus_resource='{rag_corpus_resource}'")
        except Exception as e:
            logging.error(f"Failed to update config.yaml with rag_corpus_resource: {e}", exc_info=True)
            raise

    def import_files_from_gcs(self, rag_corpus_resource: str, gcs_file_uris: List[str], chunk_size: int = 1536, chunk_overlap: int = 256, max_embedding_requests_per_min: int = 1000):
        """
        Imports files directly from GCS into the specified RAG Corpus using rag.import_files.

        Args:
            rag_corpus_resource (str): The resource name of the RAG Corpus.
            gcs_file_uris (List[str]): List of GCS URIs pointing to the files to import.
            chunk_size (int, optional): Number of tokens each chunk has. Defaults to 512.
            chunk_overlap (int, optional): The overlap between chunks. Defaults to 100.
            max_embedding_requests_per_min (int, optional): The maximum number of embedding requests per minute. Defaults to 1000.
        """
        try:
            if not rag_corpus_resource:
                raise ValueError("rag_corpus_resource must be provided.")

            logging.info(f"Importing {len(gcs_file_uris)} files into RAG Corpus '{rag_corpus_resource}' from GCS.")

            response = rag.import_files(
                corpus_name=rag_corpus_resource,
                paths=gcs_file_uris,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_embedding_requests_per_min=max_embedding_requests_per_min,
            )
            logging.info(f"Imported {response.imported_rag_files_count} files into RAG Corpus '{rag_corpus_resource}'.")
            logging.info(f"Skipped {response.skipped_rag_files_count} files.")

            # Return the response so the caller can parse
            return {
                'importedRagFilesCount': response.imported_rag_files_count,
                'skippedRagFilesCount': response.skipped_rag_files_count
            }
        except Exception as e:
            logging.error(f"Failed to import files into RAG Corpus: {e}", exc_info=True)
            raise