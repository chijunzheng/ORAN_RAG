# src/vector_search/reranker.py

import logging
from typing import List, Dict
from vertexai.preview import rag
import vertexai

class Reranker:
    def __init__(self, 
                 config: dict,
                 credentials
                 ):
        """
        Initializes the Reranker with configurations from config.yaml.

        Args:
            config (dict): Configuration dictionary containing reranker parameters.
        """
        try:
            # Extract Reranker configuration
            self.project_id = config.get('gcp', {}).get('project_id')
            self.location = config.get('gcp', {}).get('location')
            self.rag_corpus_resource = config.get('vector_search', {}).get('rag_corpus_resource')
            self.model_name = config.get('vector_search', {}).get('reranker_model_name')
            self.reranker_type = config.get('vector_search', {}).get('reranker_type', 'rank_service')  # 'llm' or 'rank_service'

            if not all([self.project_id, self.location, self.rag_corpus_resource, self.model_name]):
                raise ValueError("Missing reranker configuration parameters in config.yaml.")

            # Initialize Vertex AI API
            vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            logging.info(f"Initialized Reranker with project_id='{self.project_id}', location='{self.location}'")

            # Validate reranker type
            if self.reranker_type not in ['llm', 'rank_service']:
                raise ValueError("reranker_type must be either 'llm' or 'rank_service'.")

        except Exception as e:
            logging.error(f"Error initializing Reranker: {e}", exc_info=True)
            raise

    def rerank_chunks_rank_service(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Uses the Vertex AI Rank Service reranker to assess and reorder the relevance of chunks.

        Args:
            query (str): The user's query.
            chunks (List[Dict]): List of retrieved chunks.

        Returns:
            List[Dict]: Reranked list of chunks.
        """
        try:
            # Perform retrieval query with reranker
            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=self.rag_corpus_resource,
                    )
                ],
                text=query,
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=len(chunks),
                    ranking=rag.Ranking(
                        rank_service=rag.RankService(
                            model_name=self.model_name  # Uses the configured reranker model
                        )
                    )
                )
            )

            reranked_chunks = []
            for context in response.contexts:
                reranked_chunks.append({
                    'source_uri': context.source_uri,
                    'text': context.text,
                    'distance': context.distance  # Assuming distance is available
                })

            logging.info(f"Reranked {len(reranked_chunks)} chunks using Rank Service reranker.")
            return reranked_chunks

        except Exception as e:
            logging.error(f"Rank Service Reranker failed: {e}", exc_info=True)
            raise

    def rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Selects the reranking method based on configuration and reranks the chunks.

        Args:
            query (str): The user's query.
            chunks (List[Dict]): List of retrieved chunks.

        Returns:
            List[Dict]: Reranked list of chunks.
        """
        if self.reranker_type == 'rank_service':
            return self.rerank_chunks_rank_service(query, chunks)
        else:
            logging.error(f"Unsupported reranker_type: {self.reranker_type}")
            return chunks  # Return original chunks if reranker type is unsupported