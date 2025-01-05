# src/vector_search/reranker.py

import logging
from typing import List, Dict
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.exceptions import GoogleAPICallError, NotFound, FailedPrecondition

class Reranker:
    def __init__(
        self,
        project_id: str,
        location: str,
        ranking_config: str,
        credentials,  
        model: str = "semantic-ranker-512@latest",
        rerank_top_n: int = 10
    ):
        """
        Initializes the Reranker with necessary configurations.
        
        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud location (e.g., 'us-central1').
            ranking_config (str): Name of the ranking configuration.
            credentials: Google Cloud credentials object.
            model (str, optional): Model name for ranking. Defaults to "semantic-ranker-512@latest".
        """
        self.project_id = project_id
        self.location = location
        self.ranking_config = ranking_config
        self.model = model
        self.rerank_top_n = rerank_top_n
        
        try:
            self.client = discoveryengine.RankServiceClient(credentials=credentials)
            self.ranking_config_path = self.client.ranking_config_path(
                project=self.project_id,
                location=self.location,
                ranking_config=self.ranking_config,
            )
            logging.info(f"Initialized Reranker with ranking_config='{self.ranking_config_path}'")
        except Exception as e:
            logging.error(f"Failed to initialize Reranker: {e}", exc_info=True)
            raise

    def rerank(self, query: str, records: List[Dict]) -> List[Dict]:
        """
        Reranks the provided records based on the query using the ranking API.
        
        Args:
            query (str): The user's query.
            records (List[Dict]): List of records to rerank. Each record must have 'id', 'title', and/or 'content'.
            top_n (int, optional): Number of top records to return. Defaults to 10.
        
        Returns:
            List[Dict]: Reranked list of records.
        """
        # Prepare RankingRecords
        ranking_records = []
        for record in records:
            # Ensure each record has at least 'content'. 'title' is optional.
            if 'content' not in record or not record['content']:
                logging.warning(f"Record {record.get('id', 'unknown')} is missing 'content'. Skipping.")
                continue
            ranking_record = discoveryengine.RankingRecord(
                id=record['id'],
                title=record.get('title', ''),
                content=record['content'][:512] # Limit content to 512 characters
            )
            ranking_records.append(ranking_record)
        
        if not ranking_records:
            logging.warning("No valid records to rerank.")
            return []
        
        # Construct the RankRequest
        request = discoveryengine.RankRequest(
            ranking_config=self.ranking_config_path,
            model=self.model,
            top_n=self.rerank_top_n,
            query=query,
            records=ranking_records,
            ignore_record_details_in_response=False,  # Set to True if you only need 'id' and 'score'
            user_labels={}  # Add any necessary user labels here
        )
        
        try:
            response = self.client.rank(request=request)
            logging.info(f"Reranking completed successfully for query: '{query}'")
            logging.debug(f"RankResponse: {response}")
        except (GoogleAPICallError, NotFound, FailedPrecondition) as e:
            logging.error(f"Reranking API call failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during reranking: {e}", exc_info=True)
            raise
        
        # Map the response to the original records
        reranked_records = []
        try:
            for ranked_record in response.records:
                datapoint_id = ranked_record.id
                score = ranked_record.score
                # Find the original record
                original_record = next((rec for rec in records if rec['id'] == datapoint_id), None)
                if original_record:
                    original_record['rank_score'] = score
                    reranked_records.append(original_record)
        except AttributeError as ae:
            logging.error(f"AttributeError while accessing RankResponse fields: {ae}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Error while processing RankResponse: {e}", exc_info=True)
            raise
        
        # Sort the records based on the score in descending order
        reranked_records.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        
        return reranked_records[:self.rerank_top_n]