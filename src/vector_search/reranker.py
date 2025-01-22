# src/vector_search/reranker.py

import logging
from typing import List, Dict, Tuple
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
        rerank_top_n: int = 20
    ):
        """
        Initializes the Reranker with necessary configurations.
        
        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud location (e.g., 'us-central1').
            ranking_config (str): Name of the ranking configuration.
            credentials: Google Cloud credentials object.
            model (str, optional): Model name for ranking. Defaults to "semantic-ranker-512@latest".
            rerank_top_n (int, optional): Number of top records to return after reranking. Defaults to 10.
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
        Ensures that all records have unique IDs before sending the request.
        
        Args:
            query (str): The user's query.
            records (List[Dict]): List of records to rerank. Each record must have 'id', 'title', and/or 'content'.
        
        Returns:
            List[Dict]: Reranked list of records.
        """
        # Step 1: Deduplicate Records
        unique_records = self._deduplicate_records(records)
        logging.info(f"Deduplicated records to {len(unique_records)} unique chunks.")

        if not unique_records:
            logging.warning("No unique records available for reranking.")
            return []

        try:
            # Step 2: Prepare RankingRecords
            ranking_records = self._prepare_ranking_records(unique_records)
            logging.info(f"Prepared {len(ranking_records)} RankingRecords for the API.")

            if not ranking_records:
                logging.warning("No valid RankingRecords to send to the API.")
                return []

            # Step 3: Construct the RankRequest
            request = discoveryengine.RankRequest(
                ranking_config=self.ranking_config_path,
                model=self.model,
                top_n=self.rerank_top_n,
                query=query,
                records=ranking_records,
                ignore_record_details_in_response=False,  # Set to True if only 'id' and 'score' are needed
                user_labels={}  # Add any necessary user labels here
            )
            logging.debug(f"Constructed RankRequest: {request}")

            # Step 4: Send the RankRequest to the API
            response = self.client.rank(request=request)
            logging.info("Reranking API call completed successfully.")

            # Step 5: Process the RankResponse
            reranked_records = self._process_rank_response(response, unique_records)
            logging.info(f"Reranked to top {len(reranked_records)} records.")

            return reranked_records

        except (GoogleAPICallError, NotFound, FailedPrecondition) as e:
            logging.error(f"Reranking API call failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during reranking: {e}", exc_info=True)
            raise

    def _deduplicate_records(self, records: List[Dict]) -> List[Dict]:
        """
        Removes duplicate records based on the 'id' field.
        Keeps the first occurrence of each unique ID.

        Args:
            records (List[Dict]): List of records to deduplicate.

        Returns:
            List[Dict]: Deduplicated list of records.
        """
        seen_ids = set()
        unique_records = []
        for record in records:
            record_id = record.get('id')
            if not record_id:
                logging.warning(f"Record missing 'id': {record}. Skipping.")
                continue
            if record_id not in seen_ids:
                seen_ids.add(record_id)
                unique_records.append(record)
            else:
                logging.debug(f"Duplicate record with 'id': {record_id} found. Skipping duplicate.")
        return unique_records

    def _prepare_ranking_records(self, records: List[Dict]) -> List[discoveryengine.RankingRecord]:
        """
        Converts input records to Discovery Engine's RankingRecord format.

        Args:
            records (List[Dict]): List of unique records.

        Returns:
            List[discoveryengine.RankingRecord]: List of RankingRecords.
        """
        ranking_records = []
        for record in records:
            if 'content' not in record or not record['content']:
                logging.warning(f"Record {record.get('id', 'unknown')} is missing 'content'. Skipping.")
                continue
            ranking_record = discoveryengine.RankingRecord(
                id=record['id'],
                title=record.get('title', ''),
                content=record['content'][:512]  # Limiting content to 512 characters as per API requirements
            )
            ranking_records.append(ranking_record)
        return ranking_records

    def _process_rank_response(self, response: discoveryengine.RankResponse, original_records: List[Dict]) -> List[Dict]:
        """
        Processes the RankResponse to extract and map the top-ranked records.

        Args:
            response (discoveryengine.RankResponse): The response from the reranking API.
            original_records (List[Dict]): The original list of unique records.

        Returns:
            List[Dict]: The top-ranked records with added 'rank_score'.
        """
        reranked_records = []
        try:
            for ranked_record in response.records[:self.rerank_top_n]:
                datapoint_id = ranked_record.id
                score = ranked_record.score
                # Find the original record
                original_record = next((rec for rec in original_records if rec['id'] == datapoint_id), None)
                if original_record:
                    original_record['rank_score'] = score
                    reranked_records.append(original_record)
                else:
                    logging.warning(f"Ranked record ID '{datapoint_id}' not found in original records.")
        except AttributeError as ae:
            logging.error(f"AttributeError while accessing RankResponse fields: {ae}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Error while processing RankResponse: {e}", exc_info=True)
            raise

        # Sort the records based on the score in descending order
        reranked_records.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        return reranked_records