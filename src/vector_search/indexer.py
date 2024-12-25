# src/vector_search/indexer.py

import os
from google.cloud import aiplatform
import logging
from typing import Tuple

class VectorIndexer:
    def __init__(self, config: dict):
        """
        Initializes the VectorIndexer with configurations from config.yaml.
        
        Args:
            config (dict): Configuration dictionary containing index creation and deployment parameters.
        """
        try:
            # Extract vector_search configuration
            vector_search_config = config.get('vector_search', {})
            
            # Index creation parameters
            self.index_display_name = vector_search_config.get('index_display_name')
            self.index_description = vector_search_config.get('index_description', "Index for O-RAN document embeddings")
            self.dimensions = vector_search_config.get('dimensions', 768)
            self.approximate_neighbors_count = vector_search_config.get('approximate_neighbors_count', 5)
            self.leaf_node_embedding_count = vector_search_config.get('leaf_node_embedding_count', 500)
            self.leaf_nodes_to_search_percent = vector_search_config.get('leaf_nodes_to_search_percent', 7)
            self.distance_measure_type = vector_search_config.get('distance_measure_type', 'COSINE_DISTANCE')
            self.feature_norm_type = vector_search_config.get('feature_norm_type', 'UNIT_L2_NORM')
            self.shard_size = vector_search_config.get('shard_size', 'SHARD_SIZE_SMALL')
            
            # Deployment parameters
            self.endpoint_display_name = vector_search_config.get('endpoint_display_name')
            self.machine_type = vector_search_config.get('machine_type', 'e2-standard-2')
            self.min_replica_count = vector_search_config.get('min_replica_count', 1)
            self.max_replica_count = vector_search_config.get('max_replica_count', 1)
            self.deployed_index_id = vector_search_config.get('deployed_index_id')
            
            # General parameters
            self.bucket_uri = config.get('gcp', {}).get('bucket_uri')
            if not self.bucket_uri:
                raise ValueError("bucket_uri must be specified in config.yaml under 'gcp.bucket_uri'")
            
            # Initialize AI Platform
            self.project_id = config.get('gcp', {}).get('project_id')
            self.location = config.get('gcp', {}).get('location')
            if not self.project_id or not self.location:
                raise ValueError("project_id and location must be specified in config.yaml under 'gcp'")
            
            aiplatform.init(project=self.project_id, location=self.location)
            logging.info(f"Initialized VectorIndexer with project_id='{self.project_id}', location='{self.location}', bucket_uri='{self.bucket_uri}'")
        
            # Validation of parameters
            self.validate_create_index_params()
            self.validate_deploy_index_params()
        
        except KeyError as ke:
            logging.error(f"Missing configuration key: {ke}")
            raise ValueError(f"Missing configuration key: {ke}") from ke
        except Exception as e:
            logging.error(f"Error initializing VectorIndexer: {e}", exc_info=True)
            raise

    def validate_create_index_params(self):
        """
        Validates parameters for creating an index.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        if not self.index_display_name:
            raise ValueError("index_display_name cannot be empty.")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be a positive integer.")
        if self.approximate_neighbors_count <= 0:
            raise ValueError("approximate_neighbors_count must be a positive integer.")
        if self.leaf_node_embedding_count <= 0:
            raise ValueError("leaf_node_embedding_count must be a positive integer.")
        if not (0 < self.leaf_nodes_to_search_percent <= 100):
            raise ValueError("leaf_nodes_to_search_percent must be between 1 and 100.")
        if self.distance_measure_type not in ["COSINE_DISTANCE", "EUCLIDEAN_DISTANCE"]:
            raise ValueError("distance_measure_type must be 'COSINE_DISTANCE' or 'EUCLIDEAN_DISTANCE'.")
        if self.feature_norm_type not in ["UNIT_L2_NORM", "NONE"]:
            raise ValueError("feature_norm_type must be 'UNIT_L2_NORM' or 'NONE'.")
        if self.shard_size not in ["SHARD_SIZE_SMALL", "SHARD_SIZE_MEDIUM", "SHARD_SIZE_LARGE"]:
            raise ValueError("shard_size must be 'SHARD_SIZE_SMALL', 'SHARD_SIZE_MEDIUM', or 'SHARD_SIZE_LARGE'.")
        logging.debug("Index creation parameters validated successfully.")

    def validate_deploy_index_params(self):
        """
        Validates parameters for deploying an index.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        if not self.endpoint_display_name:
            raise ValueError("endpoint_display_name cannot be empty.")
        if not isinstance(self.machine_type, str):
            raise ValueError("machine_type must be a string.")
        if not isinstance(self.min_replica_count, int) or self.min_replica_count <= 0:
            raise ValueError("min_replica_count must be a positive integer.")
        if not isinstance(self.max_replica_count, int) or self.max_replica_count <= 0:
            raise ValueError("max_replica_count must be a positive integer.")
        if self.min_replica_count > self.max_replica_count:
            raise ValueError("min_replica_count cannot be greater than max_replica_count.")
        logging.debug("Index deployment parameters validated successfully.")

    def create_index(self) -> aiplatform.MatchingEngineIndex:
        """
        Creates a Matching Engine index using the initialized parameters.
        
        Returns:
            aiplatform.MatchingEngineIndex: The created index.
        """
        logging.info(f"Creating Matching Engine index with display_name='{self.index_display_name}'")
        
        try:
            index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=self.index_display_name,
                description=self.index_description,
                dimensions=self.dimensions,
                approximate_neighbors_count=self.approximate_neighbors_count,
                leaf_node_embedding_count=self.leaf_node_embedding_count,
                leaf_nodes_to_search_percent=self.leaf_nodes_to_search_percent,
                distance_measure_type=self.distance_measure_type,
                feature_norm_type=self.feature_norm_type,
                contents_delta_uri=self.bucket_uri,
                shard_size=self.shard_size
            )
            index.wait()
            logging.info(f"Index created: {index.resource_name}")
            return index
        except Exception as e:
            logging.error(f"Failed to create index '{self.index_display_name}': {e}", exc_info=True)
            raise

    def create_index_endpoint(self) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Creates a new Matching Engine index endpoint.
        
        Returns:
            aiplatform.MatchingEngineIndexEndpoint: The created index endpoint.
        """
        logging.info(f"Creating Matching Engine index endpoint with display_name='{self.endpoint_display_name}'")
        
        try:
            endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=self.endpoint_display_name,
                public_endpoint_enabled=True,
                description="Endpoint for O-RAN RAG vector search"
            )
            endpoint.wait()
            logging.info(f"Index endpoint created: {endpoint.resource_name}")
            return endpoint
        except Exception as e:
            logging.error(f"Failed to create index endpoint '{self.endpoint_display_name}': {e}", exc_info=True)
            raise
        
    def deploy_index(self, index: aiplatform.MatchingEngineIndex) -> Tuple[aiplatform.MatchingEngineIndexEndpoint, str]:
        """
        Deploys the created index to an endpoint and retrieves the deployed index ID.
        
        Args:
            index (aiplatform.MatchingEngineIndex): The index to deploy.
        
        Returns:
            Tuple[aiplatform.MatchingEngineIndexEndpoint, str]: The deployed endpoint and the deployed index ID.
        """
        logging.info(f"Deploying index '{index.display_name}' to endpoint '{self.endpoint_display_name}'")
        
        try:
            # Check if the endpoint already exists
            existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
            logging.debug(f"Retrieved {len(existing_endpoints)} existing endpoints.")
            
            target_endpoint = None
            for ep in existing_endpoints:
                if ep.display_name == self.endpoint_display_name:
                    target_endpoint = ep
                    logging.info(f"Found existing endpoint: {ep.resource_name}")
                    break
            
            if not target_endpoint:
                logging.info(f"Endpoint '{self.endpoint_display_name}' not found. Creating a new one.")
                target_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                    display_name=self.endpoint_display_name,
                    public_endpoint_enabled=True,
                    description="Endpoint for O-RAN RAG vector search"
                )
                target_endpoint.wait()
                logging.info(f"Created new endpoint: {target_endpoint.resource_name}")
            else:
                logging.info(f"Using existing endpoint: {target_endpoint.resource_name}")
            
            # Deploy the index to the endpoint
            logging.info("Starting index deployment...")
            deployed_index = target_endpoint.deploy_index(
                index=index,
                deployed_index_id=self.deployed_index_id,
                machine_type=self.machine_type,
                min_replica_count=self.min_replica_count,
                max_replica_count=self.max_replica_count
            )
            deployed_index.wait()
            logging.info(f"Successfully deployed index to endpoint: {target_endpoint.resource_name}")
            
            # Retrieve the deployed index ID
            logging.info("Retrieving deployed index ID.")
            deployed_indexes = target_endpoint.deployed_indexes
            deployed_index_id_retrieved = next((dep.id for dep in deployed_indexes if dep.index == index.resource_name), None)
            if deployed_index_id_retrieved:
                logging.info(f"Deployed Index ID: {deployed_index_id_retrieved}")
            else:
                logging.error(f"Deployed index ID not found for index '{index.display_name}'")
                raise ValueError(f"Deployed index ID not found for index '{index.display_name}'")
            
            return target_endpoint, deployed_index_id_retrieved
        
        except Exception as e:
            logging.error(f"Failed to deploy index '{index.display_name}' to endpoint '{self.endpoint_display_name}': {e}", exc_info=True)
            raise

    def deploy_index_existing(self) -> Tuple[aiplatform.MatchingEngineIndexEndpoint, str]:
        """
        Deploys an existing index to an endpoint.
        
        Returns:
            Tuple[aiplatform.MatchingEngineIndexEndpoint, str]: The deployed endpoint and the deployed index ID.
        """
        logging.info(f"Deploying existing index to endpoint '{self.endpoint_display_name}'")
        
        try:
            # Check if the endpoint already exists
            existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
            logging.debug(f"Retrieved {len(existing_endpoints)} existing endpoints.")
            
            target_endpoint = None
            for ep in existing_endpoints:
                if ep.display_name == self.endpoint_display_name:
                    target_endpoint = ep
                    logging.info(f"Found existing endpoint: {ep.resource_name}")
                    break
            
            if not target_endpoint:
                logging.info(f"Endpoint '{self.endpoint_display_name}' not found. Creating a new one.")
                target_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                    display_name=self.endpoint_display_name,
                    public_endpoint_enabled=True,
                    description="Endpoint for O-RAN RAG vector search"
                )
                target_endpoint.wait()
                logging.info(f"Created new endpoint: {target_endpoint.resource_name}")
            else:
                logging.info(f"Using existing endpoint: {target_endpoint.resource_name}")
            
            # Retrieve the existing index by display_name
            logging.info(f"Retrieving existing index with display_name='{self.index_display_name}'")
            existing_indexes = aiplatform.MatchingEngineIndex.list()
            existing_index = next((idx for idx in existing_indexes if idx.display_name == self.index_display_name), None)
            if not existing_index:
                logging.error(f"No MatchingEngineIndex found with display_name='{self.index_display_name}'")
                raise ValueError(f"No MatchingEngineIndex found with display_name='{self.index_display_name}'")
            logging.info(f"Found existing index: {existing_index.resource_name}")
            
            # Deploy the existing index to the endpoint
            logging.info("Starting deployment of existing index...")
            deployed_index = target_endpoint.deploy_index(
                index=existing_index,
                deployed_index_id=self.deployed_index_id,
                machine_type=self.machine_type,
                min_replica_count=self.min_replica_count,
                max_replica_count=self.max_replica_count
            )
            deployed_index.wait()
            logging.info(f"Successfully deployed existing index to endpoint: {target_endpoint.resource_name}")
            
            # Retrieve the deployed index ID
            logging.info("Retrieving deployed index ID.")
            deployed_indexes = target_endpoint.deployed_indexes
            deployed_index_id_retrieved = next((dep.id for dep in deployed_indexes if dep.index == existing_index.resource_name), None)
            if deployed_index_id_retrieved:
                logging.info(f"Deployed Index ID: {deployed_index_id_retrieved}")
            else:
                logging.error(f"Deployed index ID not found for index '{self.index_display_name}'")
                raise ValueError(f"Deployed index ID not found for index '{self.index_display_name}'")
            
            return target_endpoint, deployed_index_id_retrieved
        
        except Exception as e:
            logging.error(f"Failed to deploy existing index '{self.index_display_name}' to endpoint '{self.endpoint_display_name}': {e}", exc_info=True)
            raise