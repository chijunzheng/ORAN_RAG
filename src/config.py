# src/config.py

import yaml
import logging
from typing import Dict

def load_config(config_path: str) -> Dict:
    """
    Loads the YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        Dict: Parsed configuration dictionary.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.debug(f"Configuration loaded from {config_path}.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise

def validate_config(config: Dict):
    """
    Validates the presence of required fields in the configuration.
    
    Args:
        config (Dict): Parsed configuration dictionary.
    
    Raises:
        ValueError: If any required fields are missing.
    """
    required_gcp_fields = [
        'project_id', 'location', 'bucket_name', 'embeddings_path',
        'bucket_uri', 'qna_dataset_path', 'credentials_path'
    ]
    for field in required_gcp_fields:
        if field not in config.get('gcp', {}):
            raise ValueError(f"Missing required GCP configuration field: '{field}'")
    
    # Validate vector_search parameters
    required_vector_search_fields = [
        'index_display_name', 'endpoint_display_name',
        'index_description', 'dimensions',
        'approximate_neighbors_count', 'leaf_node_embedding_count',
        'leaf_nodes_to_search_percent', 'distance_measure_type',
        'feature_norm_type', 'shard_size',
        'machine_type', 'min_replica_count', 'max_replica_count'
    ]
    for field in required_vector_search_fields:
        if field not in config.get('vector_search', {}):
            raise ValueError(f"Missing required vector_search configuration field: '{field}'")
    
    # Validate chunking parameters
    required_chunking_fields = ['chunk_size', 'chunk_overlap']
    for field in required_chunking_fields:
        if field not in config.get('chunking', {}):
            raise ValueError(f"Missing required chunking configuration field: '{field}'")
    
    # Validate generation parameters
    required_generation_fields = ['temperature', 'top_p', 'max_output_tokens']
    for field in required_generation_fields:
        if field not in config.get('generation', {}):
            raise ValueError(f"Missing required generation configuration field: '{field}'")
    
    logging.info("Configuration validation successful.")