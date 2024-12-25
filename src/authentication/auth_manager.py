# src/authentication/auth_manager.py

from google.oauth2 import service_account
from google.cloud import aiplatform
import logging
import json
import os
from typing import Dict

class AuthManager:
    def __init__(self, config: Dict):
        """
        Initializes the AuthManager with configurations from config.yaml.
        
        Args:
            config (Dict): Configuration dictionary containing necessary parameters.
        """
        try:
            # Extract GCP configuration
            gcp_config = config['gcp']
            self.project_id = gcp_config['project_id']
            self.location = gcp_config['location']
            self.credentials_path = gcp_config['credentials_path']
            
            # Optional fields
            self.bucket_name = gcp_config.get('bucket_name')
            self.embeddings_path = gcp_config.get('embeddings_path')
            self.bucket_uri = gcp_config.get('bucket_uri')
            self.qna_dataset_path = gcp_config.get('qna_dataset_path')
            
            self.credentials = None
            self.client = None  # Placeholder for any future client initialization
            
            logging.debug(f"AuthManager initialized with project_id='{self.project_id}', location='{self.location}', credentials_path='{self.credentials_path}'")
        
        except KeyError as ke:
            logging.error(f"Missing configuration key: {ke}")
            raise ValueError(f"Missing configuration key: {ke}") from ke
        except Exception as e:
            logging.error(f"Error initializing AuthManager: {e}")
            raise

    def authenticate_user(self):
        """Authenticates using the provided Service Account credentials from the JSON file."""
        try:
            if not os.path.exists(self.credentials_path):
                logging.error(f"Credentials file not found at: {self.credentials_path}")
                raise FileNotFoundError(f"Credentials file not found at: {self.credentials_path}")

            with open(self.credentials_path, 'r') as f:
                credentials_info = json.load(f)

            self.credentials = service_account.Credentials.from_service_account_info(credentials_info)
            aiplatform.init(credentials=self.credentials, project=self.project_id, location=self.location)
            logging.info("Successfully authenticated with Google Cloud using Service Account credentials.")
        
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            raise

    def get_credentials(self):
        """Returns the loaded service account credentials."""
        if not self.credentials:
            logging.error("Credentials not initialized. Call authenticate_user() first.")
            raise ValueError("Credentials not initialized. Call authenticate_user() first.")
        return self.credentials