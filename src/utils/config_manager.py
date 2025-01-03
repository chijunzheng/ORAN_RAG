# src/utils/config_manager.py

import yaml
import logging
from typing import Dict, Any

class QuotedStringDumper(yaml.SafeDumper):
    """
    Custom YAML dumper that quotes strings containing spaces, colons, or slashes.
    """
    def represent_str(self, data):
        if any(char in data for char in [' ', ':', '/']):
            return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')
        return self.represent_scalar('tag:yaml.org,2002:str', data)

# Register the custom representer
QuotedStringDumper.add_representer(str, QuotedStringDumper.represent_str)

class ConfigManager:
    def __init__(self, config_path: str):
        """
        Initializes the ConfigManager with the path to the configuration file.
        
        Args:
            config_path (str): Path to the config.yaml file.
        """
        self.config_path = config_path
        self.config = self.load_config()
        logging.debug(f"ConfigManager initialized with config: {self.config}")

    def load_config(self) -> Dict[str, Any]:
        """
        Loads the YAML configuration file.
        
        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logging.debug(f"Loaded configuration from {self.config_path}: {config}")
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            raise

    def save_config(self):
        """
        Saves the current configuration dictionary back to the YAML file.
        Ensures that strings with spaces are quoted and lists are preserved.
        Prevents line wrapping for long strings.
        """
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(
                    self.config,
                    file,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                    explicit_start=True,
                    explicit_end=True,
                    Dumper=QuotedStringDumper,  # Correct usage with yaml.dump
                    width=1000  # Set a high width to prevent line wrapping
                )
            logging.debug(f"Saved configuration to {self.config_path}: {self.config}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            raise

    def update_config(self, updates: Dict[str, Any]):
        """
        Updates the configuration with the provided key-value pairs.
        Supports nested keys using dot notation.
        
        Args:
            updates (Dict[str, Any]): Dictionary containing configuration updates.
        """
        for key, value in updates.items():
            keys = key.split('.')
            d = self.config
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
            logging.debug(f"Updated config key '{key}' with value '{value}'")
        self.save_config()

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the current configuration dictionary.
        
        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        return self.config