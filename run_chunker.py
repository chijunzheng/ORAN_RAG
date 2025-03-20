#!/usr/bin/env python
"""
Standalone script to run just the contextual chunking part of the O-RAN RAG pipeline.
This script avoids importing problematic libraries like vertexai or google.cloud.firestore
that might be causing timeout errors.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

# Add the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run contextual chunking for O-RAN RAG")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--input-dir', type=str, help='Directory containing cleaned documents (overrides config)')
    parser.add_argument('--output-dir', type=str, help='Directory to save chunked documents (overrides config)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_directory(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

def main():
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Extract necessary configuration
    chunking_config = config['chunking']
    
    # Determine input and output directories
    input_dir = args.input_dir if args.input_dir else config['paths']['documents']
    output_dir = args.output_dir if args.output_dir else config['paths']['embeddings_save_path']
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Import necessary modules
    try:
        from src.data_processing.contextual_chunker import ContextualChunker
        from src.data_processing.loaders import PDFLoader
        from src.data_processing.text_formatter import TextFormatter
        logging.info("Successfully imported required modules")
    except ImportError as e:
        logging.error(f"Import error: {e}")
        sys.exit(1)
    
    # Process files
    try:
        # Load PDFs
        loader = PDFLoader(pdf_directory=input_dir)
        documents = loader.load_multiple_pdfs()
        logging.info(f"Loaded {len(documents)} PDF documents from {input_dir}")
        
        # Format text
        formatter = TextFormatter()
        cleaned_docs = formatter.format_documents(documents)
        logging.info(f"Formatted {len(cleaned_docs)} documents")
        
        # Initialize contextual chunker
        chunker = ContextualChunker(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap'],
            separators=chunking_config['separators'],
            min_char_count=chunking_config['min_char_count']
        )
        logging.info("Initialized ContextualChunker")
        
        # Generate chunks with context
        chunks = chunker.split_documents(cleaned_docs)
        logging.info(f"Created {len(chunks)} chunks with context")
        
        # Save chunks to file
        output_file = os.path.join(output_dir, 'contextual_chunks.json')
        chunker.save_chunks_to_json(chunks, file_path=output_file)
        logging.info(f"Saved chunks to {output_file}")
        
        logging.info("Contextual chunking completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 