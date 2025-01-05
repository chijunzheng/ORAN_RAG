# src/main.py

import os
import argparse
import logging
import sys
import json
from google.cloud import storage

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import load_config, validate_config
from src.utils.logger import setup_logging
from src.utils.helpers import ensure_directory
from src.authentication.auth_manager import AuthManager
from src.data_processing.converters import DocumentConverter
from src.data_processing.loaders import PDFLoader
from src.data_processing.text_formatter import TextFormatter
from src.data_processing.document_chunker import DocumentChunker
from src.embeddings.embedder import Embedder
from src.vector_search.indexer import VectorIndexer
from src.vector_search.searcher import VectorSearcher
from src.chatbot.chatbot import Chatbot
from src.evaluation.evaluator import Evaluator
from src.vector_search.reranker import Reranker

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="O-RAN RAG Project",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Usage Examples:
          1. Run the full pipeline with evaluation:
             python src/main.py --config configs/config.yaml --evaluation on

          2. Run the pipeline without evaluation and skip preprocessing:
             python src/main.py --config configs/config.yaml --evaluation off --skip-preprocessing

          3. Display the help message:
             python src/main.py --help

        Note:
          - The --skip-preprocessing flag skips document conversion, loading PDFs, text formatting, chunking, and embedding.
        """
    )

    # Create argument groups for better organization
    config_group = parser.add_argument_group('Configuration')
    pipeline_group = parser.add_argument_group('Pipeline Control')
    evaluation_group = parser.add_argument_group('Evaluation')

    # Configuration arguments
    config_group.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to the configuration file. (default: configs/config.yaml)'
    )

    # Evaluation arguments
    evaluation_group.add_argument(
        '--evaluation',
        type=str,
        choices=['on', 'off'],
        default='off',
        help='Toggle evaluation mode: "on" to perform evaluation, "off" to skip. (default: off)'
    )

    # Pipeline Control arguments
    pipeline_group.add_argument(
        '--skip-preprocessing',  
        action='store_true',
        help='Skip all preprocessing stages: document conversion, loading PDFs, text formatting, chunking, and embedding.'
    )

    pipeline_group.add_argument(
        '--skip-create-index',
        action='store_true',
        help='Skip the index creation stage.'
    )
    
    pipeline_group.add_argument(
        '--skip-deploy-index',
        action='store_true',
        help='Skip the index deployment stage.'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Initialize logging early to capture errors during config loading
    setup_logging(log_level=logging.INFO, log_file=None)  
    logging.info("Starting O-RAN RAG Project")

    # Load configurations (assuming YAML format)
    try:
        config = load_config(args.config)
        validate_config(config)
    except Exception as e:
        # Initialize logging early to capture errors during config loading
        setup_logging(log_level=logging.ERROR, log_file=None)
        logging.error(f"Failed to load or validate config: {e}")
        sys.exit(1)

    # Setup logging after config is loaded
    # Define log file path from config.yaml
    log_file = config.get('logging', {}).get('log_file')
    if log_file:
        setup_logging(log_level=logging.INFO, log_file=log_file)
        logging.info(f"Logging to file: {log_file}")    

    # Extract necessary parameters from config.yaml
    gcp_config = config['gcp']
    paths_config = config['paths']
    vector_search_config = config['vector_search']
    chunking_config = config['chunking']
    generation_config = config['generation']
    logging_config = config['logging']
    evaluation_config = config['evaluation']
    ranking_config = config['ranking']

    # Validate chunk_size and chunk_overlap
    if chunking_config['chunk_size'] <= 0:
        logging.error("chunk_size must be a positive integer.")
        sys.exit(1)
    if chunking_config['chunk_overlap'] < 0:
        logging.error("chunk_overlap cannot be negative.")
        sys.exit(1)
    if chunking_config['chunk_overlap'] >= chunking_config['chunk_size']:
        logging.error("chunk_overlap must be smaller than chunk_size.")
        sys.exit(1)

    # Ensure necessary directories exist
    ensure_directory(config['paths']['embeddings_save_path'])
    ensure_directory(os.path.dirname(config['logging']['log_file']) or 'log')  # Ensure log directory exists
    ensure_directory(os.path.dirname(config['evaluation']['excel_file_path']) or 'Evaluation')
    ensure_directory(os.path.dirname(config['evaluation']['plot_save_path']) or 'Evaluation')
    

    # Initialize Authentication with Service Account
    try:
        auth_manager = AuthManager(config=config)  # Pass config dict
        auth_manager.authenticate_user()
        logging.info("Authentication successful.")
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        sys.exit(1)

    # Assign flags to variables for clarity
    skip_preprocessing = args.skip_preprocessing
    skip_create_index = args.skip_create_index
    skip_deploy_index = args.skip_deploy_index

    # Initialize VectorIndexer regardless of flags (required for indexing operations)
    try:
        indexer = VectorIndexer(config=config)
    except Exception as e:
        logging.error(f"Failed to initialize VectorIndexer: {e}")
        sys.exit(1)

    # Initialize Reranker
    try:
        reranker = Reranker(
            project_id=gcp_config['project_id'],
            location=gcp_config['location'],
            ranking_config=ranking_config['ranking_config'],
            credentials=auth_manager.credentials,
            model=ranking_config['model'],
            rerank_top_n=ranking_config['rerank_top_n']
        )
    except Exception as e:
        logging.error(f"Failed to initialize Reranker: {e}")
        sys.exit(1)

    # 1. Preprocessing Stage
    if not skip_preprocessing:
        logging.info("Starting preprocessing stages.")
        # Perform Preprocessing
        # converter = DocumentConverter(directory_path=config['paths']['documents'])
        # try:
        #     converter.convert_docx_to_pdf()
        #     logging.info("Document conversion completed.")
        # except Exception as e:
        #     logging.error(f"Document conversion failed: {e}")
        #     sys.exit(1)

        # Load PDFs
        loader = PDFLoader(pdf_directory=config['paths']['documents'])
        try:
            documents = loader.load_multiple_pdfs()
            logging.info(f"Loaded {len(documents)} PDF documents.")
        except Exception as e:
            logging.error(f"Failed to load PDFs: {e}")
            sys.exit(1)

        # Text Formatting
        formatter = TextFormatter()
        try:
            all_cleaned_documents = formatter.format_documents(documents)
            logging.info(f"Formatted {len(all_cleaned_documents)} documents.")
        except Exception as e:
            logging.error(f"Text formatting failed: {e}")
            sys.exit(1)

        # Split into Chunks
        chunker = DocumentChunker(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap'],
            separators=chunking_config['separators'],
            gcs_bucket_name=gcp_config['bucket_name'],
            gcs_embeddings_path=gcp_config['embeddings_path'],
            credentials=auth_manager.credentials,
            min_char_count=chunking_config['min_char_count']
        )
        try:
            chunks_with_ids = chunker.split_documents(all_cleaned_documents)
            logging.info(f"Split documents into {len(chunks_with_ids)} chunks.")
        except Exception as e:
            logging.error(f"Document chunking failed: {e}")
            sys.exit(1)

        # Define the chunks file path from config
        chunks_file = os.path.join(config['paths']['embeddings_save_path'], 'chunks.json')
        
        # Save chunks to JSON Lines file
        try:
            chunker.save_chunks_to_json(chunks_with_ids, file_path=chunks_file)
            logging.info(f"Saved chunks to {chunks_file}")
        except Exception as e:
            logging.error(f"Failed to save chunks: {e}")
            sys.exit(1)
        
        # Upload chunks to GCS
        try:
            chunker.upload_to_gcs(file_path=chunks_file, overwrite=True)
            logging.info("Uploaded chunks to GCS.")
        except Exception as e:
            logging.error(f"Failed to upload chunks to GCS: {e}")
            sys.exit(1)

        # Initialize Embedder
        embedder = Embedder(
            project_id=gcp_config['project_id'],
            location=gcp_config['location'],
            bucket_name=gcp_config['bucket_name'],
            embeddings_path=gcp_config['embeddings_path'],
            credentials=auth_manager.credentials
        )
        # Generate and Upload Embeddings
        try:
            embeddings_file = os.path.join(config['paths']['embeddings_save_path'], 'embeddings.json')
            embedder.generate_and_store_embeddings(chunks_with_ids, local_jsonl_path=embeddings_file)
            logging.info(f"Embeddings generated and saved to {embeddings_file}")
        except Exception as e:
            logging.error(f"Embeddings generation failed: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping preprocessing stages: document conversion, loading PDFs, text formatting, chunking, and embedding.")

    # 2. Index Creation Stage
    index = None
    if not skip_create_index:
        logging.info("Starting index creation stage.")
        try: 
            index = indexer.create_index()
            logging.info(f"Index '{index.display_name}' created successfully.")
        except Exception as e:
            logging.error(f"Index creation failed: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping index creation stage.")

    # 3. Index Deployment Stage
    if not skip_deploy_index:
        logging.info("Starting index deployment stage.")
        try:
            # Deploy the newly created index or an existing one
            endpoint, deployed_index_id = indexer.deploy_index(index=index)
            logging.info(f"Index deployed with ID: {deployed_index_id}")
            # Update config with deployed_index_id for downstream components
            config['vector_search']['deployed_index_id'] = deployed_index_id
        except Exception as e:
            logging.error(f"Index deployment failed: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping index deployment stage.")

    # 4. Initialize VectorSearcher (needed for both Chatbot and Evaluator)
    try:
        vector_searcher = VectorSearcher(
            project_id=gcp_config['project_id'],
            location=gcp_config['location'],
            bucket_name=gcp_config['bucket_name'],
            embeddings_path=gcp_config['embeddings_path'],
            bucket_uri=gcp_config['bucket_uri'],
            credentials=auth_manager.credentials
        )
    except Exception as e:
        logging.error(f"Failed to initialize VectorSearcher: {e}")
        sys.exit(1)

    # 5. Chatbot or Evaluation
    if args.evaluation == 'on':
        logging.info("Evaluation mode is ON. Skipping chatbot interaction.")
    else:
        logging.info("Starting chatbot interaction stage.")
        try:
            chatbot = Chatbot(
                project_id=gcp_config['project_id'],
                location=gcp_config['location'],
                bucket_name=gcp_config['bucket_name'],
                embeddings_path=gcp_config['embeddings_path'],
                bucket_uri=gcp_config['bucket_uri'],
                index_endpoint_display_name=vector_search_config['endpoint_display_name'],
                deployed_index_id=vector_search_config.get('deployed_index_id'),
                generation_temperature=generation_config.get('temperature'),
                generation_top_p=generation_config.get('top_p'),
                generation_max_output_tokens=generation_config.get('max_output_tokens'),
                vector_searcher=vector_searcher,
                reranker=reranker,
                credentials=auth_manager.credentials,
                num_neighbors=vector_search_config.get('num_neighbors')
            )
            logging.info("Starting chatbot interaction loop.")
            chatbot.chat_loop()
        except Exception as e:
            logging.error(f"Chatbot encountered an error: {e}")
            sys.exit(1)

    # 6. Conditional Evaluation
    if args.evaluation == 'on':
        logging.info("Starting evaluation stage.")
        try:
            # Initialize Evaluator
            evaluator = Evaluator(
                project_id=gcp_config['project_id'],
                location=gcp_config['location'],
                bucket_name=gcp_config['bucket_name'],
                index_endpoint_display_name=vector_search_config['endpoint_display_name'],
                deployed_index_id=vector_search_config['deployed_index_id'],
                embeddings_path=gcp_config['embeddings_path'],
                qna_dataset_path=config['gcp']['qna_dataset_path'],
                generation_config=generation_config,  
                vector_searcher=vector_searcher, 
                credentials=auth_manager.credentials,
                num_neighbors=vector_search_config['num_neighbors'],
                reranker=reranker
            )
            qna_dataset = evaluator.load_qna_dataset_from_gcs()
            num_questions = evaluation_config['num_questions']
            
            # Dynamically generate Excel filename
            base_excel_path = config['evaluation']['excel_file_path']
            excel_dir, excel_filename = os.path.split(base_excel_path)
            excel_name, excel_ext = os.path.splitext(excel_filename)
            dynamic_excel_filename = f"{excel_name}_{num_questions}{excel_ext}"
            dynamic_excel_path = os.path.join(excel_dir, dynamic_excel_filename)

            rag_accuracy, gemini_accuracy = evaluator.evaluate_models_parallel(
                qna_dataset=qna_dataset,
                num_questions=num_questions,
                excel_file_path=dynamic_excel_path,
                max_workers=evaluation_config['max_workers']
            )
            logging.info(f"Evaluation completed. RAG Accuracy: {rag_accuracy}%, Gemini Accuracy: {gemini_accuracy}%")

            # Dynamically generate plot filename
            base_plot_path = evaluation_config['plot_save_path']
            if base_plot_path:
                plot_dir, plot_filename = os.path.split(base_plot_path)
                plot_name, plot_ext = os.path.splitext(plot_filename)
                dynamic_plot_filename = f"{plot_name}_{num_questions}{plot_ext}"
                dynamic_plot_path = os.path.join(plot_dir, dynamic_plot_filename)
            else:
                dynamic_plot_path = None  # If not specified, no plot will be saved

            # Call the visualize_accuracies method from Evaluator
            evaluator.visualize_accuracies(rag_accuracy, gemini_accuracy, save_path=dynamic_plot_path)
        
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
    else:
        logging.info("Evaluation mode is OFF. Skipping evaluation.")

    # Exit after all stages are completed
    logging.info("O-RAN RAG Project completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()