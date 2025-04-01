# src/main.py

import os
import argparse
import logging
import sys
import json
import uuid
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
from src.data_processing.contextual_chunker import ContextualChunker
from src.embeddings.embedder import Embedder
from src.vector_search.indexer import VectorIndexer
from src.vector_search.searcher import VectorSearcher
# Only import these when needed
# from src.chatbot.chatbot import Chatbot
# from src.evaluation.evaluator import Evaluator
# from src.vector_search.reranker import Reranker


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
          - The --skip-preprocessing flag skips document conversion, loading PDFs, and text formatting.
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
        help='Skip initial preprocessing stages: document conversion, loading PDFs, and text formatting.'
    )

    pipeline_group.add_argument(
        '--skip-chunking',
        action='store_true',
        help='Skip the document chunking stage.'
    )

    pipeline_group.add_argument(
        '--skip-embedding',
        action='store_true',
        help='Skip the embedding generation stage.'
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

    pipeline_group.add_argument(
        '--skip-chatbot',
        action='store_true',
        help='Skip chatbot initialization (use for preprocessing only).'
    )

    pipeline_group.add_argument(
        '--chunks-only',
        action='store_true',
        help='Only perform document conversion, text formatting, and chunking operations (skip embedding, indexing, chatbot, and evaluation).'
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

    # Setup logging after config is loaded - always ensure console output
    # Define log file path from config.yaml
    log_file = config.get('logging', {}).get('log_file')
    # Always keep logging to console even when writing to file
    setup_logging(log_level=logging.INFO, log_file=log_file)
    if log_file:
        logging.info(f"Logging to file: {log_file} and console")
    else:
        logging.info("Logging to console only")

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
    ensure_directory(config['paths']['documents'])
    ensure_directory(os.path.dirname(config['logging']['log_file']))
    ensure_directory(os.path.dirname(config['evaluation']['excel_file_path']))
    ensure_directory(os.path.dirname(config['evaluation']['plot_save_path']))

    # Print debug info about directories
    logging.info(f"Embeddings will be saved to: {config['paths']['embeddings_save_path']}")
    logging.info(f"Logs will be written to: {config['logging']['log_file']}")

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
    skip_chunking = args.skip_chunking
    skip_embedding = args.skip_embedding
    skip_create_index = args.skip_create_index
    skip_deploy_index = args.skip_deploy_index
    skip_chatbot = args.skip_chatbot
    chunks_only = args.chunks_only
    
    # If chunks_only flag is set, use it to override other flags
    if chunks_only:
        skip_create_index = True
        skip_deploy_index = True
        skip_chatbot = True
        # No need to override evaluation as it's controlled by the evaluation flag

    # Initialize VectorIndexer regardless of flags (required for indexing operations)
    try:
        indexer = VectorIndexer(config=config)
    except Exception as e:
        logging.error(f"Failed to initialize VectorIndexer: {e}")
        sys.exit(1)

    # Initialize Reranker only if not in chunks-only mode or skip-chatbot mode
    if not chunks_only and not skip_chatbot:
        try:
            # Import Reranker only when needed
            from src.vector_search.reranker import Reranker
            
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
    else:
        logging.info("Skipping Reranker initialization as requested.")
        # Create a dummy reranker for later use
        reranker = None

    # 1. Preprocessing Stage (ORAN + YANG)
    if not skip_preprocessing:
        logging.info("Starting document preprocessing stages.")
        # Perform ORAN PDF-based Preprocessing
        converter = DocumentConverter(directory_path=config['paths']['documents'])
        try:
            converter.convert_docx_to_pdf()
            logging.info("Document conversion completed.")
        except Exception as e:
            logging.error(f"Document conversion failed: {e}")
            sys.exit(1)

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
    else:
        logging.info("Skipping document preprocessing stages.")
        all_cleaned_documents = []  # Initialize as empty if preprocessing is skipped
    
    # 2. Document Chunking Stage
    if not skip_chunking and not chunks_only:
        logging.info("Starting document chunking stage.")
        # Replace DocumentChunker with ContextualChunker
        chunker = ContextualChunker(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap'],
            separators=chunking_config['separators'],
            gcs_bucket_name=gcp_config['bucket_name'],
            gcs_embeddings_path=gcp_config['embeddings_path'],
            credentials=auth_manager.credentials,
            min_char_count=chunking_config['min_char_count']
        )
        
        # Only try to split documents if we have documents from the preprocessing step
        if skip_preprocessing:
            logging.info("Loading existing documents for chunking...")
            # TODO: If needed, add logic to load existing documents here
            # For now, assume we don't have documents if preprocessing was skipped
            # and chunking needs to be done using existing data
            
        try:
            if not all_cleaned_documents and skip_preprocessing:
                # Load existing chunks if available
                chunks_file = os.path.join(paths_config['embeddings_save_path'], 'chunks.json')
                if os.path.exists(chunks_file):
                    logging.info(f"Loading existing chunks from {chunks_file}")
                    with open(chunks_file, 'r') as f:
                        oran_chunks = json.load(f)
                    logging.info(f"Loaded {len(oran_chunks)} existing chunks.")
                else:
                    logging.error("No documents to chunk and no existing chunks found.")
                    sys.exit(1)
            else:
                oran_chunks = chunker.split_documents(all_cleaned_documents)
                logging.info(f"Split documents into {len(oran_chunks)} chunks with context.")
                
                # Save chunks to JSON file
                chunks_file = os.path.join(paths_config['embeddings_save_path'], 'chunks.json')
                try:
                    chunker.save_chunks_to_json(oran_chunks, file_path=chunks_file)
                    logging.info(f"Saved chunks to {chunks_file}")
                except Exception as e:
                    logging.error(f"Failed to save chunks: {e}")
                    sys.exit(1)
                
                # Upload chunks file to GCS
                try:
                    chunker.upload_to_gcs(file_path=chunks_file, overwrite=True)
                    logging.info("Uploaded chunks to GCS.")
                except Exception as e:
                    logging.error(f"Failed to upload chunks to GCS: {e}")
                    sys.exit(1)
                
        except Exception as e:
            logging.error(f"Document chunking failed: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping document chunking stage.")
        # Load existing chunks if available and if needed for embedding
        if not skip_embedding:
            # First try to load chunks.json
            chunks_json_file = os.path.join(paths_config['embeddings_save_path'], 'chunks.json')   
            if os.path.exists(chunks_json_file):
                logging.info(f"Loading existing chunks from {chunks_json_file} for embedding")
                try:
                    with open(chunks_json_file, 'r') as f:
                        oran_chunks = json.load(f)
                    logging.info(f"Loaded {len(oran_chunks)} existing chunks.")
                except Exception as e:
                    logging.error(f"Failed to load JSON chunks file: {e}")
                    logging.info("Trying JSONL format instead...")
                    
            else:
                logging.error("Cannot proceed with embedding: No chunks file found.")
                sys.exit(1)
        else:
            oran_chunks = []  # Initialize as empty if both chunking and embedding are skipped
    
    # 3. Embedding Generation Stage
    if not skip_embedding and not chunks_only:
        logging.info("Starting embedding generation stage.")
        try:
            embedder = Embedder(
                project_id=gcp_config['project_id'],
                location=gcp_config['location'],
                bucket_name=gcp_config['bucket_name'],
                embeddings_path=gcp_config['embeddings_path'],
                credentials=auth_manager.get_credentials()
            )
            embeddings_file = os.path.join(paths_config['embeddings_save_path'], 'embeddings.json')
            embedder.generate_and_store_embeddings(oran_chunks, local_json_path=embeddings_file)
            logging.info(f"Embeddings generated and saved to {embeddings_file}")
        except Exception as e:
            logging.error(f"Embeddings generation failed: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping embedding generation stage.")
        
    # If chunks-only flag is set, exit after processing chunks
    if chunks_only:
        logging.info("Operating in chunks-only mode. Exiting after chunk processing.")
        sys.exit(0)

    # 4. Index Creation Stage
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

    # 5. Index Deployment Stage
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

    # 6. Initialize VectorSearcher (needed for both Chatbot and Evaluator)
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

    # 7. Chatbot or Evaluation
    if args.evaluation == 'on':
        logging.info("Evaluation mode is ON. Skipping chatbot interaction.")
    else:
        if skip_chatbot:
            logging.info("Skipping chatbot initialization as requested.")
        else:
            logging.info("Starting chatbot interaction stage.")
            try:
                # Import Chatbot only when needed
                from src.chatbot.chatbot import Chatbot
                
                # Check if reranker is available
                if reranker is None:
                    logging.warning("Reranker is not available. Initializing Chatbot without reranking capability.")
                    
                chatbot = Chatbot(
                    project_id=gcp_config['project_id'],
                    location=gcp_config['location'],
                    bucket_name=gcp_config['bucket_name'],
                    embeddings_path=gcp_config['embeddings_path'],
                    bucket_uri=gcp_config['bucket_uri'],
                    index_endpoint_display_name=vector_search_config['endpoint_display_name'],
                    deployed_index_id=vector_search_config['deployed_index_id'],
                    generation_temperature=generation_config['temperature'],
                    generation_top_p=generation_config['top_p'],
                    generation_max_output_tokens=generation_config['max_output_tokens'],
                    vector_searcher=vector_searcher,
                    reranker=reranker,
                    credentials=auth_manager.credentials,
                    num_neighbors=vector_search_config['num_neighbors']
                )
                logging.info("Chatbot initialized successfully")
                
            except Exception as e:
                logging.error(f"Chatbot encountered an error: {e}")
                sys.exit(1)

    # 8. Conditional Evaluation
    if args.evaluation == 'on':
        logging.info("Starting evaluation stage.")
        try:
            # Initialize Evaluator
            from src.evaluation.evaluator import Evaluator
            
            # Check if reranker is available
            if reranker is None:
                logging.warning("Reranker is not available. Initializing Evaluator without reranking capability.")
            
            evaluator = Evaluator(
                project_id=gcp_config['project_id'],
                location=gcp_config['location'],
                bucket_name=gcp_config['bucket_name'],
                index_endpoint_display_name=vector_search_config['endpoint_display_name'],
                deployed_index_id=vector_search_config['deployed_index_id'],
                embeddings_path=gcp_config['embeddings_path'],
                qna_dataset_path=gcp_config['qna_dataset_path'],
                qna_dataset_local_path=evaluation_config['qna_dataset_local_path'],
                generation_config=generation_config,  
                vector_searcher=vector_searcher, 
                credentials=auth_manager.credentials,
                num_neighbors=vector_search_config['num_neighbors'],
                reranker=reranker
            )
            
            # Load dataset from local path (falls back to GCS if needed)
            qna_dataset = evaluator.load_qna_dataset()
            num_questions = evaluation_config['num_questions']
            
            # Dynamically generate Excel filename
            base_excel_path = config['evaluation']['excel_file_path']
            excel_dir, excel_filename = os.path.split(base_excel_path)
            excel_name, excel_ext = os.path.splitext(excel_filename)
            dynamic_excel_filename = f"{excel_name}_{num_questions}{excel_ext}"
            dynamic_excel_path = os.path.join(excel_dir, dynamic_excel_filename)

            chain_of_rag_accuracy, gemini_accuracy = evaluator.evaluate_models_parallel(
                qna_dataset=qna_dataset,
                num_questions=num_questions,
                excel_file_path=dynamic_excel_path,
                max_workers=evaluation_config['max_workers']
            )
            logging.info(f"Evaluation completed. Chain of RAG Accuracy: {chain_of_rag_accuracy}%, Gemini Accuracy: {gemini_accuracy}%")

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
            evaluator.visualize_accuracies(chain_of_rag_accuracy, gemini_accuracy, save_path=dynamic_plot_path)
        
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
    else:
        logging.info("Evaluation mode is OFF. Skipping evaluation.")

    # Exit after all stages are completed
    logging.info("O-RAN RAG Project completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()