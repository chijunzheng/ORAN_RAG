# src/main.py

import os
import argparse
import logging
import sys
from google.cloud import storage

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils.config_manager import ConfigManager
from src.authentication.auth_manager import AuthManager
from src.data_processing.converters import DocumentConverter
from src.data_processing.loaders import PDFLoader
from src.data_processing.text_formatter import TextFormatter
from src.data_processing.document_chunker import DocumentChunker
from src.embeddings.embedder import Embedder
from src.vector_search.indexer import VectorIndexer
from src.vector_search.searcher import VectorSearcher
from src.vector_search.corpus_manager import RagCorpusManager
from src.chatbot.chatbot import Chatbot
from src.evaluation.evaluator import Evaluator
from src.utils.logger import setup_logging
from src.utils.helpers import ensure_directory
from src.utils.rag_importer import RagImporter


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="O-RAN RAG Project",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Usage Examples:
          1. Run the full pipeline with evaluation:
             python src/main.py --preprocessing on --create-index on --deploy-index on --initialize-rag-corpus on --upload-files-to-rag-corpus on --evaluation on

          2. Run the pipeline without evaluation and skip preprocessing:
             python src/main.py --preprocessing off --evaluation off

          3. Initialize RAG Corpus and upload files:
             python src/main.py --initialize-rag-corpus on --upload-files-to-rag-corpus on

          4. Display the help message:
             python src/main.py --help

        Note:
          - The `--preprocessing` flag controls all preprocessing stages: "on" to perform, "off" to skip.
          - The `--create-index` flag controls the index creation stage.
          - The `--deploy-index` flag controls the index deployment stage.
          - The `--initialize-rag-corpus` flag controls the RAG corpus creation.
          - The `--upload-files-to-rag-corpus` flag controls uploading files to the RAG corpus.
          - The `--evaluation` flag controls model evaluation.
        """
    )

    # Pipeline stage flags
    parser.add_argument(
        '--preprocessing',
        type=str,
        choices=['on', 'off'],
        default='on',
        help='Toggle preprocessing: "on" to perform, "off" to skip. (default: on)'
    )
    parser.add_argument(
        '--create-index',
        type=str,
        choices=['on', 'off'],
        default='on',
        help='Toggle index creation: "on" to create, "off" to skip. (default: on)'
    )
    parser.add_argument(
        '--deploy-index',
        type=str,
        choices=['on', 'off'],
        default='on',
        help='Toggle index deployment: "on" to deploy, "off" to skip. (default: on)'
    )
    parser.add_argument(
        '--initialize-rag-corpus',
        type=str,
        choices=['on', 'off'],
        default='off',
        help='Toggle RAG corpus initialization: "on" to create or "off" to skip. (default: off)'
    )
    parser.add_argument(
        '--upload-files-to-rag-corpus',
        type=str,
        choices=['on', 'off'],
        default='off',
        help='Toggle uploading files into RAG corpus: "on" to upload, "off" to skip. (default: off)'
    )
    parser.add_argument(
        '--evaluation',
        type=str,
        choices=['on', 'off'],
        default='off',
        help='Toggle evaluation mode: "on" to evaluate, "off" to skip. (default: off)'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize logging early
    setup_logging(log_level=logging.INFO, log_file=None)
    logging.info("Starting O-RAN RAG Project")

    # Load configurations
    try:
        config_manager = ConfigManager(config_path='configs/config.yaml')
        config = config_manager.get_config()
    except Exception as e:
        setup_logging(log_level=logging.ERROR, log_file=None)
        logging.error(f"Failed to load or validate config: {e}")
        sys.exit(1)

    # Re-initialize logging to a file if specified in the config
    log_file = config.get('logging', {}).get('log_file')
    if log_file:
        setup_logging(log_level=logging.INFO, log_file=log_file)
        logging.info(f"Logging to file: {log_file}")

    # Initialize authentication
    try:
        auth_manager = AuthManager(config=config_manager.get_config())
        auth_manager.authenticate_user()
        logging.info("Authentication successful.")
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        sys.exit(1)

    # Extract relevant configurations
    gcp_config = config_manager.get_config()['gcp']
    paths_config = config_manager.get_config()['paths']
    vector_search_config = config_manager.get_config()['vector_search']
    chunking_config = config_manager.get_config()['chunking']
    generation_config = config_manager.get_config()['generation']
    evaluation_config = config_manager.get_config()['evaluation']

    # Ensure directories exist
    ensure_directory(paths_config['embeddings_save_path'])
    log_dir = os.path.dirname(log_file) or 'log'
    ensure_directory(log_dir)
    ensure_directory(os.path.dirname(evaluation_config['excel_file_path']) or 'Evaluation')
    ensure_directory(os.path.dirname(evaluation_config['plot_save_path']) or 'Evaluation')

    # Pipeline flags
    perform_preprocessing = (args.preprocessing == 'on')
    perform_create_index = (args.create_index == 'on')
    perform_deploy_index = (args.deploy_index == 'on')
    initialize_rag_corpus = (args.initialize_rag_corpus == 'on')
    upload_files_to_rag_corpus = (args.upload_files_to_rag_corpus == 'on')
    perform_evaluation = (args.evaluation == 'on')

    # Validate chunking config
    if chunking_config['chunk_size'] <= 0:
        logging.error("chunk_size must be a positive integer.")
        sys.exit(1)
    if chunking_config['chunk_overlap'] < 0:
        logging.error("chunk_overlap cannot be negative.")
        sys.exit(1)
    if chunking_config['chunk_overlap'] >= chunking_config['chunk_size']:
        logging.error("chunk_overlap must be smaller than chunk_size.")
        sys.exit(1)

    # Initialize VectorIndexer and VectorSearcher
    try:
        indexer = VectorIndexer(config=config_manager.get_config())
    except Exception as e:
        logging.error(f"Failed to initialize VectorIndexer: {e}")
        sys.exit(1)

    try:
        vector_searcher = VectorSearcher(
            config=config_manager.get_config(),
            credentials=auth_manager.get_credentials()
        )
    except Exception as e:
        logging.error(f"Failed to initialize VectorSearcher: {e}")
        sys.exit(1)

    # ------------------ PREPROCESSING STAGE ------------------ #
    if perform_preprocessing:
        logging.info("Starting preprocessing stages.")
        # 1. Convert docx to PDF
        # converter = DocumentConverter(directory_path=paths_config['documents'])
        # try:
        #     converter.convert_docx_to_pdf()
        #     logging.info("Document conversion completed.")
        # except Exception as e:
        #     logging.error(f"Document conversion failed: {e}")
        #     sys.exit(1)

        # 2. Load PDFs
        loader = PDFLoader(pdf_directory=paths_config['documents'])
        try:
            documents = loader.load_multiple_pdfs()
            logging.info(f"Loaded {len(documents)} PDF documents.")
        except Exception as e:
            logging.error(f"Failed to load PDFs: {e}")
            sys.exit(1)

        # 3. Text Formatting
        formatter = TextFormatter()
        try:
            all_cleaned_documents = formatter.format_documents(documents)
            logging.info(f"Formatted {len(all_cleaned_documents)} documents.")
        except Exception as e:
            logging.error(f"Text formatting failed: {e}")
            sys.exit(1)

        # 4. Document Chunking
        chunker = DocumentChunker(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap'],
            separators=chunking_config['separators'],
            gcs_bucket_name=gcp_config['bucket_name'],
            gcs_embeddings_path=gcp_config['embeddings_path'],
            credentials=auth_manager.get_credentials(),
            min_char_count=chunking_config['min_char_count']
        )
        try:
            chunks_with_ids = chunker.split_documents(all_cleaned_documents)
            logging.info(f"Split documents into {len(chunks_with_ids)} chunks.")
        except Exception as e:
            logging.error(f"Document chunking failed: {e}")
            sys.exit(1)

        # 5. Save chunks
        chunks_file = os.path.join(paths_config['embeddings_save_path'], 'chunks.jsonl')
        try:
            chunker.save_chunks_to_jsonl(chunks_with_ids, file_path=chunks_file)
            logging.info(f"Saved chunks to {chunks_file}")
        except Exception as e:
            logging.error(f"Failed to save chunks: {e}")
            sys.exit(1)

        # 6. Upload chunks to GCS (single file)
        try:
            chunker.upload_to_gcs(file_path=chunks_file, overwrite=True)
            logging.info("Uploaded chunks to GCS.")
        except Exception as e:
            logging.error(f"Failed to upload chunks to GCS: {e}")
            sys.exit(1)

        # 7. Split + Upload chunk files to GCS under 'chunks_split/'
        try:
            chunker.split_and_upload_chunks(local_chunks_file=chunks_file)
            logging.info("Successfully split and uploaded chunk files under 'chunks_split/'.")
        except Exception as e:
            logging.error(f"Failed to split and upload chunk files: {e}")
            sys.exit(1)

        # 8. Generate Embeddings
        embedder = Embedder(
            config=config_manager.get_config(),
            credentials=auth_manager.get_credentials()
        )
        try:
            embeddings_file = os.path.join(paths_config['embeddings_save_path'], 'embeddings.json')
            embedder.generate_and_store_embeddings(chunks_with_ids, local_jsonl_path=embeddings_file)
            logging.info(f"Embeddings generated and saved to {embeddings_file}")
        except Exception as e:
            logging.error(f"Embeddings generation failed: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping preprocessing stages.")

    # ------------------ INDEX CREATION STAGE ------------------ #
    newly_created_index = None
    existing_index = None
    if perform_create_index:
        logging.info("Starting index creation.")
        index_display_name = vector_search_config.get('index_display_name', 'default_index')

        # Attempt to find an existing index with that display name
        existing_index = indexer.get_index_by_display_name(index_display_name)
        if existing_index:
            logging.info(f"Index '{index_display_name}' already exists. Skipping creation.")
        else:
            try:
                newly_created_index = indexer.create_index()
                logging.info(f"Index created: {newly_created_index.resource_name}")
            except Exception as e:
                logging.error(f"Index creation failed: {e}")
                sys.exit(1)
    else:
        logging.info("Skipping index creation stage.")

    # ------------------ INDEX DEPLOYMENT STAGE ------------------ #
    if perform_deploy_index:
        logging.info("Starting index deployment.")
        try:
            if newly_created_index:
                # Use the index we just created in this run
                logging.info("Using newly created index for deployment.")
                index_to_deploy = newly_created_index
            else:
                # If no new index was created, check if we found an existing index
                index_display_name = vector_search_config.get('index_display_name', 'default_index')
                if existing_index:
                    logging.info(f"Using existing index '{existing_index.display_name}' for deployment.")
                    index_to_deploy = existing_index
                else:
                    # Retrieve existing by display name in case we haven't yet
                    index_to_deploy = indexer.get_index_by_display_name(index_display_name)
                    if not index_to_deploy:
                        raise ValueError(
                            f"No index found with display_name='{index_display_name}'. "
                            "Please create one first or set '--create-index on'."
                        )

            # Check if already deployed
            is_already_deployed = indexer.is_index_deployed(index=index_to_deploy)
            if is_already_deployed:
                logging.info(f"Index '{index_to_deploy.display_name}' is already deployed. Skipping deployment.")
            else:
                endpoint, deployed_index_id = indexer.deploy_index(index=index_to_deploy)
                config_manager.update_config({
                    'vector_search.deployed_index_id': deployed_index_id
                })
                logging.info(f"Deployed index with ID: {deployed_index_id}")

        except Exception as e:
            logging.error(f"Index deployment failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        logging.info("Skipping index deployment stage.")

    # ------------------ RAG CORPUS INITIALIZATION STAGE ------------------ #
    if initialize_rag_corpus:
        logging.info("Initializing RAG Corpus.")
        try:
            rag_corpus_manager = RagCorpusManager(
                config=config_manager.get_config(),
                credentials=auth_manager.get_credentials()
            )
            rag_corpus_resource = rag_corpus_manager.initialize_corpus(
                display_name=vector_search_config['rag_corpus_display_name'],
                description=vector_search_config['rag_corpus_description']
            )
            config_manager.update_config({
                'vector_search.rag_corpus_resource': rag_corpus_resource
            })
            logging.info(f"RAG Corpus initialized: {rag_corpus_resource}")
        except Exception as e:
            logging.error(f"Failed to initialize RAG Corpus: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping RAG Corpus initialization stage.")

        # ------------------ UPLOAD FILES TO RAG CORPUS STAGE ------------------ #
    if upload_files_to_rag_corpus:
        logging.info("Uploading files to RAG Corpus.")
        try:
            rag_corpus_manager = RagCorpusManager(
                config=config_manager.get_config(),
                credentials=auth_manager.get_credentials()
            )
            rag_corpus_resource = config_manager.get_config()['vector_search']['rag_corpus_resource']

            # Create a storage client
            storage_client = storage.Client(
                project=gcp_config['project_id'],
                credentials=auth_manager.get_credentials()
            )
            bucket = storage_client.bucket(gcp_config['bucket_name'])

            # Directories with all splitted JSON files
            split_directories = [
                "embeddings/chunks_split",
                "embeddings/embeddings_split"
            ]

            # Gather all .json file URIs
            gcs_file_uris = []
            for prefix in split_directories:
                if not prefix.endswith('/'):
                    prefix += '/'
                blobs = bucket.list_blobs(prefix=prefix)
                for blob in blobs:
                    if blob.name.endswith(".json"):
                        file_uri = f"gs://{gcp_config['bucket_name']}/{blob.name}"
                        gcs_file_uris.append(file_uri)

            if not gcs_file_uris:
                logging.warning(
                    "No splitted .json files found in 'chunks_split/' or 'embeddings_split/'. "
                    "Ensure splitted .json files exist before proceeding."
                )
            else:
                logging.info(f"Total splitted JSON files found: {len(gcs_file_uris)}")

            # NEW: Use RagImporter to batch-import URIs in sub-lists of 25
            results = RagImporter.batch_import_rag_files(
                rag_corpus_mgr=rag_corpus_manager,
                rag_corpus_res=rag_corpus_resource,
                uri_list=gcs_file_uris,
                chunk_size=512,
                chunk_overlap=100,
                max_embedding_requests_per_min=900
            )
            logging.info(
                f"Completed batch import. total_imported={results['total_imported']}, "
                f"total_skipped={results['total_skipped']}"
            )

        except Exception as e:
            logging.error(f"Failed to upload splitted JSON files to RAG Corpus: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping upload of files to RAG Corpus.")

    # ------------------ EVALUATION STAGE ------------------ #
    if perform_evaluation and evaluation_config.get('num_questions', 0) > 0:
        logging.info("Starting evaluation stage.")
        try:
            evaluator = Evaluator(
                project_id=gcp_config['project_id'],
                location=gcp_config['location'],
                bucket_name=gcp_config['bucket_name'],
                embeddings_path=gcp_config['embeddings_path'],
                qna_dataset_path=gcp_config['qna_dataset_path'],
                generation_config=generation_config,
                vector_searcher=vector_searcher,
                credentials=auth_manager.get_credentials(),
                num_neighbors=vector_search_config['num_neighbors']
            )

            qna_dataset = evaluator.load_qna_dataset_from_gcs()
            num_questions = evaluation_config['num_questions']
            excel_file_path = evaluation_config['excel_file_path']
            max_workers = evaluation_config.get('max_workers', 1)

            rag_accuracy, gemini_accuracy = evaluator.evaluate_models_parallel(
                qna_dataset=qna_dataset,
                num_questions=num_questions,
                excel_file_path=excel_file_path,
                max_workers=max_workers
            )

            # Visualization
            evaluator.visualize_accuracies(
                rag_accuracy=rag_accuracy,
                gemini_accuracy=gemini_accuracy,
                save_path=evaluation_config.get('plot_save_path')
            )

        except Exception as e:
            logging.error(f"Evaluation stage failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        logging.info("Skipping evaluation stage or num_questions=0.")

    # ------------------ Start Chatbot or Exit Pipeline ------------------ #

    if not perform_evaluation:
        logging.info("Starting chatbot interaction stage.")
    try:
        chatbot = Chatbot(
            config=config_manager.get_config(),
            vector_searcher=vector_searcher,
            credentials=auth_manager.get_credentials()
        )
        chatbot.chat_loop()
    except Exception as e:
        logging.error(f"Chatbot encountered an error: {e}")
        sys.exit(1)

    logging.info("O-RAN RAG Project completed successfully. End of main pipeline logic.")


if __name__ == "__main__":
    main()