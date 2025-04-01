#!/usr/bin/env python3
# src/evaluation/run_evaluation.py

import os
import sys
import subprocess

# Try to ensure python-dotenv is installed
try:
    import dotenv
except ImportError:
    print("The python-dotenv package is not installed. Attempting to install it...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
        print("python-dotenv has been successfully installed.")
        import dotenv
    except Exception as e:
        print(f"Error installing python-dotenv: {e}")
        print("Continuing without environment variable loading from .env file")

# Try to ensure google-generativeai package is installed
try:
    import google.generativeai
except ImportError:
    print("The google-generativeai package is not installed. Attempting to install it...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
        print("google-generativeai has been successfully installed.")
        import google.generativeai
    except Exception as e:
        print(f"Error installing google-generativeai: {e}")
        print("Continuing with Vertex AI authentication only")

import argparse
import logging
import json
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

# Add project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.config import load_config
from src.utils.logger import setup_logging
from src.utils.helpers import ensure_directory
from src.authentication.auth_manager import AuthManager
from src.vector_search.searcher import VectorSearcher
from src.vector_search.reranker import Reranker
from src.evaluation.evaluator import Evaluator


def parse_arguments():
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run evaluation on different RAG pipelines",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Usage Examples:
          1. Run evaluation with default RAG pipeline:
             python src/evaluation/run_evaluation.py --pipeline default

          2. Run evaluation with Chain of RAG pipeline:
             python src/evaluation/run_evaluation.py --pipeline chain_of_rag

          3. Run evaluation with RAT pipeline:
             python src/evaluation/run_evaluation.py --pipeline rat

          4. Run evaluation with a limited number of questions:
             python src/evaluation/run_evaluation.py --pipeline chain_of_rag --num-questions 50
             
          5. Create a sample dataset and run evaluation:
             python src/evaluation/run_evaluation.py --pipeline default --create-sample-dataset
        """
    )

    # Configuration group
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to the configuration file. (default: configs/config.yaml)'
    )

    # Evaluation pipeline group
    pipeline_group = parser.add_argument_group('Pipeline Selection')
    pipeline_group.add_argument(
        '--pipeline',
        type=str,
        choices=['default', 'chain_of_rag', 'rat'],
        default='default',
        help='RAG pipeline to evaluate: default, chain_of_rag, or rat. (default: default)'
    )

    # Evaluation parameters group
    eval_group = parser.add_argument_group('Evaluation Parameters')
    eval_group.add_argument(
        '--num-questions',
        type=int,
        help='Number of questions to evaluate. If not provided, uses value from config.'
    )
    eval_group.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of worker threads for parallel evaluation. If not provided, uses value from config.'
    )
    eval_group.add_argument(
        '--results-dir',
        type=str,
        help='Directory to save evaluation results. If not provided, uses directory from config.'
    )
    eval_group.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of evaluation results. (default: False)'
    )

    return parser.parse_args()


def main():
    """Main function to run the evaluation script."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Initialize logging
    log_file = config.get('logging', {}).get('log_file')
    if log_file:
        setup_logging(log_level=logging.INFO, log_file=log_file)
    else:
        setup_logging(log_level=logging.INFO)

    logging.info("Starting O-RAN RAG Evaluation Script")
    logging.info(f"Selected RAG pipeline: {args.pipeline}")

    # Extract necessary parameters from config
    gcp_config = config['gcp']
    evaluation_config = config['evaluation']
    vector_search_config = config['vector_search']
    generation_config = config['generation']

    # Get evaluation parameters from args or config
    num_questions = args.num_questions or evaluation_config.get('num_questions', 100)
    max_workers = args.max_workers or evaluation_config.get('max_workers', 1)

    # Set up results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.results_dir:
        results_dir = args.results_dir
    else:
        base_results_dir = os.path.dirname(evaluation_config.get('excel_file_path', 'evaluation_results'))
        pipeline_dir = f"{args.pipeline}_pipeline"
        results_dir = os.path.join(base_results_dir, pipeline_dir)
    
    ensure_directory(results_dir)
    
    # Generate filenames for results
    excel_filename = f"evaluation_results_{args.pipeline}_{num_questions}_{timestamp}.xlsx"
    excel_file_path = os.path.join(results_dir, excel_filename)
    
    if args.visualize:
        plot_filename = f"accuracy_plot_{args.pipeline}_{num_questions}_{timestamp}.png"
        plot_file_path = os.path.join(results_dir, plot_filename)
    else:
        plot_file_path = None

    # Initialize authentication
    try:
        auth_manager = AuthManager(config=config)
        auth_manager.authenticate_user()
        logging.info("Authentication successful.")
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        sys.exit(1)

    # Initialize VectorSearcher
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

    # Initialize Reranker
    try:
        reranker = Reranker(
            project_id=gcp_config['project_id'],
            location=gcp_config['location'],
            ranking_config=config['ranking']['ranking_config'],
            credentials=auth_manager.credentials,
            model=config['ranking']['model'],
            rerank_top_n=config['ranking']['rerank_top_n']
        )
    except Exception as e:
        logging.error(f"Failed to initialize Reranker: {e}")
        sys.exit(1)

    # Initialize Evaluator
    try:
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
    except Exception as e:
        logging.error(f"Failed to initialize Evaluator: {e}")
        sys.exit(1)

    # Load dataset
    try:
        qna_dataset = evaluator.load_qna_dataset()
        logging.info(f"Loaded {len(qna_dataset)} Q&A entries.")
    except Exception as e:
        logging.error(f"Failed to load Q&A dataset: {e}")
        sys.exit(1)

    # Set pipeline name based on selection
    if args.pipeline == 'default':
        pipeline_name = "Default RAG"
    elif args.pipeline == 'chain_of_rag':
        pipeline_name = "Chain of RAG"
    elif args.pipeline == 'rat':
        pipeline_name = "RAT"

    # Customize the evaluation method based on selected pipeline
    def evaluate_single_entry_with_pipeline(entry: List[str], delay: float = 0.5) -> Dict:
        """Evaluate a single Q&A entry using the selected pipeline."""
        try:
            time.sleep(delay)
            question, choices, correct_str = entry
            correct_choice = correct_str.strip()

            # Select pipeline for RAG query
            if args.pipeline == 'default':
                rag_full_answer = evaluator.query_rag_pipeline(question, choices)
            elif args.pipeline == 'chain_of_rag':
                rag_full_answer = evaluator.query_chain_of_rag_pipeline(question, choices)
            elif args.pipeline == 'rat':
                rag_full_answer = evaluator.query_rat_pipeline(question, choices)
            
            rag_pred_choice = evaluator.extract_choice_from_answer(rag_full_answer)
            rag_correct = (rag_pred_choice == correct_choice)

            # Always query Gemini for comparison
            gemini_full_answer = evaluator.query_gemini_llm(question, choices)
            gemini_pred_choice = evaluator.extract_choice_from_answer(gemini_full_answer)
            gemini_correct = (gemini_pred_choice == correct_choice)

            return {
                'Question': question,
                'Correct Answer': correct_choice,
                f'{pipeline_name} Predicted Answer': rag_full_answer,
                f'{pipeline_name} Correct': rag_correct,
                'Gemini Predicted Answer': gemini_full_answer,
                'Gemini Correct': gemini_correct
            }
        except Exception as e:
            logging.error(f"Error processing entry: {e}")
            return {
                'Question': entry[0],
                'Correct Answer': entry[2].strip(),
                f'{pipeline_name} Predicted Answer': "Error",
                f'{pipeline_name} Correct': False,
                'Gemini Predicted Answer': "Error",
                'Gemini Correct': False,
            }
    
    # Monkey patch the evaluate_single_entry method in the evaluator
    evaluator.evaluate_single_entry = evaluate_single_entry_with_pipeline
    
    # Add the missing query_chain_of_rag_pipeline method to evaluator if it doesn't exist
    if not hasattr(evaluator, 'query_chain_of_rag_pipeline'):
        def query_chain_of_rag_pipeline(self, question: str, choices: List[str]) -> str:
            """
            Queries the Chain of RAG pipeline to get an answer.
            """
            try:
                from src.chatbot.chain_of_rag import ChainOfRagProcessor
                
                # Initialize Chain of RAG processor with reranker
                chain_of_rag = ChainOfRagProcessor(
                    vector_searcher=self.vector_searcher,
                    llm=self.generative_model,
                    generation_config=self.generation_config,
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    reranker=self.reranker,  # Pass the reranker instance
                    max_iterations=4,  # Use config value if available
                    early_stopping=True,
                    search_neighbors=40,  # Number of initial neighbors to retrieve
                    rerank_top_n=10  # Number of top reranked results to keep
                )
                
                # Combine the question and choices
                joined_choices = ', '.join(f"\"{i+1}. {c}\"" for i, c in enumerate(choices))
                combined_query = (
                    f"question: {question}\n"
                    f"options: {joined_choices}\n"
                    "Please provide the correct answer option number (1, 2, 3, or 4) only."
                )
                
                # Process query with Chain of RAG
                conversation_history = []
                final_answer, _ = chain_of_rag.process_query(combined_query, conversation_history)
                
                return final_answer
            except Exception as e:
                logging.error(f"Error in Chain of RAG pipeline: {e}")
                return "Error"
        
        # Add the method to the evaluator instance
        import types
        evaluator.query_chain_of_rag_pipeline = types.MethodType(query_chain_of_rag_pipeline, evaluator)

    # Override the evaluator's evaluate_models_parallel method to use custom headers
    original_evaluate_models_parallel = evaluator.evaluate_models_parallel
    
    def custom_evaluate_models_parallel(qna_dataset, num_questions, excel_file_path, max_workers):
        # Start by initializing the Excel workbook with our custom headers
        from openpyxl import Workbook
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Evaluation Results"
        
        headers = [
            'Question',
            'Correct Answer',
            f'{pipeline_name} Predicted Answer',
            f'{pipeline_name} Correct',
            'Gemini Predicted Answer',
            'Gemini Correct',
        ]
        
        ws.append(headers)
        wb.save(excel_file_path)
        logging.info(f"Excel file created at {excel_file_path}")
        
        # We'll implement a custom version of evaluate_models_parallel instead of using the original
        # This ensures we correctly handle the pipeline-specific column names
        qna_subset = qna_dataset[:num_questions]
        
        rag_correct = 0
        gemini_correct = 0
        processed = 0

        excel_lock = threading.Lock()
        rows_buffer = []
        buffer_size = 100

        def flush_rows_buffer():
            if not rows_buffer:
                return
            with excel_lock:
                for row in rows_buffer:
                    ws.append(row)
                wb.save(excel_file_path)
            rows_buffer.clear()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluator.evaluate_single_entry, entry): idx
                for idx, entry in enumerate(qna_subset, 1)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating Q&A Entries"
            ):
                idx = futures[future]
                try:
                    result = future.result()

                    # Use the pipeline-specific keys
                    if result.get(f'{pipeline_name} Correct'):
                        rag_correct += 1
                    if result.get('Gemini Correct'):
                        gemini_correct += 1

                    processed += 1
                    row = [
                        result.get('Question', ''),
                        result.get('Correct Answer', ''),
                        result.get(f'{pipeline_name} Predicted Answer', ''),
                        result.get(f'{pipeline_name} Correct', False),
                        result.get('Gemini Predicted Answer', ''),
                        result.get('Gemini Correct', False),
                    ]
                    rows_buffer.append(row)

                    if len(rows_buffer) >= buffer_size:
                        flush_rows_buffer()

                except Exception as exc:
                    logging.error(f"Exception for question {idx}: {exc}")

        flush_rows_buffer()
        wb.close()

        rag_accuracy = (rag_correct / processed) * 100.0 if processed > 0 else 0
        gemini_accuracy = (gemini_correct / processed) * 100.0 if processed > 0 else 0

        logging.info(f"{pipeline_name} Accuracy: {rag_accuracy:.2f}%")
        logging.info(f"Gemini Accuracy: {gemini_accuracy:.2f}%")
        
        # Use modified visualize_accuracies method with appropriate labels
        original_visualize = evaluator.visualize_accuracies
        
        def custom_visualize_accuracies(rag_acc, gemini_acc, save_path=None):
            import matplotlib.pyplot as plt
            models = [pipeline_name, 'Raw Gemini LLM']
            accuracies = [rag_acc, gemini_acc]

            plt.figure(figsize=(8,6))
            bars = plt.bar(models, accuracies, color=['blue', 'green'])
            plt.xlabel('Models')
            plt.ylabel('Accuracy (%)')
            plt.title('Model Accuracy Comparison')
            plt.ylim(0, 100)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}%', ha='center', va='bottom')

            if save_path:
                plt.savefig(save_path)
                logging.info(f"Accuracy comparison plot saved to {save_path}")
            else:
                plt.show()
        
        # Replace the method temporarily
        evaluator.visualize_accuracies = custom_visualize_accuracies
        
        # Call the visualization directly if needed
        if args.visualize and plot_file_path:
            custom_visualize_accuracies(rag_accuracy, gemini_accuracy, plot_file_path)
        
        # Restore the original visualization method
        evaluator.visualize_accuracies = original_visualize
        
        return rag_accuracy, gemini_accuracy
    
    # Run evaluation
    try:
        logging.info(f"Starting evaluation with {pipeline_name} pipeline on {num_questions} questions")
        logging.info(f"Results will be saved to {excel_file_path}")
        
        # Run the evaluation with our custom method
        rag_accuracy, gemini_accuracy = custom_evaluate_models_parallel(
            qna_dataset=qna_dataset,
            num_questions=num_questions,
            excel_file_path=excel_file_path,
            max_workers=max_workers
        )
        
        logging.info(f"Evaluation completed.")
        logging.info(f"{pipeline_name} Accuracy: {rag_accuracy:.2f}%")
        logging.info(f"Gemini Accuracy: {gemini_accuracy:.2f}%")
        
        # Visualization is now handled inside custom_evaluate_models_parallel when args.visualize is True
    
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)
    
    logging.info("Evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 