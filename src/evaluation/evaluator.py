# src/evaluation/evaluator.py

import json
import logging
import os
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from google.cloud import storage
from openpyxl import Workbook
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part
)
from src.vector_search.searcher import VectorSearcher
from src.vector_search.reranker import Reranker
from src.chatbot.rat_processor import RATProcessor
from src.chatbot.chain_of_rag import ChainOfRagProcessor
from src.evaluation.rat_processor_eval import RATProcessorEval

# Add imports for reading .env file
try:
    from dotenv import load_dotenv
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    dotenv_path = os.path.join(project_root, '.env')
    
    # Check if .env file exists
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logging.info(f"Loaded .env file from {dotenv_path}")
    else:
        logging.warning(f".env file not found at {dotenv_path}")
except ImportError:
    logging.warning("python-dotenv not installed. Reading .env file manually.")
    try:
        # Define possible locations for the .env file
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        possible_paths = [
            '.env',  # Current directory
            os.path.join(project_root, '.env'),  # Project root
        ]
        
        # Try each path
        for env_path in possible_paths:
            if os.path.exists(env_path):
                logging.info(f"Found .env file at {env_path}")
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                            logging.info(f"Manually set {key} from .env file")
                break  # Exit loop if a file was found and processed
        else:
            logging.warning("No .env file found in expected locations")
    except Exception as e:
        logging.error(f"Error manually reading .env file: {e}")
        logging.warning("Falling back to environment variables")


class Evaluator:
    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: str,
        embeddings_path: str,
        qna_dataset_path: str,
        qna_dataset_local_path: str,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        generation_config: Dict,
        vector_searcher: VectorSearcher,
        credentials,
        num_neighbors: int,
        reranker: Reranker
    ):
        """
        Initializes the Evaluator with necessary configurations.

        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud region.
            bucket_name (str): GCS bucket name.
            embeddings_path (str): Path within the bucket where embeddings are stored.
            qna_dataset_path (str): Path to the Q&A dataset in GCS (deprecated, kept for compatibility).
            qna_dataset_local_path (str): Local path to the Q&A dataset file.
            generation_config (Dict): Configuration for text generation parameters.
            vector_searcher (VectorSearcher): Instance of VectorSearcher for performing vector searches.
            credentials: Google Cloud credentials object.
            num_neighbors (int): Number of neighbors to retrieve in vector search.
            reranker (Reranker): Instance of Reranker for reranking search results.
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.embeddings_path = embeddings_path
        self.qna_dataset_path = qna_dataset_path  # Kept for backward compatibility
        self.qna_dataset_local_path = qna_dataset_local_path
        self.vector_searcher = vector_searcher  
        
        # Get Gemini API key from environment variables (loaded from .env file)
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            # Import the direct API client
            import google.generativeai as genai
            
            # Configure with API key
            genai.configure(api_key=api_key)
            
            # Use the GenerativeModel from google.generativeai
            self.generative_model = genai.GenerativeModel("gemini-1.5-flash-002")
            logging.info("Initialized GenerativeModel using API key from .env file")
        else:
            # Fall back to VertexAI authentication
            self.generative_model = GenerativeModel("gemini-1.5-flash-002")
            logging.warning("GEMINI_API_KEY not found in environment variables. Falling back to Google Cloud authentication.")
        
        self.generation_config = GenerationConfig(
            temperature=generation_config.get('temperature', 0.3),
            top_p=generation_config.get('top_p', 0.9),
            max_output_tokens=generation_config.get('max_output_tokens', 8192),
        )
        self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.num_neighbors = num_neighbors
        self.index_endpoint_display_name = index_endpoint_display_name
        self.deployed_index_id = deployed_index_id
        self.reranker = reranker

        logging.info("Evaluator initialized successfully.")

    def upload_qna_dataset_to_gcs(self, local_file_path: str, gcs_file_name: str):
        """
        Uploads the Q&A dataset to GCS.

        Args:
            local_file_path (str): Local path of the Q&A dataset.
            gcs_file_name (str): Desired GCS path for the dataset.
        """
        try:
            blob = self.bucket.blob(gcs_file_name)
            blob.upload_from_filename(local_file_path, content_type="application/json")
            logging.info(f"Uploaded {local_file_path} to gs://{self.bucket_name}/{gcs_file_name}")
        except Exception as e:
            logging.error(f"Failed to upload Q&A dataset to GCS: {e}", exc_info=True)
            raise

    def load_qna_dataset(self) -> List[List[str]]:
        """
        Loads the Q&A dataset from a local file.

        Returns:
            List[List[str]]: List of Q&A entries as [question, choices, correct_answer].
        """
        try:
            # Check if local path exists
            if not os.path.exists(self.qna_dataset_local_path):
                logging.warning(f"Local Q&A dataset file not found at {self.qna_dataset_local_path}. Trying to fall back to GCS.")
                return self.load_qna_dataset_from_gcs()
            
            qna_dataset = []
            valid_entries = 0
            error_entries = 0
            
            # First try to load the entire file as a JSON array
            try:
                with open(self.qna_dataset_local_path, 'r') as file:
                    content = file.read().strip()
                    if content.startswith('[') and content.endswith(']'):
                        # Try parsing as a JSON array
                        entries = json.loads(content)
                        if isinstance(entries, list):
                            for entry in entries:
                                if isinstance(entry, dict):
                                    question = entry.get("question")
                                    # Handle both "choices" and "options" keys
                                    choices = entry.get("choices") or entry.get("options")
                                    answer = entry.get("answer")
                                    
                                    if question and isinstance(choices, list) and answer:
                                        qna_dataset.append([question, choices, answer])
                                        valid_entries += 1
                                    else:
                                        error_entries += 1
                            
                            if valid_entries > 0:
                                logging.info(f"Loaded {valid_entries} valid Q&A entries from JSON array. Skipped {error_entries} invalid entries.")
                                return qna_dataset
            except json.JSONDecodeError:
                # Not a valid JSON array, try parsing line by line
                pass
            
            # If we reach here, try parsing line by line (JSON Lines format)
            valid_entries = 0
            error_entries = 0
            
            with open(self.qna_dataset_local_path, 'r') as file:
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Try to parse the line as JSON
                        qna_entry = json.loads(line)
                        
                        # Handle different possible formats
                        
                        # Format 1: [question, choices, answer] array directly
                        if isinstance(qna_entry, list) and len(qna_entry) == 3:
                            question = qna_entry[0]
                            choices = qna_entry[1]
                            answer = qna_entry[2]
                            
                            if isinstance(question, str) and isinstance(choices, list) and isinstance(answer, str):
                                qna_dataset.append([question, choices, answer])
                                valid_entries += 1
                            else:
                                logging.warning(f"Line {line_number}: Invalid array format - expected [string, list, string].")
                                error_entries += 1
                        
                        # Format 2: {"question": "...", "choices": [...], "answer": "..."}
                        # or {"question": "...", "options": [...], "answer": "..."}
                        elif isinstance(qna_entry, dict):
                            question = qna_entry.get("question")
                            # Handle both "choices" and "options" keys
                            choices = qna_entry.get("choices") or qna_entry.get("options")
                            answer = qna_entry.get("answer")
                            
                            if question and isinstance(choices, list) and answer:
                                qna_dataset.append([question, choices, answer])
                                valid_entries += 1
                            else:
                                logging.warning(
                                    f"Line {line_number}: Invalid dictionary format - "
                                    f"missing required fields or invalid types."
                                )
                                error_entries += 1
                        else:
                            logging.warning(f"Line {line_number}: Unexpected format - {type(qna_entry)}.")
                            error_entries += 1
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"Line {line_number}: JSONDecodeError - {e}")
                        error_entries += 1
            
            if valid_entries == 0 and error_entries > 0:
                logging.error(
                    f"No valid entries found in the dataset file. "
                    f"Found {error_entries} invalid entries."
                )
                return []
                
            logging.info(
                f"Loaded {valid_entries} valid Q&A entries from local file. "
                f"Skipped {error_entries} invalid entries."
            )
            return qna_dataset
        except Exception as e:
            logging.error(f"Failed to load Q&A dataset from local file: {e}", exc_info=True)
            logging.warning("Trying to fall back to GCS.")
            return self.load_qna_dataset_from_gcs()

    def load_qna_dataset_from_gcs(self) -> List[List[str]]:
        """
        Loads the Q&A dataset from GCS (Legacy method, kept for backward compatibility).

        Returns:
            List[List[str]]: List of Q&A entries as [question, choices, correct_answer].
        """
        try:
            blob = self.bucket.blob(self.qna_dataset_path)
            content = blob.download_as_text()

            qna_dataset = []
            for line_number, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    qna_entry = json.loads(line)
                    if isinstance(qna_entry, list) and len(qna_entry) == 3:
                        qna_dataset.append(qna_entry)
                    elif isinstance(qna_entry, dict):
                        question = qna_entry.get("question")
                        choices = qna_entry.get("choices")
                        correct_answer = qna_entry.get("answer")
                        if question and choices and correct_answer:
                            qna_dataset.append([question, choices, correct_answer])
                        else:
                            logging.warning(f"Line {line_number}: Missing fields.")
                    else:
                        logging.warning(f"Line {line_number}: Unexpected format.")
                except json.JSONDecodeError as e:
                    logging.error(f"Line {line_number}: JSONDecodeError - {e}")
            logging.info(f"Loaded {len(qna_dataset)} Q&A entries from GCS.")
            return qna_dataset
        except Exception as e:
            logging.error(f"Failed to load Q&A dataset from GCS: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # (1) STEP-BACK PROMPT: Generate a single core concept
    #     Mimics the prompt snippet from chatbot.py
    # ------------------------------------------------------------------
    def generate_eval_core_concept(self, question: str) -> str:
        """
        Calls the LLM to generate the single core concept behind the question,
        using the same prompt pattern from chatbot.py.
        """
        # This is a minimal example: in evaluation, we often
        # don't have a "conversation history," so we skip that or feed empty.
        conversation_text = ""  # Or store last few Q&A turns if you wish

        prompt = f"""
        You are an O-RAN expert. Analyze the user's query below 
        and identify the single core concept needed to answer it.

        Conversation (recent):
        {conversation_text}

        User Query: {question}

        Instructions:
        1. Provide a concise concept or principle behind this query.
        2. Do not provide any further explanation—only briefly describe the concept.

        Concept:
        """
        
        try:
            # Check if we're using the API key integration (google.generativeai package)
            if 'google.generativeai' in str(type(self.generative_model)):
                # For google.generativeai, we can directly pass the text
                # OVERRIDE: Use lower randomness for concept extraction
                import google.generativeai as genai
                
                # Create a generation config with lower randomness
                concept_generation_params = {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": 128
                }
                
                response = self.generative_model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(**concept_generation_params)
                )
            else:
                # For vertexai.generative_models, we need to create a Content object
                user_prompt = Content(
                    role="user",
                    parts=[Part.from_text(prompt)]
                )

                # OVERRIDE: Use lower randomness for concept extraction
                concept_generation_config = GenerationConfig(
                    temperature=0.0,   # or 0.1
                    top_p=1.0,         # no nucleus sampling
                    max_output_tokens=128 
                )
                
                response = self.generative_model.generate_content(
                    user_prompt,
                    generation_config=concept_generation_config,
                )
                
            raw_text = response.text.strip()
            # Basic fallback if empty
            if not raw_text:
                return "O-RAN Architecture (General)"
            return raw_text
        except Exception as e:
            logging.error(f"Error generating eval core concept: {e}", exc_info=True)
            return "O-RAN Architecture (General)"

    # ------------------------------------------------------------------
    # (2) SAFE GENERATE CONTENT
    # ------------------------------------------------------------------


    def safe_generate_content(
        self,
        content: Content,
        retries: int = 10,
        backoff_factor: int = 2,
        max_wait: int = 300  # Maximum wait increased to 5 minutes for non-quota errors.
    ) -> str:
        """
        Generates content with exponential backoff and jitter.
        If a quota or rate-limit error is encountered, it waits a fixed longer duration
        (e.g., 120 seconds) before retrying, to allow the quota to recover.
        """
        wait_time = 1
        for attempt in range(1, retries + 1):
            try:
                # Check if we're using the API key integration (google.generativeai package)
                if 'google.generativeai' in str(type(self.generative_model)):
                    # For google.generativeai.generative_models.GenerativeModel
                    if isinstance(content, Content):
                        # Extract the text from the Content object
                        prompt_text = ""
                        for part in content.parts:
                            if hasattr(part, 'text'):
                                prompt_text += part.text
                        
                        response = self.generative_model.generate_content(prompt_text)
                    else:
                        # If it's already a string
                        response = self.generative_model.generate_content(content)
                else:
                    # For vertexai.generative_models.GenerativeModel
                    response = self.generative_model.generate_content(
                        content,
                        generation_config=self.generation_config,
                    )
                
                response_text = response.text.strip()
                # If response is empty or equals "error", consider it a failure.
                if not response_text or response_text.lower() == "error":
                    raise ValueError("Empty or error response.")
                return response_text

            except Exception as e:
                error_text = str(e).lower()
                # Detect quota or rate limit errors.
                if any(keyword in error_text for keyword in ["429", "quota", "rate limit", "resourceexhausted"]):
                    logging.warning("Quota or rate limit error detected; forcing an extended wait.")
                    fixed_wait = 60  # Fixed wait time for quota errors.
                    if attempt == retries:
                        logging.error(f"Final attempt {attempt}: {e}. Returning error message.")
                        return "Error: LLM generation failed after multiple attempts due to quota issues."
                    logging.warning(f"Attempt {attempt}: {e}. Retrying in {fixed_wait} seconds.")
                    time.sleep(fixed_wait)
                    continue  # Skip the exponential backoff for this iteration.

                # For other errors, use exponential backoff with jitter.
                if attempt == retries:
                    logging.error(f"Final attempt {attempt}: {e}. Returning error message.")
                    return "Error: LLM generation failed after multiple attempts."
                jitter = random.uniform(0, 1)
                sleep_time = min(wait_time * backoff_factor, max_wait)
                logging.warning(f"Attempt {attempt}: {e}. Retrying in {sleep_time + jitter:.2f} seconds.")
                time.sleep(sleep_time + jitter)
                wait_time *= backoff_factor
        return "Error: LLM generation failed after multiple attempts."

    # ------------------------------------------------------------------
    # (3) HELPER: Build prompt for final multiple-choice generation
    #     We'll incorporate the core concept in the final prompt
    # ------------------------------------------------------------------
    def generate_prompt_content_evaluation(
        self,
        query: str,
        choices: List[str],
        chunks: List[Dict],
        core_concept: str
    ) -> Content:
        """
        Generates prompt content for evaluation using RAG, 
        injecting the single core concept from Step-Back.
        """
        context = "\n\n".join([
            f"Chunk {i+1}:\n{chunk['content']}" for i, chunk in enumerate(chunks)
        ])
        choices_text = "\n".join(choices)

        prompt_text = f"""
        You are an O-RAN expert. Focus on this core concept when forming your answer:
        "{core_concept}"

        Utilize the context below if relevant, then decide which choice is correct.

        Context:
        {context}

        Question:
        {query}

        Choices:
        {choices_text}

        Please provide the correct answer choice number (1, 2, 3, or 4) only.
        """
        user_prompt = Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )
        return user_prompt

    # ------------------------------------------------------------------
    # (4) RAG PIPELINE with STEP-BACK
    # ------------------------------------------------------------------
    def query_rag_pipeline(self, question: str, choices: List[str]) -> str:
        """
        Queries the RAG pipeline to get an answer, but includes:
        1) Step-Back: core concept extraction
        2) Retrieval & Reranking
        3) Final prompt referencing the concept
        """
        try:
            # (A) STEP-BACK: Extract the single core concept
            core_concept = self.generate_eval_core_concept(question)
            logging.debug(f"[Step-Back] Concept: {core_concept}")

            # (B) Vector Search
            retrieved_chunks = self.vector_searcher.vector_search(
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
                query_text=question,
                num_neighbors=self.num_neighbors,
            )
            if not retrieved_chunks:
                logging.warning("No chunks retrieved for the query.")
                return "No relevant information found."

            # (C) Reranking with concept
            reranked_chunks = self.reranker.rerank(
                query=core_concept,
                records=retrieved_chunks,
            )
            if not reranked_chunks:
                logging.warning("Reranking returned no results.")
                return "No relevant information found after reranking."

            # (D) Build final prompt with concept + chunks
            prompt_content = self.generate_prompt_content_evaluation(
                query=question,
                choices=choices,
                chunks=reranked_chunks,
                core_concept=core_concept
            )

            # (E) Generate final answer
            assistant_response = self.safe_generate_content(prompt_content, retries=10, backoff_factor=2, max_wait=30)
            return assistant_response.strip() if assistant_response else "Error"

        except Exception as e:
            logging.error(f"Error in RAG pipeline for question '{question}': {e}", exc_info=True)
            return "Error"
        
    # ------------------------------------------------------------------
    # (5) OR Evaluate with Retrieval Augmented Thoughts (RAT)
    # ------------------------------------------------------------------

    def query_rat_pipeline(self, question: str, choices: List[str]) -> str:
        """
        Uses the evaluation-specific RATProcessor (RATProcessorEval) to generate an answer for a multiple-choice question.
        
        Pipeline:
        (A) Use RATProcessorEval (with iterative CoT refinement) to generate a chain-of-thought that incorporates both the question and the answer options.
        (B) Build a final prompt that instructs the LLM to provide detailed, step-by-step reasoning and conclude with a final line in the format:
                "Final Answer: X"
            where X is one of 1, 2, 3, or 4.
        
        Returns:
            A string representing the final LLM output.
        """
        try:

            # (A) Combine the question and choices into a single JSON-like array string
            joined_choices = ', '.join(f"\"{i+1}. {c}\"" for i, c in enumerate(choices))
            combined_query_str = (
                "["
                f"\"{question}\", "
                "\"Pick from the following choices and only return the ID of the choice:\", "
                f"[{joined_choices}]"
                "]"
            )
            # (B) Use RATProcessorEval to generate the chain-of-thought with choices.
            rat = RATProcessorEval(
                vector_searcher=self.vector_searcher,
                llm=self.generative_model,
                generation_config=self.generation_config,
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
                num_iterations=3,
            )
            conversation_history = []  # For evaluation, we typically have no history
            final_cot = rat.process_query(combined_query_str, conversation_history, choices=None)

            # (C) Construct the final refine prompt
            refine_prompt = f"""
            You are an O-RAN expert. The user provided the question and choices in a single array, exactly like so:
            {combined_query_str}

            Below is your final chain-of-thought from iterative retrieval augmentation. 
            Based on that chain-of-thought, refine it and use it as your reasoning first and then provide the final answer at the end:

            Final Chain-of-Thought:
            {final_cot}

            Final Answer: X

            where X is one of 1, 2, 3, or 4 (the correct choice).
            """

            refine_content = Content(role="user", parts=[Part.from_text(refine_prompt)])
            
            refined_response = self.safe_generate_content(
                refine_content, 
                retries=5, 
                backoff_factor=2, 
                max_wait=60
            ).strip()

            if not refined_response:
                logging.warning("[Eval] Final step empty; returning the chain-of-thought as fallback.")
                return final_cot

            logging.info(f"[Eval] Final Answer Extracted: {refined_response}")
            return refined_response

        except Exception as e:
            logging.error(f"Error in RAT evaluation for question '{question}': {e}", exc_info=True)
            return "Error"

    # ------------------------------------------------------------------
    # (5) Evaluate a single Q&A entry
    # ------------------------------------------------------------------
    def evaluate_single_entry(self, entry: List[str], delay: float = 0.5) -> Dict:
        """
        Evaluates a single Q&A entry using both RAG (with Step-Back) and Gemini LLM.

        Args:
            entry (List[str]): [question, choices, correct_answer]
            delay (float, optional): Delay between requests. Defaults to 0.5.

        Returns:
            Dict: Evaluation results.
        """
        try:
            time.sleep(delay)
            question, choices, correct_str = entry
            correct_choice = correct_str.strip()

            # Query RAG with Step-Back
            rag_full_answer = self.query_rag_pipeline(question, choices)
            rag_pred_choice = self.extract_choice_from_answer(rag_full_answer)
            rag_correct = (rag_pred_choice == correct_choice)

            # Query RAG with RAT
            # rag_full_answer = self.query_rat_pipeline(question, choices)
            # rag_pred_choice = self.extract_choice_from_answer(rag_full_answer)
            # rag_correct = (rag_pred_choice == correct_choice)

            # Query Gemini (no RAG)
            gemini_full_answer = self.query_gemini_llm(question, choices)
            gemini_pred_choice = self.extract_choice_from_answer(gemini_full_answer)
            gemini_correct = (gemini_pred_choice == correct_choice)

            return {
                'Question': question,
                'Correct Answer': correct_choice,
                'RAG Predicted Answer': rag_full_answer,
                'RAG Correct': rag_correct,
                'Gemini Predicted Answer': gemini_full_answer,
                'Gemini Correct': gemini_correct
            }

        except Exception as e:
            logging.error(f"Error processing entry: {e}", exc_info=True)
            return {
                'Question': entry[0],
                'Correct Answer': entry[2].strip(),
                'RAG Predicted Answer': "Error",
                'RAG Correct': False,
                'Gemini Predicted Answer': "Error",
                'Gemini Correct': False,
            }

    def extract_choice_from_answer(self, answer_text: str) -> str:
        """
        Extracts the choice number from the model's answer.
        """
        patterns = [
            r'The correct answer is[:\s]*([1-4])',
            r'Answer\s*[:\s]*([1-4])',
            r'^([1-4])\.',
            r'\b([1-4])\b',
        ]
        for pat in patterns:
            match = re.search(pat, answer_text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def query_gemini_llm(self, question: str, choices: List[str]) -> str:
        """
        Queries the raw Gemini LLM (no RAG) for an answer.
        """
        choices_text = "\n".join(choices)
        prompt_text = f"""
        You are an expert in O-RAN systems. A user has provided the following multiple-choice question:

        Question: {question}

        Choices:
        {choices_text}

        Please provide the correct answer choice number (1, 2, 3, or 4) only.
        """
        
        # Check if we're using the API key integration (google.generativeai package)
        if 'google.generativeai' in str(type(self.generative_model)):
            # For google.generativeai, we can directly pass the text
            return self.safe_generate_content(prompt_text)
        else:
            # For vertexai.generative_models, we need to create a Content object
            user_prompt = Content(
                role="user",
                parts=[Part.from_text(prompt_text)]
            )
            return self.safe_generate_content(user_prompt)

    # ------------------------------------------------------------------
    # (6) Evaluate Models in Parallel with Progress Bar and Save Results
    # ------------------------------------------------------------------
    def evaluate_models_parallel(
        self,
        qna_dataset: List[List[str]],
        num_questions: int,
        excel_file_path: str,
        max_workers: int = 1
    ) -> Tuple[float, float]:
        """
        Evaluates models in parallel and records results in an Excel file.

        Args:
            qna_dataset (List[List[str]]): List of Q&A entries.
            num_questions (int): Number of questions to evaluate.
            excel_file_path (str): Path to save the Excel results.
            max_workers (int, optional): Number of parallel threads. Defaults to 1.

        Returns:
            Tuple[float, float]: RAG and Gemini accuracies.
        """
        qna_subset = qna_dataset[:num_questions]
        wb = Workbook()
        ws = wb.active
        ws.title = "Evaluation Results"
        headers = [
            'Question',
            'Correct Answer',
            'RAG Predicted Answer',
            'RAG Correct',
            'Gemini Predicted Answer',
            'Gemini Correct',
        ]
        ws.append(headers)
        wb.save(excel_file_path)
        logging.info(f"Excel file created at {excel_file_path}")

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
                executor.submit(self.evaluate_single_entry, entry): idx
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

                    if result.get('RAG Correct'):
                        rag_correct += 1
                    if result.get('Gemini Correct'):
                        gemini_correct += 1

                    processed += 1
                    row = [
                        result.get('Question', ''),
                        result.get('Correct Answer', ''),
                        result.get('RAG Predicted Answer', ''),
                        result.get('RAG Correct', False),
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

        logging.info(f"RAG Accuracy: {rag_accuracy:.2f}%")
        logging.info(f"Gemini Accuracy: {gemini_accuracy:.2f}%")

        return rag_accuracy, gemini_accuracy
    


    def visualize_accuracies(self, rag_acc: float, gemini_acc: float, save_path: str = None):
        """
        Visualizes the accuracy comparison between RAG Pipeline and Gemini LLM.

        Args:
            rag_acc (float): Accuracy of the RAG Pipeline.
            gemini_acc (float): Accuracy of the Gemini LLM.
            save_path (str, optional): Path to save the plot image. If None, the plot is displayed.
        """
        models = ['RAG Pipeline', 'Raw Gemini LLM']
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