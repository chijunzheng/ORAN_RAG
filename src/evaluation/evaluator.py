# src/evaluation/evaluator.py

import json
import logging
import os
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from google.cloud import storage
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
from src.vector_search.searcher import VectorSearcher


class Evaluator:
    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: str,
        embeddings_path: str,
        qna_dataset_path: str,
        generation_config: Dict,
        vector_searcher: VectorSearcher,
        credentials,
        num_neighbors: int
    ):
        """
        Initializes the Evaluator with necessary configurations.

        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud region.
            bucket_name (str): GCS bucket name.
            embeddings_path (str): Path within the bucket where embeddings are stored.
            qna_dataset_path (str): Path to the Q&A dataset in GCS.
            generation_config (Dict): Configuration for text generation parameters.
            vector_searcher (VectorSearcher): Instance of VectorSearcher for performing vector searches.
            credentials: Google Cloud credentials.
            num_neighbors (int): Number of nearest neighbors to retrieve.
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.embeddings_path = embeddings_path
        self.qna_dataset_path = qna_dataset_path
        self.vector_searcher = vector_searcher
        self.generative_model = GenerativeModel("gemini-1.5-flash-002")
        self.generation_config = GenerationConfig(
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.9),
            max_output_tokens=generation_config.get('max_output_tokens', 1000),
        )
        self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
        self.bucket = self.storage_client.bucket(self.bucket_name)
        self.num_neighbors = num_neighbors

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

    def load_qna_dataset_from_gcs(self) -> List[List[str]]:
        """
        Loads the Q&A dataset from GCS.

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

    def safe_generate_content(
        self,
        content: Content,
        retries: int = 10,
        backoff_factor: float = 2.0,
        max_wait: float = 30.0
    ) -> str:
        """
        Generates content with exponential backoff and jitter.

        Args:
            content (Content): Content object for the generative model.
            retries (int, optional): Number of retry attempts. Defaults to 10.
            backoff_factor (float, optional): Backoff multiplier. Defaults to 2.0.
            max_wait (float, optional): Maximum wait time. Defaults to 30.0.

        Returns:
            str: Generated text or "Error" upon failure.
        """
        wait_time = 1.0
        for attempt in range(1, retries + 1):
            try:
                response = self.generative_model.generate_content(
                    content,
                    generation_config=self.generation_config,
                )
                response_text = response.text.strip()
                if not response_text:
                    raise ValueError("Empty response.")
                return response_text
            except Exception as e:
                if attempt == retries:
                    logging.error(f"Final attempt {attempt}: {e}. Skipping.")
                    return "Error"
                jitter = random.uniform(0, 1)
                sleep_time = min(wait_time * backoff_factor, max_wait)
                logging.warning(f"Attempt {attempt}: {e}. Retrying in {sleep_time + jitter:.2f} seconds.")
                time.sleep(sleep_time + jitter)
                wait_time *= backoff_factor
        return "Error"

    def extract_choice_from_answer(self, answer_text: str) -> Optional[str]:
        """
        Extracts the choice number from the model's answer.

        Args:
            answer_text (str): The model's response.

        Returns:
            Optional[str]: Extracted choice number or None.
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
        Queries the raw Gemini LLM for an answer.

        Args:
            question (str): The question text.
            choices (List[str]): List of choice strings.

        Returns:
            str: The model's answer.
        """
        choices_text = "\n".join(choices)
        prompt_text = f"""
You are an expert in O-RAN systems. A user has provided the following multiple-choice question:

Question: {question}

Choices:
{choices_text}

Please provide the correct answer choice number (1, 2, 3, or 4) only.
"""
        user_prompt = Content(
            role="user",
            parts=[
                Part.from_text(prompt_text)
            ]
        )
        return self.safe_generate_content(user_prompt)

    def generate_prompt_content_evaluation(
        self,
        query: str,
        choices: List[str],
        chunks: List[Dict]
    ) -> Content:
        """
        Generates prompt content for evaluation using RAG.

        Args:
            query (str): The question text.
            choices (List[str]): List of choice strings.
            chunks (List[Dict]): Retrieved text chunks.

        Returns:
            Content: The prompt content object.
        """
        sorted_chunks = sorted(chunks, key=lambda x: x['distance'])
        top_chunks = sorted_chunks[:5]
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk['content']}" for i, chunk in enumerate(top_chunks)])
        choices_text = "\n".join(choices)
        prompt_text = f"""
You are an expert in O-RAN systems. Utilize the context to provide detailed and accurate answers to the user's queries.

Instruction:
Using the information provided in the context, please provide a logical and concise answer to the question below.

If the question presents multiple choice options, you must:
- State the correct choice by its number (e.g., "The correct answer is: 3") at the start of your answer.

Context:
{context}

Question:
{query}

Choices:
{choices_text}

Please provide the correct answer choice number (1, 2, 3, or 4) only.

Answer:
"""
        user_prompt = Content(
            role="user",
            parts=[
                Part.from_text(prompt_text)
            ]
        )
        return user_prompt

    def query_rag_pipeline(self, question: str, choices: List[str]) -> str:
        """
        Queries the RAG pipeline to get an answer.

        Args:
            question (str): The question text.
            choices (List[str]): List of choice strings.

        Returns:
            str: The model's answer.
        """
        try:
            retrieved_chunks = self.vector_searcher.vector_search(
                query_text=question,
                num_neighbors=self.num_neighbors,
            )
        except Exception as e:
            logging.error(f"Vector search failed during RAG query: {e}", exc_info=True)
            return "Error"

        if not retrieved_chunks:
            return "No relevant information found."

        prompt_content = self.generate_prompt_content_evaluation(question, choices, retrieved_chunks)
        assistant_response = self.safe_generate_content(prompt_content)
        return assistant_response.strip() if assistant_response else "Error"

    def evaluate_single_entry(self, entry: List[str], delay: float = 0.5) -> Dict:
        """
        Evaluates a single Q&A entry using both RAG and Gemini LLM.

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

            # Query RAG
            rag_full_answer = self.query_rag_pipeline(question, choices)
            rag_pred_choice = self.extract_choice_from_answer(rag_full_answer)
            rag_correct = rag_pred_choice == correct_choice

            # Query Gemini
            gemini_full_answer = self.query_gemini_llm(question, choices)
            gemini_pred_choice = self.extract_choice_from_answer(gemini_full_answer)
            gemini_correct = gemini_pred_choice == correct_choice

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
        try:
            # Validate excel_file_path
            if os.path.isdir(excel_file_path):
                raise IsADirectoryError(f"Excel file path '{excel_file_path}' is a directory.")

            # Select a subset of Q&A entries
            selected_qnas = random.sample(qna_dataset, min(num_questions, len(qna_dataset)))
            logging.info(f"Selected {len(selected_qnas)} Q&A entries for evaluation.")

            # Initialize Excel Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Evaluation Results"
            headers = [
                'Question',
                'Correct Answer',
                'RAG Predicted Answer',
                'RAG Correct',
                'Gemini Predicted Answer',
                'Gemini Correct'
            ]
            ws.append(headers)
            wb.save(excel_file_path)
            logging.info(f"Excel file created at {excel_file_path}")

            # Initialize counters
            rag_correct = 0
            gemini_correct = 0
            processed = 0

            # Lock for writing to Excel
            excel_lock = threading.Lock()
            rows_buffer = []
            buffer_size = 100  # Flush every 100 rows

            def flush_rows_buffer():
                if not rows_buffer:
                    return
                with excel_lock:
                    for row in rows_buffer:
                        ws.append(row)
                    wb.save(excel_file_path)
                rows_buffer.clear()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Using tqdm for progress bar
                future_to_entry = {executor.submit(self.evaluate_single_entry, entry): entry for entry in selected_qnas}
                for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="Evaluating Q&A entries"):
                    entry = future_to_entry[future]
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
                            result.get('Gemini Correct', False)
                        ]
                        rows_buffer.append(row)

                        if len(rows_buffer) >= buffer_size:
                            flush_rows_buffer()

                    except Exception as e:
                        logging.error(f"Error processing entry: {e}", exc_info=True)

            # Flush any remaining rows
            flush_rows_buffer()
            wb.close()

            # Calculate accuracies
            rag_accuracy = (rag_correct / processed) * 100.0 if processed > 0 else 0.0
            gemini_accuracy = (gemini_correct / processed) * 100.0 if processed > 0 else 0.0

            logging.info(f"Evaluation completed. RAG Accuracy: {rag_accuracy}%, Gemini Accuracy: {gemini_accuracy}%")

            return rag_accuracy, gemini_accuracy

        except IsADirectoryError as e:
            logging.error(e)
            raise
        except Exception as e:
            logging.error(f"Failed during parallel evaluation: {e}", exc_info=True)
            raise

    def visualize_accuracies(self, rag_accuracy: float, gemini_accuracy: float, save_path: Optional[str] = None):
        """
        Visualizes the accuracies of RAG and Gemini models.

        Args:
            rag_accuracy (float): RAG model accuracy.
            gemini_accuracy (float): Gemini model accuracy.
            save_path (Optional[str], optional): Path to save the plot image. Defaults to None.
        """
        try:
            models = ['RAG Pipeline', 'Raw Gemini LLM']
            accuracies = [rag_accuracy, gemini_accuracy]
            colors = ['blue', 'green']

            plt.figure(figsize=(8, 6))
            bars = plt.bar(models, accuracies, color=colors)
            plt.ylim(0, 100)
            plt.ylabel('Accuracy (%)')
            plt.title('Model Accuracy Comparison')

            # Annotate bars with accuracy values
            for bar, acc in zip(bars, accuracies):
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{acc:.2f}%", ha='center', va='bottom')

            if save_path:
                plt.savefig(save_path)
                logging.info(f"Accuracy comparison plot saved to '{save_path}'.")
            plt.close()
        except Exception as e:
            logging.error(f"Failed to generate accuracy plot: {e}", exc_info=True)
            raise