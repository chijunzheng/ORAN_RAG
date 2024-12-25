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


class Evaluator:
    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: str,
        embeddings_path: str,
        qna_dataset_path: str,
        index_endpoint_display_name: str,
        deployed_index_id: str,
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
        self.index_endpoint_display_name = index_endpoint_display_name
        self.deployed_index_id = deployed_index_id

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
        backoff_factor: int = 2,
        max_wait: int = 30
    ) -> str:
        """
        Generates content with exponential backoff and jitter.

        Args:
            content (Content): Content object for the generative model.
            retries (int, optional): Number of retry attempts. Defaults to 10.
            backoff_factor (int, optional): Backoff multiplier. Defaults to 2.
            max_wait (int, optional): Maximum wait time. Defaults to 30.

        Returns:
            str: Generated text or "Error" upon failure.
        """
        wait_time = 1
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

    def extract_choice_from_answer(self, answer_text: str) -> str:
        """
        Extracts the choice number from the model's answer.

        Args:
            answer_text (str): The model's response.

        Returns:
            str: Extracted choice number or None.
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
            parts=[Part.from_text(prompt_text)]
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
            parts=[Part.from_text(prompt_text)]
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
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
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