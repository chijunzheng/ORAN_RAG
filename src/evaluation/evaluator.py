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

from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput

from src.vector_search.searcher import VectorSearcher
from src.vector_search.reranker import Reranker

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
        num_neighbors: int,
        reranker: Reranker
    ):
        """
        Initializes the Evaluator with necessary configurations.
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.embeddings_path = embeddings_path
        self.qna_dataset_path = qna_dataset_path
        self.index_endpoint_display_name = index_endpoint_display_name
        self.deployed_index_id = deployed_index_id
        self.vector_searcher = vector_searcher
        self.reranker = reranker
        self.num_neighbors = num_neighbors

        # Initialize the generative model (for both hypothetical and final answers)
        self.generative_model = GenerativeModel("gemini-1.5-flash-002")
        self.generation_config = GenerationConfig(
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.9),
            max_output_tokens=generation_config.get('max_output_tokens', 1000),
        )

        self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
        self.bucket = self.storage_client.bucket(self.bucket_name)

        logging.info("Evaluator initialized successfully.")

    def upload_qna_dataset_to_gcs(self, local_file_path: str, gcs_file_name: str):
        """
        Uploads the Q&A dataset to GCS.
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
                    # You can adjust your dataset parsing as needed:
                    if isinstance(qna_entry, dict):
                        question = qna_entry.get("question")
                        choices = qna_entry.get("choices")
                        correct_answer = qna_entry.get("answer")
                        if question and choices and correct_answer:
                            qna_dataset.append([question, choices, correct_answer])
                        else:
                            logging.warning(f"Line {line_number}: Missing fields.")
                    elif isinstance(qna_entry, list) and len(qna_entry) == 3:
                        qna_dataset.append(qna_entry)
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
        Extracts the choice number from the model's answer (1, 2, 3, or 4).
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
        Directly queries the raw Gemini LLM without retrieval/HyDE, for comparison.
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
        question: str,
        choices: List[str],
        chunks: List[Dict]
    ) -> Content:
        """
        Generates prompt content for the final LLM call (used in the RAG pipeline).
        """
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk['content']}" for i, chunk in enumerate(chunks)])
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
        {question}

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

    def generate_hypothetical_answer(self, question: str) -> str:
        """
        Generates a concise hypothetical answer for the user query (HyDE step).
        """
        try:
            prompt_text = f"Provide a concise hypothetical answer for the query:\n\n{question}\n\nAnswer:"
            user_prompt = Content(
                role="user",
                parts=[Part.from_text(prompt_text)]
            )
            response = self.generative_model.generate_content(
                user_prompt, generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error generating hypothetical answer: {e}", exc_info=True)
            return ""

    def query_rag_pipeline(self, question: str, choices: List[str]) -> str:
        """
        Refactored RAG pipeline using HyDE:
          1) Generate a hypothetical short answer from the question.
          2) Vector search for the query (top 30).
          3) Vector search for the hypothetical answer (top 30).
          4) Combine them, then rerank top 20.
          5) Generate a final answer with the top 20 chunks.
        """
        try:
            # 1) Hypothetical answer
            hypothetical_answer = self.generate_hypothetical_answer(question)

            # 2) Retrieve 30 chunks for the original query
            retrieved_chunks_query = self.vector_searcher.vector_search(
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
                query_text=question,
                num_neighbors=30
            )
            # 3) Retrieve 30 chunks for the hypothetical answer (if any)
            retrieved_chunks_hypo = []
            if hypothetical_answer:
                retrieved_chunks_hypo = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=hypothetical_answer,
                    num_neighbors=30
                )

            # Combine them
            combined_chunks = retrieved_chunks_query + retrieved_chunks_hypo
            if not combined_chunks:
                logging.warning("No chunks retrieved (query + hypothetical).")
                return "No relevant information found."

            # 4) Rerank to top 20
            reranked_chunks = self.reranker.rerank(query=question, records=combined_chunks)
            top_20_chunks = reranked_chunks[:20]
            if not top_20_chunks:
                logging.warning("Reranking returned no results.")
                return "No relevant information found after reranking."

            # 5) Generate final response with top 20 chunks
            prompt_content = self.generate_prompt_content_evaluation(question, choices, top_20_chunks)
            final_answer = self.safe_generate_content(prompt_content)
            return final_answer.strip() if final_answer else "Error"
        except Exception as e:
            logging.error(f"Error in HyDE RAG pipeline for question '{question}': {e}", exc_info=True)
            return "Error"

    def evaluate_single_entry(self, entry: List[str], delay: float = 0.5) -> Dict:
        """
        Evaluates a single Q&A entry using both HyDE RAG pipeline and direct LLM.
        """
        try:
            time.sleep(delay)
            question, choices, correct_str = entry
            correct_choice = correct_str.strip()

            # Query RAG (HyDE pipeline)
            rag_full_answer = self.query_rag_pipeline(question, choices)
            rag_pred_choice = self.extract_choice_from_answer(rag_full_answer)
            rag_correct = (rag_pred_choice == correct_choice)

            # Query Gemini (no RAG, direct LLM)
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

    def evaluate_models_parallel(
        self,
        qna_dataset: List[List[str]],
        num_questions: int,
        excel_file_path: str,
        max_workers: int = 1
    ) -> Tuple[float, float]:
        """
        Evaluates models in parallel on a subset of the Q&A dataset, saving results to an Excel file.
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
            future_to_result = {
                executor.submit(self.evaluate_single_entry, entry, 0.5): entry
                for entry in qna_subset
            }
            for future in tqdm(as_completed(future_to_result), total=len(future_to_result), desc="Evaluating"):
                result = future.result()
                processed += 1

                if result['RAG Correct']:
                    rag_correct += 1
                if result['Gemini Correct']:
                    gemini_correct += 1

                row = [
                    result['Question'],
                    result['Correct Answer'],
                    result['RAG Predicted Answer'],
                    str(result['RAG Correct']),
                    result['Gemini Predicted Answer'],
                    str(result['Gemini Correct'])
                ]
                rows_buffer.append(row)
                # Flush rows in batches
                if len(rows_buffer) >= buffer_size:
                    flush_rows_buffer()

        # Flush any remaining rows
        flush_rows_buffer()

        # Compute accuracies
        rag_accuracy = (rag_correct / processed) * 100 if processed else 0
        gemini_accuracy = (gemini_correct / processed) * 100 if processed else 0

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