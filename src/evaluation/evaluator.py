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

        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud region.
            bucket_name (str): GCS bucket name.
            embeddings_path (str): Path within the bucket where embeddings are stored.
            qna_dataset_path (str): Path to the Q&A dataset in GCS.
            index_endpoint_display_name (str): Display name of the index endpoint.
            deployed_index_id (str): ID of the deployed index.
            generation_config (Dict): Configuration for text generation parameters.
            vector_searcher (VectorSearcher): Instance of VectorSearcher for performing vector searches.
            credentials: Google Cloud credentials object.
            num_neighbors (int): Number of neighbors to retrieve per similar query.
            reranker (Reranker): Instance of Reranker for reranking results.
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.embeddings_path = embeddings_path
        self.qna_dataset_path = qna_dataset_path
        self.vector_searcher = vector_searcher  
        self.generation_config = GenerationConfig(
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.9),
            max_output_tokens=generation_config.get('max_output_tokens', 1000),
        )
        self.generative_model = GenerativeModel("gemini-1.5-flash-002")
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

        context = "\n\n".join([f"Chunk {i+1}:\n{chunk['content']}" for i, chunk in enumerate(chunks)])
        choices_text = "\n".join(choices)
        prompt_text = f"""
        You are an expert in O-RAN systems. Utilize the context to provide the correct and logical answer choice to the user's queries.

        Instruction:
        Please provide the correct answer choice number (1, 2, 3, or 4) only.

        Context:
        {context}

        Question:
        {query}

        Choices:
        {choices_text}


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
            # Step 1: Generate Similar Queries
            similar_queries = self.generate_similar_queries(question, num_similar=3)
            logging.info(f"Generated {len(similar_queries)} similar queries for evaluation.")

            # Step 2: Include Original Query with Similar Queries
            all_queries = [question] + similar_queries  # Original query + similar queries
            logging.info(f"Including original query along with similar queries for vector search.")

            # Step 3: Perform Vector Search for Each Similar Query
            all_retrieved_chunks = []
            for idx, sim_query in enumerate(all_queries):
                retrieved_chunks = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=sim_query,
                    num_neighbors=30  # 30 chunks per similar query
                )
                logging.info(f"Retrieved {len(retrieved_chunks)} chunks for similar query {idx + 1}: '{sim_query}'")
                all_retrieved_chunks.extend(retrieved_chunks)

            logging.info(f"Total retrieved chunks from all similar queries: {len(all_retrieved_chunks)}")

            if not all_retrieved_chunks:
                logging.warning("No chunks retrieved for any similar queries.")
                return "No relevant information found."

            # Step 3: Rerank the Retrieved Chunks
            reranked_chunks = self.reranker.rerank(
                query=question,
                records=all_retrieved_chunks
            )
            logging.info(f"Reranked to top {len(reranked_chunks)} chunks.")

            if not reranked_chunks:
                logging.warning("Reranking returned no results.")
                return "No relevant information found after reranking."

            # Step 4: Generate Prompt with Reranked Chunks
            prompt_content = self.generate_prompt_content_evaluation(question, choices, reranked_chunks)

            # Step 5: Generate Response from RAG Pipeline
            assistant_response = self.safe_generate_content(prompt_content)

            return assistant_response.strip() if assistant_response else "Error"

        except Exception as e:
            logging.error(f"Error in RAG pipeline for question '{question}': {e}", exc_info=True)
            return "Error"

    def generate_similar_queries(self, original_query: str, num_similar: int = 3) -> List[str]:
        """
        Generates similar queries to the original question using the Gemini LLM.

        Args:
            original_query (str): The original question.
            num_similar (int, optional): Number of similar queries to generate. Defaults to 3.

        Returns:
            List[str]: List of similar queries.
        """
        prompt_text = f"""
        Generate {num_similar} unique and diverse queries that address the same underlying intent as the following question but explore different aspects, perspectives, or use varied terminology. Ensure that each query focuses on a distinct facet of the topic to maximize the diversity of information retrieved.

        Original Query: "{original_query}"

        Similar Queries:
        1.
        2.
        3.
        """
        user_prompt = Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )
        similar_queries_text = self.safe_generate_content(user_prompt)
        similar_queries = []
        for line in similar_queries_text.split('\n'):
            if line.strip().startswith(tuple(str(i) + '.' for i in range(1, num_similar + 1))):
                query = line.split('.', 1)[1].strip()
                if query:
                    similar_queries.append(query)
        logging.debug(f"Generated similar queries: {similar_queries}")
        return similar_queries

    def evaluate_models_parallel(
        self,
        qna_dataset: List[List[str]],
        num_questions: int,
        excel_file_path: str,
        max_workers: int = 1
    ) -> Tuple[float, float]:
        """
        Evaluates both the RAG pipeline and the raw Gemini LLM on the Q&A dataset.

        Args:
            qna_dataset (List[List[str]]): List of Q&A entries as [question, choices, correct_answer].
            num_questions (int): Number of questions to evaluate.
            excel_file_path (str): Path to save the evaluation results Excel file.
            max_workers (int, optional): Maximum number of worker threads. Defaults to 1.

        Returns:
            Tuple[float, float]: RAG pipeline accuracy and Gemini LLM accuracy.
        """
        selected_qna = random.sample(qna_dataset, min(num_questions, len(qna_dataset)))
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qna = {
                executor.submit(self.evaluate_single_question, qna): qna for qna in selected_qna
            }

            for future in tqdm(as_completed(future_to_qna), total=len(future_to_qna), desc="Evaluating Q&A"):
                qna = future_to_qna[future]
                try:
                    rag_correct, gemini_correct = future.result()
                    results.append({
                        'Question': qna[0],
                        'Choices': qna[1],
                        'Correct Answer': qna[2],
                        'RAG Answer Correct': rag_correct,
                        'Gemini Answer Correct': gemini_correct
                    })
                except Exception as e:
                    logging.error(f"Error evaluating question '{qna[0]}': {e}", exc_info=True)

        # Save results to Excel
        self.save_results_to_excel(results, excel_file_path)

        # Calculate accuracies
        rag_accuracy = (sum(1 for r in results if r['RAG Answer Correct']) / len(results)) * 100 if results else 0
        gemini_accuracy = (sum(1 for r in results if r['Gemini Answer Correct']) / len(results)) * 100 if results else 0

        return rag_accuracy, gemini_accuracy

    def evaluate_single_question(self, qna: List[str]) -> Tuple[bool, bool]:
        """
        Evaluates a single Q&A entry for both RAG and Gemini models.

        Args:
            qna (List[str]): A Q&A entry as [question, choices, correct_answer].

        Returns:
            Tuple[bool, bool]: RAG pipeline correctness and Gemini LLM correctness.
        """
        question, choices, correct_answer = qna

        # Evaluate RAG Pipeline
        rag_response = self.query_rag_pipeline(question, choices)
        rag_choice = self.extract_choice_from_answer(rag_response)
        rag_correct = (rag_choice == correct_answer)

        # Evaluate Gemini LLM
        gemini_response = self.query_gemini_llm(question, choices)
        gemini_choice = self.extract_choice_from_answer(gemini_response)
        gemini_correct = (gemini_choice == correct_answer)

        return rag_correct, gemini_correct

    def save_results_to_excel(self, results: List[Dict], excel_file_path: str):
        """
        Saves the evaluation results to an Excel file.

        Args:
            results (List[Dict]): List of evaluation results.
            excel_file_path (str): Path to save the Excel file.
        """
        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Evaluation Results"

            # Write headers
            headers = ['Question', 'Choices', 'Correct Answer', 'RAG Answer Correct', 'Gemini Answer Correct']
            ws.append(headers)

            # Write data rows
            for result in results:
                ws.append([
                    result['Question'],
                    "\n".join(result['Choices']),
                    result['Correct Answer'],
                    "Yes" if result['RAG Answer Correct'] else "No",
                    "Yes" if result['Gemini Answer Correct'] else "No"
                ])

            wb.save(excel_file_path)
            logging.info(f"Saved evaluation results to Excel at {excel_file_path}")
        except Exception as e:
            logging.error(f"Failed to save results to Excel: {e}", exc_info=True)
            raise

    def visualize_accuracies(self, rag_accuracy: float, gemini_accuracy: float, save_path: str = None):
        """
        Visualizes the accuracies of both models using a bar chart.

        Args:
            rag_accuracy (float): RAG pipeline accuracy percentage.
            gemini_accuracy (float): Gemini LLM accuracy percentage.
            save_path (str): Path to save the plot image. Defaults to None.
        """
        try:
            models = ['RAG Pipeline', 'Gemini LLM']
            accuracies = [rag_accuracy, gemini_accuracy]

            plt.figure(figsize=(8, 6))
            bars = plt.bar(models, accuracies, color=['#1a73e8', '#34a853'])
            plt.ylim(0, 100)
            plt.ylabel('Accuracy (%)')
            plt.title('Model Accuracies')

            # Attach accuracy labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.2f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                logging.info(f"Saved accuracy plot to {save_path}")
            plt.show()
        except Exception as e:
            logging.error(f"Failed to visualize accuracies: {e}", exc_info=True)
            raise

    def evaluate_models_sequential(
        self,
        qna_dataset: List[List[str]],
        num_questions: int,
        excel_file_path: str
    ) -> Tuple[float, float]:
        """
        Sequentially evaluates both the RAG pipeline and the raw Gemini LLM on the Q&A dataset.

        Args:
            qna_dataset (List[List[str]]): List of Q&A entries as [question, choices, correct_answer].
            num_questions (int): Number of questions to evaluate.
            excel_file_path (str): Path to save the evaluation results Excel file.

        Returns:
            Tuple[float, float]: RAG pipeline accuracy and Gemini LLM accuracy.
        """
        selected_qna = random.sample(qna_dataset, min(num_questions, len(qna_dataset)))
        results = []

        for qna in tqdm(selected_qna, desc="Evaluating Q&A"):
            try:
                rag_correct, gemini_correct = self.evaluate_single_question(qna)
                results.append({
                    'Question': qna[0],
                    'Choices': qna[1],
                    'Correct Answer': qna[2],
                    'RAG Answer Correct': rag_correct,
                    'Gemini Answer Correct': gemini_correct
                })
            except Exception as e:
                logging.error(f"Error evaluating question '{qna[0]}': {e}", exc_info=True)

        # Save results to Excel
        self.save_results_to_excel(results, excel_file_path)

        # Calculate accuracies
        rag_accuracy = (sum(1 for r in results if r['RAG Answer Correct']) / len(results)) * 100 if results else 0
        gemini_accuracy = (sum(1 for r in results if r['Gemini Answer Correct']) / len(results)) * 100 if results else 0

        return rag_accuracy, gemini_accuracy