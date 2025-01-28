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
    
    # ----------------------------------------------------------------------
    # 1) LOAD Q&A DATASET
    # ----------------------------------------------------------------------
    def load_qna_dataset_from_gcs(self) -> List[List[str]]:
        """
        Loads the Q&A dataset from GCS.

        Returns:
            List[List[str]]: Q&A entries as [question, choices, correct_answer].
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
                    # Must yield [question, choices, correct_answer]
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

    # ----------------------------------------------------------------------
    # 2) STEP-BACK: Generate a single core concept for a question
    # ----------------------------------------------------------------------
    def generate_eval_core_concept(self, question: str) -> str:
        """
        Calls the LLM to generate a single "core ORAN concept" behind the question.
        We'll keep it minimal. 
        You can further lower temperature or do concept normalization as needed.
        """
        prompt_text = f"""
        You are an O-RAN expert. Analyze the question below
        and identify the single core concept needed to answer it.
        
        Question: {question}

        Instructions:
        1. Provide a concise concept or principle behind this question.
        2. Do not provide further explanation—only briefly describe the concept.

        Concept:
        """
        prompt_content = Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )

        try:
            # Use safe_generate_content
            raw_text = self.safe_generate_content(prompt_content)
            if not raw_text:
                return "O-RAN Architecture (General)"
            return raw_text
        except Exception as e:
            logging.error(f"Error generating eval core concept: {e}", exc_info=True)
            return "O-RAN Architecture (General)"

    # ----------------------------------------------------------------------
    # 3) MULTI-QUERY GENERATION (Anchored to the Concept)
    # ----------------------------------------------------------------------
    def generate_concept_anchored_queries(self, user_query: str, core_concept: str, num_variations: int = 3) -> List[str]:
        """
        Generates additional queries that revolve around the identified core concept,
        exploring the user's question from different angles but anchored to the concept.
        """
        prompt_text = f"""
        You are an O-RAN expert. The user has asked: "{user_query}"
        The core concept is: "{core_concept}"

        Generate {num_variations} unique and diverse queries that explore or elaborate on this core concept,
        while remaining relevant to the user's question. Each query should:
          - Reflect or incorporate the concept "{core_concept}"
          - Provide a different perspective or subtopic
        Ensure they are distinct yet aligned with the user's overall intent.

        Similar Queries:
        1.
        2.
        3.
        """
        prompt_content = Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )

        try:
            # Use safe_generate_content
            raw_text = self.safe_generate_content(prompt_content)

            lines = raw_text.splitlines()
            queries = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith(("1.", "2.", "3.", "4.")):
                    # parse after the digit
                    query_part = line_stripped.split(".", 1)[-1].strip()
                    if query_part:
                        queries.append(query_part)
            if not queries:
                queries = [user_query]  # fallback
            return queries
        except Exception as e:
            logging.error(f"Error generating concept-anchored queries: {e}", exc_info=True)
            return [user_query]

    # ----------------------------------------------------------------------
    # 4) PROMPT & ANSWER: Evaluate a single question using Step-Back + Multi-Query
    # ----------------------------------------------------------------------
    def query_rag_stepback_multiquery(self, question: str, choices: List[str]) -> str:
        """
        Step-Back + Multi-Query for multiple-choice questions:
         1) Generate core concept
         2) Generate multi queries anchored to concept
         3) Retrieve + Rerank
         4) Build final prompt
         5) Let LLM pick the correct choice
        """
        try:
            # (A) Generate the core concept
            core_concept = self.generate_eval_core_concept(question)
            logging.debug(f"[Step-Back] Concept: {core_concept}")

            # (B) Generate multi queries anchored to the concept
            anchored_queries = self.generate_concept_anchored_queries(question, core_concept, num_variations=3)
            all_queries = [question] + anchored_queries
            logging.debug(f"All queries: {all_queries}")

            # (C) Retrieve from each query & merge
            all_chunks = []
            for q_variant in all_queries:
                partial_chunks = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=q_variant,
                    num_neighbors=self.num_neighbors  # e.g., 30
                )
                all_chunks.extend(partial_chunks)

            if not all_chunks:
                logging.warning("No chunks retrieved for any query variant.")
                return "No relevant information found."

            # (D) Rerank the merged chunks by core concept
            rerank_method = core_concept + " " + question
            reranked_chunks = self.reranker.rerank(query=rerank_method, records=all_chunks)
            if not reranked_chunks:
                logging.warning("Reranking returned no results.")
                return "No relevant information found after reranking."

            top_k = 20
            top_chunks = reranked_chunks[:top_k]

            # (E) Build final multiple-choice prompt, asking for the correct choice
            answer_text = self.generate_mc_prompt_and_answer(question, choices, top_chunks)
            return answer_text.strip() if answer_text else "Error"
        except Exception as e:
            logging.error(f"Error in step-back multi-query pipeline for Q: '{question}' => {e}", exc_info=True)
            return "Error"

    # ----------------------------------------------------------------------
    # 5) Build the final multiple-choice prompt + answer
    # ----------------------------------------------------------------------
    def generate_mc_prompt_and_answer(self, question: str, choices: List[str], chunks: List[Dict]) -> str:
        """
        Builds a final multiple-choice prompt with the top reranked chunks, 
        then asks the LLM to pick the best answer choice.
        """
        context_text = "\n\n".join([
            f"Chunk {i+1}:\n{chunk['content']}" for i, chunk in enumerate(chunks)
        ])
        choices_text = "\n".join(choices)

        prompt_text = f"""
        You are an O-RAN expert. Use the context below to determine the correct answer choice 
        for the multiple-choice question.

        Context:
        {context_text}

        Question:
        {question}

        Choices:
        {choices_text}

        Please provide the correct answer choice number (1, 2, 3, or 4) at the start of your answer, 
        such as "The correct answer is: X", and optionally a brief rationale.
        """

        prompt_content = Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )

        try:
            # Use safe_generate_content
            raw_text = self.safe_generate_content(prompt_content)
            return raw_text
    
        except Exception as e:
            logging.error(f"Error generating MC final answer: {e}", exc_info=True)
            return "Error"

    # ----------------------------------------------------------------------
    # 6) Evaluate a single Q&A entry
    # ----------------------------------------------------------------------
    def evaluate_single_entry(self, entry: List[str], delay: float = 0.5) -> Dict:
        """
        Evaluates a single Q&A entry with Step-Back + Multi-Query RAG.
        Expects entry = [question, choices, correct_answer].
        Returns a dictionary of results.
        """
        try:
            time.sleep(delay)
            question, choices, correct_str = entry
            correct_choice = correct_str.strip()

            # Query RAG with Step-Back + Multi-Query
            rag_answer = self.query_rag_stepback_multiquery(question, choices)
            rag_pred_choice = self.extract_choice_from_answer(rag_answer)
            rag_correct = (rag_pred_choice == correct_choice)

            # Optionally, also evaluate a "direct LLM" approach (like Gemini only) if you want to compare:
            gemini_answer = self.query_gemini_llm(question, choices)
            gemini_pred_choice = self.extract_choice_from_answer(gemini_answer)
            gemini_correct = (gemini_pred_choice == correct_choice)

            return {
                'Question': question,
                'Correct Answer': correct_choice,
                'RAG Predicted Answer': rag_answer,
                'RAG Correct': rag_correct,
                'Gemini Predicted Answer': gemini_answer,
                'Gemini Correct': gemini_correct
            }
        except Exception as e:
            logging.error(f"Error evaluating single entry: {e}", exc_info=True)
            return {
                'Question': entry[0],
                'Correct Answer': entry[2].strip(),
                'RAG Predicted Answer': "Error",
                'RAG Correct': False,
                'Gemini Predicted Answer': "Error",
                'Gemini Correct': False,
            }

    # ----------------------------------------------------------------------
    # 7) Extract the choice from an LLM answer
    # ----------------------------------------------------------------------
    def extract_choice_from_answer(self, answer_text: str) -> str:
        """
        Extracts the choice number (1-4) from the model's answer.
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

    # ----------------------------------------------------------------------
    # 8) Query Gemini LLM directly (no RAG) - optional baseline
    # ----------------------------------------------------------------------
    def query_gemini_llm(self, question: str, choices: List[str]) -> str:
        """
        Queries the raw Gemini LLM for an answer (no RAG).
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
        try:
            # Use safe_generate_content
            raw_text = self.safe_generate_content(user_prompt)
            return raw_text
        except Exception as e:
            logging.error(f"Error querying Gemini LLM directly: {e}", exc_info=True)
            return "Error"

    # ----------------------------------------------------------------------
    # 9) Evaluate the entire dataset of 3243 Q&A
    # ----------------------------------------------------------------------
    def evaluate_models_parallel(
        self,
        qna_dataset: List[List[str]],
        num_questions: int,
        excel_file_path: str,
        plot_save_path: str,
        max_workers: int = 1
    ) -> Tuple[float, float]:
        """
        Evaluates RAG (Step-Back + Multi-Query) vs. direct Gemini LLM on 'num_questions' from 'qna_dataset'.
        Writes results to Excel, and optionally saves a plot comparing accuracies.

        Returns: (rag_accuracy, gemini_accuracy)
        """
        # Subset dataset
        subset_data = qna_dataset[:num_questions]

        # Create workbook and sheet
        wb = Workbook()
        ws = wb.active
        ws.title = "Evaluation Results"
        headers = [
            "Question",
            "Correct Answer",
            "RAG Predicted Answer",
            "RAG Correct",
            "Gemini Predicted Answer",
            "Gemini Correct"
        ]
        ws.append(headers)

        rag_correct_count = 0
        gemini_correct_count = 0
        processed = 0

        # Use progress bar with tqdm
        results_buffer = []
        buffer_size = 50  # flush to excel in batches
        import threading
        excel_lock = threading.Lock()

        def flush_to_excel():
            with excel_lock:
                for row in results_buffer:
                    ws.append(row)
                wb.save(excel_file_path)
            results_buffer.clear()

        # Evaluate in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_single_entry, entry): idx
                for idx, entry in enumerate(subset_data, 1)
            }

            with tqdm(total=len(futures), desc="Evaluating Q&A", ncols=100) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        res = future.result()
                        question = res['Question']
                        correct_ans = res['Correct Answer']
                        rag_pred = res['RAG Predicted Answer']
                        rag_is_correct = res['RAG Correct']
                        gemini_pred = res['Gemini Predicted Answer']
                        gemini_is_correct = res['Gemini Correct']

                        if rag_is_correct:
                            rag_correct_count += 1
                        if gemini_is_correct:
                            gemini_correct_count += 1
                        processed += 1

                        row = [
                            question,
                            correct_ans,
                            rag_pred,
                            "Yes" if rag_is_correct else "No",
                            gemini_pred,
                            "Yes" if gemini_is_correct else "No"
                        ]
                        results_buffer.append(row)
                        if len(results_buffer) >= buffer_size:
                            flush_to_excel()
                    except Exception as e:
                        logging.error(f"Error in future result: {e}", exc_info=True)
                    pbar.update(1)

        # Flush any leftover
        if results_buffer:
            flush_to_excel()

        wb.close()

        rag_accuracy = (rag_correct_count / processed) * 100.0 if processed else 0
        gemini_accuracy = (gemini_correct_count / processed) * 100.0 if processed else 0
        logging.info(f"RAG Accuracy: {rag_accuracy:.2f}% | Gemini Accuracy: {gemini_accuracy:.2f}%")

        # Optionally plot
        self.plot_accuracies(rag_accuracy, gemini_accuracy, plot_save_path)

        return (rag_accuracy, gemini_accuracy)

    def plot_accuracies(self, rag_acc: float, gemini_acc: float, save_path: str):
        """
        Creates a bar plot comparing RAG vs. Gemini accuracy, saves to 'save_path'.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,5))
        models = ["RAG (Step-Back+MQ)", "Gemini LLM"]
        accs = [rag_acc, gemini_acc]
        bars = plt.bar(models, accs, color=["#4CAF50","#2196F3"])
        plt.ylim(0, 100)
        plt.ylabel("Accuracy (%)")
        plt.title("RAG vs Gemini Accuracy")

        for i, v in enumerate(accs):
            plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

        try:
            plt.savefig(save_path, dpi=150)
            plt.close()
            logging.info(f"Saved accuracy comparison plot to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot: {e}", exc_info=True)