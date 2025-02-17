import logging
import random
import time
from typing import List, Dict
from vertexai.generative_models import Content, GenerationConfig, Part
from src.chatbot.rat_processor import RATProcessor  

class RATProcessorEval(RATProcessor):
    """
    Evaluation-specific RATProcessor variant.
    
    This class follows the RAT (Retrieval Augmented Thoughts) approach but is tailored for evaluation:
      - It accepts an optional list of multiple-choice options and injects them into the initial prompt.
      - It then performs iterative retrieval and CoT revision (using generate_retrieval_query, retrieve_context, and revise_cot).
      - Finally, process_query returns the final chain-of-thought, which in evaluation is further used
        to select the correct answer.
    """
    def __init__(
        self,
        vector_searcher,
        llm,
        generation_config: GenerationConfig,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        num_iterations: int = 3,
        reranker = None
    ):
        super().__init__(vector_searcher, llm, generation_config, index_endpoint_display_name, deployed_index_id, num_iterations=num_iterations)
        self.reranker = reranker  # Not necessarily used here but can be passed if needed
        

    def _safe_generate_content(self, content: Content, config: GenerationConfig = None, retries: int = 10, backoff_factor: int = 2, max_wait: int = 300) -> str:
        """
        Internal safe generation method tailored for RAT evaluation.
        If a quota or rate-limit error is encountered, waits a fixed long period (e.g. 300 seconds)
        before retrying indefinitely.
        For other errors, uses exponential backoff with jitter.
        """
        if config is None:
            config = self.cot_generation_config
        attempt = 0
        wait_time = 1
        while True:
            try:
                response = self.llm.generate_content(content, generation_config=config)
                response_text = response.text.strip()
                if not response_text or response_text.lower() == "error":
                    raise ValueError("Empty or error response.")
                return response_text
            except Exception as e:
                error_text = str(e).lower()
                # Detect quota or rate limit errors.
                if any(keyword in error_text for keyword in ["429", "quota", "rate limit", "resourceexhausted"]):
                    fixed_wait = 120 
                    logging.warning(f"[RAT Eval] Quota or rate limit error detected: {e}. Waiting for {fixed_wait} seconds before retrying.")
                    time.sleep(fixed_wait)
                    # Reset attempt and wait_time after quota error.
                    attempt = 0
                    wait_time = 1
                    continue
                else:
                    attempt += 1
                    if attempt >= retries:
                        logging.error(f"[RAT Eval] Final attempt {attempt}: {e}. Returning empty string as fallback.")
                        return ""
                    jitter = random.uniform(0, 1)
                    sleep_time = min(wait_time * backoff_factor, max_wait)
                    logging.warning(f"[RAT Eval] Attempt {attempt}: {e}. Retrying in {sleep_time + jitter:.2f} seconds.")
                    time.sleep(sleep_time + jitter)
                    wait_time *= backoff_factor

    def generate_initial_cot(self, query: str, choices: List[str] = None) -> str:
        """
        Generates an initial chain-of-thought (CoT) for the query.
        When choices are provided (evaluation mode), they are injected into the prompt.
        """
        if choices:
            choices_str = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
            prompt = f"""
You are an expert in complex problem solving for multiple-choice questions.
Generate a detailed initial chain-of-thought that outlines your step-by-step reasoning.
Make sure to explicitly consider the following answer options in your reasoning.
Do not worry if some steps are uncertain; provide a clear draft of your thought process.

Question: {query}

Answer Options:
{choices_str}

Chain-of-Thought:"""
        else:
            prompt = f"""
You are an expert in complex problem solving and iterative reasoning.
For the following question, generate a detailed initial chain-of-thought that outlines your step-by-step reasoning.
Do not worry if some steps are uncertain or incomplete; provide a clear draft of your thought process.

Question: {query}

Chain-of-Thought:"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        initial_cot = self._safe_generate_content(content, config=self.cot_generation_config)
        logging.info(f"[Eval] Initial CoT (first 1000 chars): {initial_cot[:1000]}...")
        return initial_cot

    def generate_retrieval_query(self, current_cot: str, query: str) -> str:
        """
        Uses the current chain-of-thought and the original query to generate a concise retrieval query.
        """
        prompt = f"""
            You are an expert at crafting precise retrieval queries.
            Based on the following chain-of-thought and the original question, generate a concise retrieval query
            that targets any uncertainties or factual gaps in your reasoning.

            Current Chain-of-Thought:
            "{current_cot}"

            Original Question:
            "{query}"

            Retrieval Query (output only plain text):"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        retrieval_query = self._safe_generate_content(content, config=self.cot_generation_config)
        logging.info(f"[Eval] Retrieval Query: {retrieval_query}")
        return retrieval_query

    def retrieve_context(self, retrieval_query: str) -> str:
        """
        Uses the vector searcher to retrieve ORAN document chunks for the retrieval query.
        Returns the concatenated text of the top results.
        """
        if not self.vector_searcher:
            logging.error("No vector_searcher available for retrieval.")
            return ""
        retrieved = self.vector_searcher.vector_search(
            query_text=retrieval_query,
            num_neighbors=5,
            index_endpoint_display_name=self.index_endpoint_display_name,
            deployed_index_id=self.deployed_index_id
        )
        context_text = "\n".join(chunk.get("content", "") for chunk in retrieved)
        logging.info(f"[Eval] Retrieved context (first 1000 chars): {context_text[:1000]}...")
        return context_text

    def revise_cot(self, current_cot: str, retrieved_context: str, query: str) -> str:
        """
        Revises the current chain-of-thought by incorporating the retrieved context.
        """
        prompt = f"""
            You are an expert in iterative reasoning refinement. Review your current chain-of-thought along with the retrieved factual context.
            Revise your chain-of-thought to correct any errors and incorporate missing details.
            Provide only the updated chain-of-thought as plain text.

            Current Chain-of-Thought:
            "{current_cot}"

            Retrieved Context:
            "{retrieved_context}"

            Original Question:
            "{query}"

            Revised Chain-of-Thought:"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        revised_cot = self._safe_generate_content(content, config=self.cot_generation_config)
        logging.info(f"[Eval] Revised CoT (first 1000 chars): {revised_cot[:1000]}...")
        return revised_cot

    def process_query(self, query: str, conversation_history: List[Dict], choices: List[str] = None) -> str:
        """
        Full evaluation pipeline:
          1. Generate an initial chain-of-thought that incorporates the question and answer choices (if provided).
          2. Iteratively refine the chain-of-thought.
          3. Return the final revised chain-of-thought.
        """
        current_cot = self.generate_initial_cot(query, choices=choices)
        for i in range(self.num_iterations):
            logging.info(f"[Eval] RAT iteration {i+1} starting.")
            retrieval_query = self.generate_retrieval_query(current_cot, query)
            retrieved_context = self.retrieve_context(retrieval_query)
            current_cot = self.revise_cot(current_cot, retrieved_context, query)
        return current_cot