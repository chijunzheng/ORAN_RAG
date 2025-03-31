import logging
import random
import time
import os
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
        
        # Check if we're using the API key integration (google.generativeai package)
        self.using_api_key = 'google.generativeai' in str(type(self.llm))
        
        if self.using_api_key:
            # For google.generativeai, we'll set up generation parameters
            self.cot_generation_params = {
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
            # Import the generativeai module for configuration
            try:
                import google.generativeai as genai
                self.genai = genai
                self.cot_generation_config = genai.GenerationConfig(**self.cot_generation_params)
            except ImportError:
                logging.error("Failed to import google.generativeai module")
                self.cot_generation_config = None
        

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
                # Handle different API types
                if self.using_api_key:
                    # For google.generativeai, we can directly pass the text
                    if isinstance(content, Content):
                        # Extract the text from the Content object
                        prompt_text = ""
                        for part in content.parts:
                            if hasattr(part, 'text'):
                                prompt_text += part.text
                        
                        response = self.llm.generate_content(prompt_text, generation_config=config)
                    else:
                        # If it's already a string
                        response = self.llm.generate_content(content, generation_config=config)
                else:
                    # For vertexai.generative_models, we need to use the Content object
                    response = self.llm.generate_content(content, generation_config=config)
                    
                response_text = response.text.strip()
                if not response_text or response_text.lower() == "error":
                    raise ValueError("Empty or error response.")
                return response_text
            except Exception as e:
                error_text = str(e).lower()
                # Detect quota or rate limit errors.
                if any(keyword in error_text for keyword in ["429", "quota", "rate limit", "resourceexhausted"]):
                    fixed_wait = 60
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

    def generate_initial_cot(self, question: str, choices: List[str] = None) -> str:
        """
        Uses the same default prompt structure from rat_processor.py,
        but if 'choices' exist, inject them into the question.
        """
        if choices:
            joined_choices = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(choices))
            # Merge them into 'question'
            combined_query = (
                f"{question}\n\n"
                "Multiple Choice Options:\n"
                f"{joined_choices}\n\n"
            )
        else:
            combined_query = question

        prompt = f"""
        You are an expert in complex problem solving and iterative reasoning. 
        For the following question, generate a detailed initial chain-of-thought 
        that outlines your step-by-step reasoning. 
        Do not worry if some steps are uncertain or incomplete; 
        provide a clear draft of your thought process.

        Question: {question}

        Chain-of-Thought:
        """
        
        # Handle different API types
        if self.using_api_key:
            # For google.generativeai, we can directly pass the text
            initial_cot = self._safe_generate_content(prompt, config=self.cot_generation_config)
        else:
            # For vertexai.generative_models, we need to create a Content object
            content = Content(role="user", parts=[Part.from_text(prompt)])
            initial_cot = self._safe_generate_content(content, config=self.cot_generation_config)
            
        logging.info(f"Initial CoT (first 1000 chars): {initial_cot[:1000]}...")
        return initial_cot

    def generate_retrieval_query(self, current_cot: str, query: str) -> str:
        """
        Uses the current chain-of-thought and the original query to generate a concise retrieval query.
        The retrieval query should focus on the uncertainties or key factual gaps in the current chain-of-thought,
        so that external reliable information can help improve the answer.
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
            
        # Handle different API types
        if self.using_api_key:
            # For google.generativeai, we can directly pass the text
            retrieval_query = self._safe_generate_content(prompt, config=self.cot_generation_config)
        else:
            # For vertexai.generative_models, we need to create a Content object
            content = Content(role="user", parts=[Part.from_text(prompt)])
            retrieval_query = self._safe_generate_content(content, config=self.cot_generation_config)

        logging.info(f"Retrieval Query: {retrieval_query}")
        return retrieval_query


    def retrieve_context(self, retrieval_query: str) -> str:
        """
        Uses the vector searcher to retrieve ORAN document chunks for the retrieval query.
        Returns the concatenated text of the top results.
        """
        if not self.vector_searcher:
            logging.error("No vector_searcher available for retrieval.")
            return ""
        # Here we assume vector_searcher.vector_search(query_text, num_neighbors) is available.
        retrieved = self.vector_searcher.vector_search(
            query_text=retrieval_query, 
            num_neighbors=5,
            index_endpoint_display_name=self.index_endpoint_display_name,
            deployed_index_id=self.deployed_index_id,
            )
        # Build a text block that contains each chunk's metadata
        context_text = ""
        for i, chunk in enumerate(retrieved, start=1):
            doc_meta = chunk.get("metadata", {})
            doc_name = chunk.get("document_name", "N/A")
            version = doc_meta.get("version","unknown")
            workgroup = doc_meta.get("workgroup","unknown")
            subcat = doc_meta.get("subcategory","unknown")
            page = chunk.get("page_number","N/A")

            # Add chunk ID for clarity
            context_text += (
                f"--- Retrieved Chunk #{i} ---\n"
                f"Document Name: {doc_name}\n"
                f"Version: {version}\n"
                f"Workgroup: {workgroup}\n"
                f"Subcategory: {subcat}\n"
                f"Page: {page}\n\n"
                f"{chunk.get('content','No content')}\n\n"
            )
        logging.info(f"Retrieved context (first 1000 chars): {context_text[1000]}...")
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
            
        # Handle different API types
        if self.using_api_key:
            # For google.generativeai, we can directly pass the text
            revised_cot = self._safe_generate_content(prompt, config=self.cot_generation_config)
        else:
            # For vertexai.generative_models, we need to create a Content object
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
        try:
            current_cot = self.generate_initial_cot(query, choices=choices)
            for i in range(self.num_iterations):
                logging.info(f"[Eval] RAT iteration {i+1} starting.")
                retrieval_query = self.generate_retrieval_query(current_cot, query)
                retrieved_context = self.retrieve_context(retrieval_query)
                current_cot = self.revise_cot(current_cot, retrieved_context, query)
            return current_cot
        except Exception as e:
            # Log the error
            logging.error(f"Error in RAT processing: {e}", exc_info=True)
            
            # Return an error message as the result
            error_message = f"Error processing query with RAT: {str(e)}"
            return error_message