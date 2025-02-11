# src/chatbot/rat_processor.py

import logging
from vertexai.generative_models import Content, GenerationConfig
from typing import List
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part
)

# We assume that the vector_searcher and generative model (llm) follow similar interfaces as before.

class RATProcessor:
    """
    Implements Retrieval Augmented Thoughts (RAT) for ORAN document chunks.
    
    Given a task query, it performs an iterative chain-of-thought (CoT) revision process:
      1. Generates an initial chain-of-thought.
      2. For a fixed number of iterations (default 3):
           a. Generates a retrieval query based on the current CoT and the original query.
           b. Retrieves relevant ORAN document chunks using the vector searcher.
           c. Revises the chain-of-thought using the retrieved context.
      3. Finally, uses the final revised CoT to generate the final answer.
    
    This is applied only for ORAN queries (e.g. when vendor is an ORAN version such as "v06.00").
    """
    def __init__(
        self,
        vector_searcher,
        llm,
        generation_config: GenerationConfig,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        num_iterations: int = 3
        ):
        self.vector_searcher = vector_searcher
        self.llm = llm  # generative model (e.g., gemini-1.5-flash-002)
        self.generation_config = generation_config
        self.num_iterations = num_iterations
        self.index_endpoint_display_name = index_endpoint_display_name
        self.deployed_index_id = deployed_index_id

    def generate_initial_cot(self, query: str) -> str:
        """
        Generates an initial chain-of-thought (CoT) for the query.
        """
        prompt = f"Let's think step-by-step to answer the following question: {query}"
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.generation_config)
        initial_cot = response.text.strip() if response and response.text.strip() else ""
        logging.info(f"Initial CoT: {initial_cot[:200]}...")
        return initial_cot

    def generate_retrieval_query(self, current_cot: str, query: str) -> str:
        """
        Uses the current chain-of-thought and the original query to generate a concise retrieval query.
        """
        prompt = f"""Based on the current chain-of-thought:
"{current_cot}"
and the original query:
"{query}"
Generate a concise retrieval query that will help ground the answer using ORAN document sources.
Output only the retrieval query as plain text."""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.generation_config)
        retrieval_query = response.text.strip() if response and response.text.strip() else ""
        logging.info(f"Retrieval query: {retrieval_query}")
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
        # For each retrieved chunk, we take the "content"
        context_text = "\n".join(chunk.get("content", "") for chunk in retrieved)
        logging.info(f"Retrieved context (first 200 chars): {context_text[:200]}...")
        return context_text

    def revise_cot(self, current_cot: str, retrieved_context: str, query: str) -> str:
        """
        Revises the current chain-of-thought by instructing the LLM to incorporate the retrieved context.
        """
        prompt = f"""The following is the current chain-of-thought (CoT):
        "{current_cot}"

        The following retrieved context is available:
        "{retrieved_context}"

        Given the original query:
        "{query}"
        Revise the chain-of-thought to incorporate relevant information from the retrieved context and correct any errors.
        Output only the revised chain-of-thought as plain text."""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.generation_config)
        revised_cot = response.text.strip() if response and response.text.strip() else current_cot
        logging.info(f"Revised CoT: {revised_cot[:200]}...")
        return revised_cot

    def generate_final_answer(self, final_cot: str, query: str) -> str:
        """
        Uses the final revised chain-of-thought to produce a final answer.
        """
        prompt = f"""
        <purpose>
            You are an expert in O-RAN systems. Always start by focusing on the refined chain-of-thought provided below to maintain context and alignment in your reasoning. Then use any relevant background knowledge along with the query to generate a precise and well-structured answer.
        </purpose>

        <core-concept>
        {final_cot}
        </core-concept>

        <instructions>
            <instruction>Use the refined chain-of-thought above to produce a concise, accurate answer.</instruction>
            <instruction>Structure your answer using high-level headings (##) and subheadings (###), and use bullet points or numbered lists where appropriate.</instruction>
            <instruction>Cover all relevant aspects in a clear, step-by-step manner.</instruction>
            <instruction>Follow the specified answer format and reference rules: do not include references inline; instead, gather them at the end of each major section in a single block formatted with HTML `<small>` tags.</instruction>
            <instruction>Keep the tone professional and informative, suitable for engineers new to O-RAN systems.</instruction>
        </instructions>

        <question>
        {query}
        </question>

        <answer>

        </answer>
        """
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.generation_config)
        final_answer = response.text.strip() if response and response.text.strip() else final_cot
        logging.info(f"Final Answer: {final_answer[:200]}...")
        return final_answer

    def process_query(self, query: str) -> str:
        """
        Performs the iterative retrieval augmented thought process:
          1. Generates an initial CoT.
          2. Iteratively refines the CoT using retrieved context for num_iterations.
          3. Generates a final answer from the final revised CoT.
        """
        current_cot = self.generate_initial_cot(query)
        for i in range(self.num_iterations):
            logging.info(f"RAT iteration {i+1} starting.")
            retrieval_query = self.generate_retrieval_query(current_cot, query)
            retrieved_context = self.retrieve_context(retrieval_query)
            current_cot = self.revise_cot(current_cot, retrieved_context, query)
        final_answer = self.generate_final_answer(current_cot, query)
        return final_answer