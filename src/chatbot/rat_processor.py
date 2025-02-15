# src/chatbot/rat_processor.py

import logging
from vertexai.generative_models import Content, GenerationConfig
from typing import List, Dict
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
        num_iterations: int = 2
        ):
        self.vector_searcher = vector_searcher
        self.llm = llm  # generative model (e.g., gemini-1.5-flash-002)
        self.generation_config = generation_config
        self.num_iterations = num_iterations
        self.index_endpoint_display_name = index_endpoint_display_name
        self.deployed_index_id = deployed_index_id

        self.cot_generation_config = GenerationConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=8192,
        )    

    def generate_initial_cot(self, query: str) -> str:
        """
        Generates an initial chain-of-thought (CoT) for the query.
        Instruct the LLM to provide a detailed, step-by-step reasoning outline—even if some steps may be uncertain—
        so that later iterations can refine and correct the reasoning.
        """
        prompt = f"""
            You are an expert in complex problem solving and iterative reasoning. For the following question,
            generate a detailed initial chain-of-thought that outlines your step-by-step reasoning.
            Do not worry if some steps are uncertain or incomplete; provide a clear draft of your thought process.

            Question: {query}

            Chain-of-Thought:"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.cot_generation_config)
        initial_cot = response.text.strip() if response and response.text.strip() else ""
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
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.cot_generation_config)
        retrieval_query = response.text.strip() if response and response.text.strip() else ""
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
        # For each retrieved chunk, we take the "content"
        context_text = "\n".join(chunk.get("content", "") for chunk in retrieved)
        logging.info(f"Retrieved context (first 1000 chars): {context_text[1000]}...")
        return context_text

    def revise_cot(self, current_cot: str, retrieved_context: str, query: str) -> str:
        """
        Revises the current chain-of-thought by incorporating the retrieved context.
        Instruct the LLM to re-read the current chain-of-thought along with the retrieved factual context and produce
        an updated chain-of-thought that corrects errors and adds missing details.
        """
        prompt = f"""
            You are an expert in iterative reasoning refinement. Review your current chain-of-thought and the context
            retrieved from grounded sources. Based on the retrieved context, thoroughly revise and improve upon your current chain-of-thought, correct any
            mistakes including false acronyms, false information, false premises, etc., and improve clarity. Provide the updated chain-of-thought as a plain text output with no additional commentary.

            <answer-format>
                Reference Rules:
                - Do NOT place references after individual bullets or sentences.
                - Do NOT place references inline within paragraphs.                    
                - Instead, gather all the references at the end of the relevant heading (## or ###) in one combined block.
                - When listing references, include the actual document name and page number(s) as provided in the retrieved context.
                - Format references in smaller font, using HTML `<small>` tags and indentation. For example:
                        <small>
                            &nbsp;&nbsp;*(Reference: [Document Name], page [Page Number(s)]; [Another Document], page [Page Number(s)])*
                        </small>
                - Only include references from the retrieved context
            </answer-format>

            Current Chain-of-Thought:
            "{current_cot}"

            Retrieved Context:
            "{retrieved_context}"

            Original Question:
            "{query}"

            Revised Chain-of-Thought:"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.cot_generation_config)
        revised_cot = response.text.strip() if response and response.text.strip() else current_cot
        logging.info(f"Revised CoT (first 1000 chars): {revised_cot[:1000]}...")
        return revised_cot

    def generate_final_answer(self, final_cot: str, query: str, conversation_history: List[Dict]) -> str:
        """
        Uses the final revised chain-of-thought to produce a final answer.
        """
        # Build conversation history string from the last 5 turns.
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"User: {turn.get('user')}\nAssistant: {turn.get('assistant')}\n"
        history_text += f"User: {query}\n"

        prompt = f"""
        <purpose>
            You are an expert in O-RAN systems. Always start by focusing on the refined chain-of-thought provided below to maintain context and alignment in your reasoning. Then use any relevant background knowledge along with the query to generate a precise and well-structured answer.
        </purpose>

        <refined chain-of-thought>
        {final_cot}
        </refined chain-of-thought>

        <instructions>
            <instruction>Use the refined chain-of-thought along with the original query and the recent conversation history to produce a concise and precise final answer.</instruction>
            <instruction>Structure your answer with high-level headings (##) and subheadings (###). Use bullet points or numbered lists where appropriate.</instruction>
            <instruction>Do NOT include reference citations immediately after individual bullet points or inline within paragraphs.</instruction>
            <instruction>Instead, at the end of each section, compile all references in a consolidated block formatted in smaller font using HTML <small> tags. For example:
                <small>
                    &nbsp;&nbsp;*(Reference: [Document Name], page [Page Number(s)]; [Another Document], page [Page Number(s)])*
                </small>
            <instruction>Keep the tone professional and informative, suitable for engineers new to O-RAN systems.</instruction>
        </instructions>

        <sections>
            <answer-format>
                Structure your answer with high-level headings (##) and subheadings (###). Present your response using bullet points or numbered lists.
                **Reference Rules**:
                - Do NOT include any references immediately after bullet points or sentences.
                - Instead, compile all relevant references into one consolidated block at the end of each section.
                - Format the consolidated reference block using HTML <small> tags. 
            </answer-format>
            <markdown-guidelines>
                <markdown-guideline>Use `##` for main sections and `###` for subheadings.</markdown-guideline>
                <markdown-guideline>Use bullet points for lists and maintain consistent indentation.</markdown-guideline>
            </markdown-guidelines>
            <important-notes>
                <important-note>Focus on delivering a complete answer that fully addresses the query.</important-note>
                <important-note>Be logical and concise, while providing detailed information.</important-note>
                <important-note>Ensure the explanation is presented step-by-step, covering all relevant stages.</important-note>
            </important-notes>
            <audience>
                Engineers new to O-RAN.
            </audience>
            <tone>
                Professional and informative.
            </tone>
        </sections>

        <conversation-history>
        {history_text}
        </conversation-history>

        <question>
        {query}
        </question>

        <answer>

        </answer>
        """
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.cot_generation_config)
        final_answer = response.text.strip() if response and response.text.strip() else final_cot
        logging.info(f"Final Answer: {final_answer[:1000]}...")
        return final_answer

    def process_query(self, query: str, conversation_history: List[Dict]) -> str:
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
        #final_answer = self.generate_final_answer(current_cot, query, conversation_history)
        return current_cot