# src/chatbot/chatbot.py

import uuid
import logging
from typing import List, Dict
from google.cloud import firestore, aiplatform
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part
)
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.evaluation.evaluator import Evaluator
from src.vector_search.reranker import Reranker

class Chatbot:
    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: str,
        embeddings_path: str,
        bucket_uri: str,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        generation_temperature: float,
        generation_top_p: float,
        generation_max_output_tokens: int,
        vector_searcher,
        credentials,
        num_neighbors: int,
        reranker: Reranker
    ):
        """
        Multi-query Chatbot that also integrates Step-Back Prompting to identify a core ORAN concept
        before retrieving and reranking.
        """
        try:
            self.project_id = project_id
            self.location = location
            self.bucket_name = bucket_name
            self.embeddings_path = embeddings_path
            self.bucket_uri = bucket_uri
            self.index_endpoint_display_name = index_endpoint_display_name
            self.deployed_index_id = deployed_index_id
            self.vector_searcher = vector_searcher
            self.num_neighbors = num_neighbors
            self.reranker = reranker

            # Initialize AI Platform
            aiplatform.init(
                project=self.project_id, 
                location=self.location, 
                credentials=credentials
            )
            logging.info(f"Initialized Chatbot with project_id='{self.project_id}', location='{self.location}', bucket_uri='{self.bucket_uri}'")

            # Initialize Firestore Client
            self.db = firestore.Client(project=self.project_id, credentials=credentials)
            logging.info("Initialized Firestore client.")

            # Initialize Generative Model
            self.generative_model = GenerativeModel("gemini-1.5-flash-002")
            logging.info("Initialized GenerativeModel 'gemini-1.5-flash-002'.")

            # Setup Generation Configuration
            self.generation_config = GenerationConfig(
                temperature=generation_temperature,
                top_p=generation_top_p,
                max_output_tokens=generation_max_output_tokens,
            )

        except KeyError as ke:
            logging.error(f"Missing configuration key: {ke}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to initialize Chatbot: {e}", exc_info=True)
            raise

    def generate_session_id(self) -> str:
        """Generates a unique session ID."""
        return str(uuid.uuid4())

    def save_conversation(self, session_id: str, conversation_history: List[Dict]):
        """Saves the conversation history to Firestore."""
        try:
            self.db.collection('conversations').document(session_id).set({
                'history': conversation_history
            })
            logging.debug(f"Saved conversation history for session_id='{session_id}'.")
        except Exception as e:
            logging.error(f"Failed to save conversation history: {e}", exc_info=True)
            raise

    def load_conversation(self, session_id: str) -> List[Dict]:
        """Loads the conversation history from Firestore."""
        try:
            doc = self.db.collection('conversations').document(session_id).get()
            if doc.exists:
                history = doc.to_dict().get('history', [])
                logging.debug(f"Loaded conversation history for session_id='{session_id}'.")
                return history
            else:
                logging.debug(f"No existing conversation history for session_id='{session_id}'.")
                return []
        except Exception as e:
            logging.error(f"Failed to load conversation history: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------
    # A) STEP-BACK: Identify a "Core ORAN Concept" from the user query
    # -------------------------------------------------------------------
    def generate_core_concept(self, user_query: str, conversation_history: List[Dict]) -> str:
        """
        Calls the LLM to generate the core ORAN concept behind a query (Step-Back).
        """
        truncated_history = conversation_history[-3:]  # Last 3 turns
        conversation_text = ""
        for turn in truncated_history:
            conversation_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        prompt = f"""
        You are an O-RAN expert. Analyze the user's query below 
        and identify the single core concept needed to answer it.

        Conversation (recent):
        {conversation_text}

        User Query: {user_query}

        Instructions:
        1. Provide a concise concept or principle behind this query.
        2. Do not provide any further explanation—only briefly describe the concept.

        Concept:
        """
        prompt_content = Content(
            role="user",
            parts=[Part.from_text(prompt)]
        )

        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config
            )
            core_concept = response.text.strip()
            if not core_concept:
                core_concept = "O-RAN Architecture (General)"
            return core_concept
        except Exception as e:
            logging.error(f"Error generating core concept: {e}", exc_info=True)
            return "O-RAN Architecture (General)"

    # -------------------------------------------------------------------
    # B) MULTI-QUERY GENERATION (Anchored to core concept)
    # -------------------------------------------------------------------
    def generate_similar_queries(self, original_query: str, core_concept: str, num_similar: int = 3) -> List[str]:
        """
        Generates additional queries that revolve around the identified core concept, 
        exploring the user's query from different angles, but anchored to the concept.
        """
        prompt_text = f"""
        You are an O-RAN expert. The user has asked: "{original_query}"
        The core concept identified is: "{core_concept}"

        Generate {num_similar} unique and diverse queries that explore or elaborate on this core concept,
        while remaining relevant to the user's original question. Each query should:
          - Reflect or incorporate the concept "{core_concept}"
          - Provide a different perspective or subtopic
        Ensure they are distinct yet still aligned with the user's intent.

        Similar Queries:
        1.
        2.
        3.
        """

        user_prompt = Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )

        try:
            response = self.generative_model.generate_content(
                user_prompt,
                generation_config=self.generation_config,
            )
            raw_text = response.text.strip()

            lines = raw_text.split("\n")
            similar_queries = []
            # A quick parse looking for lines like '1. ...', '2. ...', etc.
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith(("1.", "2.", "3.", "4.")):
                    # after the number, the rest is the query
                    query_part = line_stripped.split(".", 1)[-1].strip()
                    if query_part:
                        similar_queries.append(query_part)
            # Fallback if we didn't parse anything
            if not similar_queries:
                similar_queries = [original_query]
            return similar_queries
        except Exception as e:
            logging.error(f"Error generating similar queries: {e}", exc_info=True)
            # Return just the original if generation fails
            return [original_query]

    # -------------------------------------------------------------------
    # C) PROMPT BUILDER
    # -------------------------------------------------------------------
    def generate_prompt_content(self, query: str, concept: str, chunks: List[Dict], conversation_history: List[Dict]) -> Content:
        """
        Build the final user prompt, injecting the 'core concept' from Step-Back 
        and the relevant chunks from multi-query retrieval.
        """
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        history_text += f"User: {query}\n"

        context = "\n\n".join([
            f"Chunk {i+1} ({chunk['document_name']}, Page {chunk['page_number']}):\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])

        prompt_text = f"""
        <purpose>
            You are an expert in O-RAN systems. Always start by focusing on the "core concept" below 
            to keep your reasoning aligned. Then use conversation history, context, 
            and pre-trained knowledge (if necessary) to provide an accurate, thorough answer.
        </purpose>

        <core-concept>
        {concept}
        </core-concept>

        <instructions>
            <instruction>Use the concept and the context to form a concise, thorough response.</instruction>
            <instruction>Cover all relevant aspects in a clear, step-by-step manner.</instruction>
            <instruction>Follow the specified answer format, headings, and style guides.</instruction>
            <instruction>Keep the tone professional and informative, suitable for engineers new to O-RAN systems.</instruction>
        </instructions>

        <context>
        {context}
        </context>

        <conversation-history>
        {history_text}
        </conversation-history>

        <question>
        {query}
        </question>

        <sections>
            <answer-format>
                Begin with a brief introduction summarizing the entire answer.
                Use high-level headings (##) and subheadings (###) to organize content.
                Present information in bullet points or numbered lists to illustrate hierarchy.

                **References Rule**:
                - **Do NOT** place references after individual bullets or sentences.
                - **Do NOT** place references inline within paragraphs.
                - Instead, gather all the references at the **end of the relevant heading** (## or ###) in **one combined block**.
                - Format references in smaller font, using HTML `<small>` tags and indentation.
            </answer-format>
            <markdown-guidelines>
                <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                <markdown-guideline>Use bullet points ( - or * ) for sub-steps and indent consistently.</markdown-guideline>
                <markdown-guideline>Use **bold** for emphasis and *italics* for subtle highlights.</markdown-guideline>
            </markdown-guidelines>
            <important-notes>
                <important-note>Focus on delivering a complete answer that fully addresses the query.</important-note>
                <important-note>Be logical and concise, while providing as much detail as possible.</important-note>
                <important-note>Ensure the explanation is presented step-by-step, covering relevant stages.</important-note>
            </important-notes>
            <audience>
                Engineers new to O-RAN systems.
            </audience>
            <tone>
                Professional and informative.
            </tone>
        </sections>

        <answer>

        </answer>
        """

        user_prompt_content = Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )
        return user_prompt_content

    # -------------------------------------------------------------------
    # D) GENERATE RESPONSE
    # -------------------------------------------------------------------
    def generate_response(self, prompt_content: Content) -> str:
        """
        Calls the generative model to produce a response from the constructed prompt.
        """
        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config
            )
            assistant_response = response.text.strip()

            # Remove wrapping code blocks if present
            if assistant_response.startswith("```") and assistant_response.endswith("```"):
                lines = assistant_response.split("\n")
                if len(lines) >= 3:
                    assistant_response = "\n".join(lines[1:-1])

            logging.debug("Generated assistant response.")
            return assistant_response
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request."

    # -------------------------------------------------------------------
    # E) get_response: Combine Step-Back + Multi-Query
    # -------------------------------------------------------------------
    def get_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        """
        Orchestrates the entire pipeline:
          1) Step-Back to identify a core concept
          2) Multi-query generation (optionally anchored to concept)
          3) Retrieval for each query
          4) Combine & Rerank chunks
          5) Build final prompt referencing the concept
          6) Generate the final LLM response
        """
        try:
            # (1) Step-Back: Extract concept
            core_concept = self.generate_core_concept(user_query, conversation_history)
            logging.info(f"[Step-Back] Core concept: {core_concept}")

            # (2) Multi-Query Generation
            # Option 1: Generate them from the user query and core concept:
            similar_queries = self.generate_similar_queries(user_query,
                                                            core_concept=core_concept,
                                                              num_similar=3)
 

            logging.info(f"Generated {len(similar_queries)} similar queries for multi-query retrieval.")
            
            # Combine original + similar
            all_queries = [user_query] + similar_queries
            
            # (3) Retrieve chunks for each query
            all_chunks = []
            # Adjust as needed or read from config
            neighbors_per_query = 30
            for idx, query_variant in enumerate(all_queries):
                partial_chunks = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=query_variant,
                    num_neighbors=neighbors_per_query
                )
                logging.debug(f"Retrieved {len(partial_chunks)} chunks for query variant [{idx+1}]: {query_variant}")
                all_chunks.extend(partial_chunks)

            if not all_chunks:
                logging.warning("No chunks retrieved for any query variant.")
                return "I'm sorry, I couldn't find relevant information to answer your question."

            # (4) Rerank the combined set of chunks
            # We can choose to re-rank by 'core_concept' or the original user query
            # or even combine them. E.g. concept + user_query
            # For simplicity, let's do re-rank by 'core_concept'
            rerank_method = user_query + " " + core_concept
            reranked_chunks = self.reranker.rerank(query=rerank_method, records=all_chunks)

            if not reranked_chunks:
                logging.warning("Reranking returned no results.")
                return "I'm sorry, I couldn't find relevant information after reranking."

            logging.debug(f"Reranked to {len(reranked_chunks)} top chunks (global).")

            # (E) Build the final prompt with concept + top reranked chunks
            # Typically you'd keep top X after reranking (like 20) to reduce prompt size
            top_k = 20
            top_chunks = reranked_chunks[:top_k]
            prompt_content = self.generate_prompt_content(
                query=user_query,
                concept=core_concept,
                chunks=top_chunks,
                conversation_history=conversation_history
            )

            # (F) Generate the final response
            assistant_response = self.generate_response(prompt_content)
            return assistant_response.strip() if assistant_response else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logging.error(f"Error in multi-query step-back pipeline: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."