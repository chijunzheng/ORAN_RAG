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
        Initializes the Chatbot with specific configuration parameters.
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

    # ---------------------------------------------------------
    # 1) STEP-BACK METHOD: Identify the "Core ORAN Concept" 
    # ---------------------------------------------------------
    def generate_core_concept(self, user_query: str, conversation_history: List[Dict]) -> str:
        """
        Calls the LLM to generate the core ORAN concept behind a query.
        Minimal step-back approach: we feed the user_query (and optionally part of the conversation).
        """
        # You can optionally incorporate the last N conversation turns if you want
        # context from the conversation. For brevity, we’ll just do the user query.
        truncated_history = conversation_history[-3:]  # Last 3 turns, for example
        conversation_text = ""
        for turn in truncated_history:
            conversation_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        # Create a simple prompt to instruct the model to identify the “core concept”
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
                generation_config=self.generation_config,
            )
            core_concept = response.text.strip()
            # Basic sanitization
            return core_concept if core_concept else "O-RAN Architecture (General)"
        except Exception as e:
            logging.error(f"Error generating core concept: {e}", exc_info=True)
            # Return a fallback concept if generation fails
            return "O-RAN Architecture (General)"

    # ---------------------------------------------------------
    # 2) PROMPT BUILDING 
    # ---------------------------------------------------------
    def generate_prompt_content(self, query: str, concept: str, chunks: List[Dict], conversation_history: List[Dict]) -> Content:
        """
        Generates the prompt content for the LLM using the core concept 
        from the step-back prompting stage.
        """
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        history_text += f"User: {query}\n"

        context = "\n\n".join([
            f"Chunk {i+1} ({chunk['document_name']}, Page {chunk['page_number']}):\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])

        # Note the insertion of the "Core Concept" in the prompt
        prompt_text = f"""
        <purpose>
            You are an expert in O-RAN systems. Always start by focusing on the "core concept" below 
            to keep the reasoning aligned. Then use conversation history and context 
            (plus pre-trained knowledge if necessary) to provide an accurate answer.
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
                <markdown-guideline>For references, use HTML `<small>` tags to reduce font size and indent them using spaces.</markdown-guideline>
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

    # ---------------------------------------------------------
    # 3) FINAL RESPONSE
    # ---------------------------------------------------------
    def generate_response(self, prompt_content: Content) -> str:
        """
        Generates a response from the generative model.
        """
        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config,
            )
            assistant_response = response.text.strip()

            # Remove any wrapping code blocks from the response
            if assistant_response.startswith("```") and assistant_response.endswith("```"):
                lines = assistant_response.split("\n")
                if len(lines) >= 3:
                    # Remove the first and last lines (```)
                    assistant_response = "\n".join(lines[1:-1])
            
            logging.debug("Generated assistant response.")
            return assistant_response
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request."

    def get_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        """
        Processes the user query along with conversation history and returns the chatbot response
        with Step-Back Prompting integrated.
        """
        try:
            # --------------------------------------------------------
            # (A) STEP-BACK: Identify the core concept
            # --------------------------------------------------------
            core_concept = self.generate_core_concept(user_query, conversation_history)
            logging.info(f"Step-Back concept extracted: {core_concept}")

            # --------------------------------------------------------
            # (B) Retrieve relevant chunks using vector search
            # --------------------------------------------------------
            retrieved_chunks = self.vector_searcher.vector_search(
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
                query_text=user_query,
                num_neighbors=self.num_neighbors
            )
            logging.info(f"Retrieved {len(retrieved_chunks)} chunks for the query.")

            if not retrieved_chunks:
                logging.warning("No chunks retrieved for the query.")
                return "I'm sorry, I couldn't find relevant information to answer your question."

            # --------------------------------------------------------
            # (C) Rerank the retrieved chunks
            #     (Optional) If you want to rerank with the concept, 
            #     pass `core_concept` instead of `user_query` below.
            # --------------------------------------------------------
            reranked_chunks = self.reranker.rerank(
                query=core_concept,
                records=retrieved_chunks
            )
            logging.info(f"Reranked to {len(reranked_chunks)} top chunks.")

            if not reranked_chunks:
                logging.warning("Reranking returned no results.")
                return "I'm sorry, I couldn't find relevant information after reranking."

            # --------------------------------------------------------
            # (D) Generate final prompt with concept + reranked chunks
            # --------------------------------------------------------
            prompt_content = self.generate_prompt_content(
                query=user_query,
                concept=core_concept,
                chunks=reranked_chunks,
                conversation_history=conversation_history
            )

            # --------------------------------------------------------
            # (E) Generate the final LLM response
            # --------------------------------------------------------
            assistant_response = self.generate_response(prompt_content)

            return assistant_response.strip() if assistant_response else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logging.error(f"Error in get_response: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."