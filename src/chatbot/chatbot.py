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
        
        Args:
            project_id (str): Google Cloud project ID.
            location (str): Google Cloud location (e.g., 'us-central1').
            bucket_name (str): Name of the GCS bucket.
            embeddings_path (str): Path within the GCS bucket for embeddings.
            bucket_uri (str): URI of the GCS bucket.
            index_endpoint_display_name (str): Name of the index endpoint.
            deployed_index_id (str): ID of the deployed index.
            generation_temperature (float): Temperature parameter for generation.
            generation_top_p (float): Top-p parameter for generation.
            generation_max_output_tokens (int): Maximum tokens for generation.
            vector_searcher (VectorSearcher): Instance of VectorSearcher for performing searches.
            reranker (Reranker): Instance of Reranker for reranking results.
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
            aiplatform.init(project=self.project_id, location=self.location, credentials=credentials)
            logging.info(f"Initialized Chatbot with project_id='{self.project_id}', location='{self.location}', bucket_uri='{self.bucket_uri}'")

            # Initialize Firestore Client
            self.db = firestore.Client(project=self.project_id, credentials=credentials)
            logging.info("Initialized Firestore client.")

            # Initialize Generative Model (for both hypothetical & final answers)
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
        """
        Saves the conversation history to Firestore.
        
        Args:
            session_id (str): Unique session identifier.
            conversation_history (List[Dict]): List of conversation turns.
        """
        try:
            self.db.collection('conversations').document(session_id).set({
                'history': conversation_history
            })
            logging.debug(f"Saved conversation history for session_id='{session_id}'.")
        except Exception as e:
            logging.error(f"Failed to save conversation history: {e}", exc_info=True)
            raise

    def load_conversation(self, session_id: str) -> List[Dict]:
        """
        Loads the conversation history from Firestore.
        
        Args:
            session_id (str): Unique session identifier.
        
        Returns:
            List[Dict]: List of conversation turns.
        """
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

    def generate_prompt_content(self, query: str, chunks: List[Dict], conversation_history: List[Dict]) -> Content:
        """
        Generates the prompt content for the final generative LLM.
        
        Args:
            query (str): User's query.
            chunks (List[Dict]): Retrieved text chunks.
            conversation_history (List[Dict]): Past conversation history.
        
        Returns:
            Content: The prompt content object.
        """
        # Build up to five turns of conversation
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        history_text += f"User: {query}\n"

        # Build context from chunks
        context = "\n\n".join([
            f"Chunk {i+1} ({chunk['document_name']}, Page {chunk['page_number']}):\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])

        prompt_text = f"""
        <purpose>
            You are an expert in O-RAN systems. Utilize the conversation history, context, and if necessary, pre-trained knowledge to provide detailed and accurate answers to the user's queries.
        </purpose>

        <instructions>
            <instruction>Use the context, conversation history, and if necessary, pre-trained knowledge to form a concise, thorough response.</instruction>
            <instruction>Cover all relevant aspects in a clear, step-by-step manner.</instruction>
            <instruction>Follow the specified answer format, headings, and style guides.</instruction>
            <instruction>Keep the tone professional and informative, suitable for engineers new to O-RAN systems.</instruction>
            <instruction>Do not enclose the entire response within code blocks. Only use code blocks for specific code snippets or technical examples if necessary.</instruction>
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
                - Format references in smaller font, using HTML `<small>` tags and indentation. For example:
                <small>
                    &nbsp;&nbsp;*(Reference: [Document Name], page [Page Number(s)])*
                    &nbsp;&nbsp;*(Reference: [Another Document], page [Page Number(s)])*
                </small>
            </answer-format>
            <markdown-guidelines>
                <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                <markdown-guideline>Use bullet points ( - or * ) for sub-steps and indent consistently.</markdown-guideline>
                <markdown-guideline>Use **bold** for emphasis and *italics* for subtle highlights.</markdown-guideline>
                <markdown-guideline>For references, use HTML `<small>` tags to reduce font size and indent them using spaces.</markdown-guideline>
            </markdown-guidelines>
            <important-notes>
                <important-note>Focus on delivering a complete answer that fully addresses the query.</important-note>
                <important-note>Be logical and concise, while providing as much details as possible.</important-note>
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

    def generate_response(self, prompt_content: Content) -> str:
        """
        Generates a response from the generative model (final answer).
        
        Args:
            prompt_content (Content): The prompt content object.
        
        Returns:
            str: The assistant's response.
        """
        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config,
            )
            assistant_response = response.text.strip()

            # Remove any wrapping code blocks if present
            if assistant_response.startswith("```") and assistant_response.endswith("```"):
                lines = assistant_response.split("\n")
                if len(lines) >= 3:
                    assistant_response = "\n".join(lines[1:-1])
            
            logging.debug("Generated assistant response.")
            return assistant_response
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request."

    def generate_hypothetical_answer(self, user_query: str) -> str:
        """
        Generates a hypothetical short answer for the user query using the LLM.
        This hypothetical answer is used to improve retrieval performance (HyDE).
        
        Args:
            user_query (str): The user's query.
        
        Returns:
            str: A hypothetical answer text.
        """
        try:
            # Simple prompt to produce a concise hypothetical answer
            prompt_text = f"Provide a concise hypothetical answer for the query:\n\n{user_query}\n\nAnswer:"
            prompt_content = Content(
                role="user",
                parts=[Part.from_text(prompt_text)]
            )

            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config,
            )
            hypothetical = response.text.strip()
            logging.debug(f"Hypothetical answer generated: {hypothetical}")
            return hypothetical
        except Exception as e:
            logging.error(f"Error generating hypothetical answer: {e}", exc_info=True)
            return ""

    def get_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        """
        Implements the HyDE approach to retrieve top-ranked chunks and generate a final answer:
        
        1) Generate a hypothetical answer from the query using the LLM.
        2) Perform vector search for the original query (top 30).
        3) Perform vector search for the hypothetical answer (top 30).
        4) Combine and rerank the total retrieved chunks to top 20.
        5) Generate a final answer using the retrieved + reranked chunks.
        
        Args:
            user_query (str): The user's query.
            conversation_history (List[Dict]): List of previous conversation turns.
        
        Returns:
            str: The chatbot's final answer.
        """
        try:
            # Step 1: Generate Hypothetical Answer
            hypothetical_answer = self.generate_hypothetical_answer(user_query)

            # Step 2: Retrieve 30 chunks using original query
            retrieved_chunks_query = self.vector_searcher.vector_search(
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
                query_text=user_query,
                num_neighbors=30
            )
            logging.info(f"Retrieved {len(retrieved_chunks_query)} chunks for the original query.")

            # Step 3: Retrieve 30 chunks using hypothetical answer
            retrieved_chunks_hypo = []
            if hypothetical_answer:
                retrieved_chunks_hypo = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=hypothetical_answer,
                    num_neighbors=30
                )
            logging.info(f"Retrieved {len(retrieved_chunks_hypo)} chunks for the hypothetical answer.")

            # Combine both sets of chunks
            combined_chunks = retrieved_chunks_query + retrieved_chunks_hypo
            if not combined_chunks:
                logging.warning("No relevant chunks found (query + hypothetical).")
                return "I'm sorry, I couldn't find relevant information to answer your question."

            # Step 4: Rerank to produce top 20
            reranked_chunks = self.reranker.rerank(
                query=user_query,       # We keep the user query as the main anchor
                records=combined_chunks
            )
            top_20_chunks = reranked_chunks[:20]
            logging.info(f"Reranked to {len(top_20_chunks)} top chunks using HyDE approach.")

            if not top_20_chunks:
                logging.warning("Reranking returned no results.")
                return "I'm sorry, I couldn't find relevant information after reranking."

            # Step 5: Generate final response from the top 20 chunks
            prompt_content = self.generate_prompt_content(
                query=user_query,
                chunks=top_20_chunks,
                conversation_history=conversation_history
            )
            assistant_response = self.generate_response(prompt_content)
            return assistant_response.strip() if assistant_response else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logging.error(f"Error in get_response (HyDE pipeline): {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."