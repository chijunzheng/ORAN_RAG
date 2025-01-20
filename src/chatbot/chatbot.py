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

    def generate_similar_queries(self, original_query: str, num_similar: int = 3) -> List[str]:
        """
        Generates similar queries to the original user query using the generative LLM.
        
        Args:
            original_query (str): The original user query.
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

        prompt_content = Content(
            role="user",
            parts=[
                Part.from_text(prompt_text)
            ]
        )

        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config,
            )
            similar_queries_text = response.text.strip()
            # Extract the queries from the response
            similar_queries = []
            for line in similar_queries_text.split('\n'):
                if line.strip().startswith(tuple(str(i) + '.' for i in range(1, num_similar + 1))):
                    query = line.split('.', 1)[1].strip()
                    if query:
                        similar_queries.append(query)
            logging.debug(f"Generated similar queries: {similar_queries}")
            return similar_queries
        except Exception as e:
            logging.error(f"Error generating similar queries: {e}", exc_info=True)
            # Fallback: Return the original query if generation fails
            return [original_query]

    def generate_prompt_content(self, query: str, chunks: List[Dict], conversation_history: List[Dict]) -> Content:
        """
        Generates the prompt content for the LLM.
        
        Args:
            query (str): User's query.
            chunks (List[Dict]): Retrieved text chunks.
            conversation_history (List[Dict]): Past conversation history.
        
        Returns:
            Content: The prompt content object.
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
                Begin with a brief introduction summarizing the entire answer. Do not use filler words such as "This response ...", "This document...", "This section...". etc.
                Use high-level headings (##) and subheadings (###) to organize content.
                Present information in bullet points or numbered lists to show the hierarchical structure.
                **Use the document names and page numbers from the context for references, formatted as:** *(Reference: [Document Name], page [Page Number(s)])*.
                **Under each high-level heading(##) and its subheadings(###), **DO NOT** include references right after bullet points( - or *) and numbered lists.**
                **Only include one combined reference at the end of the major heading(##) on separate, indented lines with a smaller font.**
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
            parts=[
                Part.from_text(prompt_text)
            ]
        )
        return user_prompt_content

    def generate_response(self, prompt_content: Content) -> str:
        """
        Generates a response from the generative model.
        
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
        Processes the user query along with conversation history and returns the chatbot response.
        
        Args:
            user_query (str): The user's query.
            conversation_history (List[Dict]): List of previous conversation turns.
        
        Returns:
            str: The chatbot's response.
        """
        try:
            # 1. Generate similar queries
            similar_queries = self.generate_similar_queries(user_query, num_similar=3)
            logging.info(f"Generated similar queries: {similar_queries}")

            # 2. Include original query + similar queries for vector search
            all_queries = [user_query] + similar_queries
            logging.info(f"Performing vector search for the original query and {len(similar_queries)} similar queries.")
            
            # 3. Perform vector search for each similar query
            all_retrieved_chunks = []
            for idx, query in enumerate(all_queries):
                retrieved_chunks = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=query,
                    num_neighbors=30  # 30 chunks per similar query
                )
                logging.info(f"Retrieved {len(retrieved_chunks)} chunks for similar query {idx + 1}: '{query}'")
                all_retrieved_chunks.extend(retrieved_chunks)

            logging.info(f"Total retrieved chunks from similar queries: {len(all_retrieved_chunks)}")

            if not all_retrieved_chunks:
                logging.warning("No chunks retrieved for the similar queries.")
                return "I'm sorry, I couldn't find relevant information to answer your question."

            # 3. Rerank the retrieved chunks
            reranked_chunks = self.reranker.rerank(
                query=user_query,
                records=all_retrieved_chunks
            )
            logging.info(f"Reranked to {len(reranked_chunks)} top chunks.")

            if not reranked_chunks:
                logging.warning("Reranking returned no results.")
                return "I'm sorry, I couldn't find relevant information after reranking."

            # 4. Generate prompt with reranked chunks and conversation history
            prompt_content = self.generate_prompt_content(
                query=user_query,
                chunks=reranked_chunks,
                conversation_history=conversation_history
            )

            # 5. Generate response from the model
            assistant_response = self.generate_response(prompt_content)

            return assistant_response.strip() if assistant_response else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logging.error(f"Error in get_response: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."