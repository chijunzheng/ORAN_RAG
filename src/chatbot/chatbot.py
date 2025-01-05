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
You are an expert in O-RAN systems. Utilize the conversation history and context to provide detailed and accurate answers to the user's queries.

Instruction:
Using the information provided in the context, please provide a logical and concise answer to the question below. Focus on delivering the best possible answer based on the available data without mentioning any limitations or referencing the context explicitly. Ensure that your answer covers all relevant aspects and includes necessary details.

Context:
{context}

Conversation History:
{history_text}

Question:
{query}

Answer Format:
- Begin with a brief introduction summarizing the procedure.
- Organize your answer using high-level headings if applicable, for better readability. Use numbered lists for main steps and bullet points for sub-steps.
- Use clear and simple language that is easy to understand.
- Include relevant background information and technical details from other documents where applicable.
- Use markdown to make the content appealing and easy to read.
- **Include references only once after each major heading or section. Do not include references after individual sentences, numbered lists, or bullet points.**
  - **Format references as:** *(Reference: [Document Name], page [Page Number(s)])*

**Important Notes:**
- Focus on delivering a complete answer that fully addresses the query.
- Be logical and concise, shortening the answer while retaining key main ideas.
- Ensure that the explanation is presented in a step-by-step manner, covering all stages relevant to the question.

Audience:
Engineers new to O-RAN systems.

Tone:
Professional and informative.

Answer:
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
            logging.debug("Generated assistant response.")
            return assistant_response
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request."

    def chat_loop(self):
        """
        Starts the chatbot interaction loop.
        """
        session_id = self.generate_session_id()
        conversation_history = self.load_conversation(session_id)

        print("Welcome to the O-RAN Chatbot! Type 'exit' to quit.\n")

        while True:
            try:
                query_text = input("User: ")
            except (EOFError, KeyboardInterrupt):
                print("\nChatbot: Goodbye!")
                break

            if query_text.lower() in ['exit', 'quit']:
                print("Chatbot: Goodbye!")
                break

            if not query_text.strip():
                print("Chatbot: Please enter a valid query.\n")
                continue

            try:
                # Step 1: Vector Search
                retrieved_chunks = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=query_text,
                    num_neighbors=self.num_neighbors
                )

                # Step 2: Reranking
                reranked_chunks = self.reranker.rerank(
                    query=query_text,
                    records=retrieved_chunks,
                )

                if not reranked_chunks:
                    logging.warning("Reranking returned no results.")
                    print("Chatbot: I'm sorry, I couldn't retrieve the necessary information to answer your query.\n")
                    continue

                logging.info(f"Reranked to {len(reranked_chunks)} top chunks.")

                # Step 3: Generate Prompt
                prompt_content = self.generate_prompt_content(
                    query=query_text,
                    chunks=reranked_chunks,
                    conversation_history=conversation_history
                )

                # Step 4: Generate Response
                assistant_response = self.generate_response(prompt_content)

                # Step 5: Update Conversation History
                conversation_history.append({"user": query_text, "assistant": assistant_response})
                self.save_conversation(session_id, conversation_history)

                # Step 6: Display Assistant Response
                print(f"Chatbot: {assistant_response}\n")

            except Exception as e:
                logging.error(f"Failed during search or reranking: {e}", exc_info=True)
                print("Chatbot: I'm sorry, I couldn't retrieve the necessary information to answer your query.\n")