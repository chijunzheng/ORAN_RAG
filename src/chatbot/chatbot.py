# src/chatbot/chatbot.py

import uuid
import logging
from typing import List, Dict
from google.cloud import firestore, aiplatform
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part,
    Tool
)
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.preview import rag
from src.vector_search.searcher import VectorSearcher
import vertexai


class Chatbot:
    def __init__(
        self,
        config: dict,
        vector_searcher: VectorSearcher,  
        credentials,
    ):
        """
        Initializes the Chatbot with specific configuration parameters.

        Args:
            config (dict): Configuration dictionary containing all necessary parameters.
            vector_searcher (VectorSearcher): Instance of VectorSearcher for performing searches.
            credentials: Google Cloud credentials.
        """
        try:
            self.project_id = config.get('gcp', {}).get('project_id')
            self.location = config.get('gcp', {}).get('location')
            self.bucket_name = config.get('gcp', {}).get('bucket_name')
            self.embeddings_path = config.get('gcp', {}).get('embeddings_path')
            self.bucket_uri = config.get('gcp', {}).get('bucket_uri')
            self.index_display_name = config.get('vector_search', {}).get('index_display_name')
            self.endpoint_display_name = config.get('vector_search', {}).get('endpoint_display_name')
            self.deployed_index_id = config.get('vector_search', {}).get('deployed_index_id')
            self.num_neighbors = config.get('vector_search', {}).get('num_neighbors', 10)
            self.generation_temperature = config.get('generation', {}).get('temperature', 0.7)
            self.generation_top_p = config.get('generation', {}).get('top_p', 0.9)
            self.generation_max_output_tokens = config.get('generation', {}).get('max_output_tokens', 2000)
            self.generative_model_name = config.get('generation', {}).get('generative_model_name', "gemini-1.5-flash-002")
            self.rag_corpus_resource = config.get('vector_search', {}).get('rag_corpus_resource')
            self.vector_searcher = vector_searcher
            self.credentials = credentials

            # Initialize AI Platform and Vertex AI
            vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            aiplatform.init(project=self.project_id, location=self.location, credentials=credentials)
            logging.info(f"Initialized Chatbot with project_id='{self.project_id}', location='{self.location}', bucket_uri='{self.bucket_uri}'")

            # Initialize Firestore Client
            self.db = firestore.Client(project=self.project_id, credentials=credentials)
            logging.info("Initialized Firestore client.")

            # Initialize Vertex AI Generative Model
            self.generative_model = GenerativeModel(self.generative_model_name)
            logging.info(f"Initialized GenerativeModel '{self.generative_model_name}'.")

            # Setup Generation Configuration
            self.generation_config = GenerationConfig(
                temperature=self.generation_temperature,
                top_p=self.generation_top_p,
                max_output_tokens=self.generation_max_output_tokens,
            )

            # Initialize RAG API with Retrieval and Reranking
            self.initialize_rag_model()

        except KeyError as ke:
            logging.error(f"Missing configuration key: {ke}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to initialize Chatbot: {e}", exc_info=True)
            raise

    def initialize_rag_model(self):
        """
        Initializes the RAG Generative Model with Retrieval capabilities.
        """
        try:
            # Define RAG Retrieval Configuration
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=self.num_neighbors,
                
            )

            # Define RAG Resources
            rag_retrieval_resources = [
                rag.RagResource(
                    rag_corpus=self.rag_corpus_resource
                )
            ]

            # Initialize RAG Retrieval
            rag_retrieval = rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=rag_retrieval_resources,
                    rag_retrieval_config=rag_retrieval_config
                )
            )

            # Initialize RAG Retrieval Tool
            rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag_retrieval
            )

            # Initialize the RAG Generative Model with the Retrieval Tool and Generative Model
            self.rag_model = GenerativeModel(
                model_name=self.generative_model_name,  # Generative model from config
                tools=[rag_retrieval_tool],
                generation_config=self.generation_config
            )

            logging.info("Initialized RAG GenerativeModel with Retrieval and Generative Model for generation.")

        except Exception as e:
            logging.error(f"Failed to initialize RAG model: {e}", exc_info=True)
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

        sorted_chunks = sorted(chunks, key=lambda x: x['distance'])
        top_chunks = sorted_chunks[:15]
        context = "\n\n".join([
            f"Chunk {i+1} ({chunk['document_name']}, Page {chunk['page_number']}):\n{chunk['content']}"
            for i, chunk in enumerate(top_chunks)
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
            response = self.rag_model.generate_content(
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
                retrieved_chunks = self.vector_searcher.vector_search(
                    query_text=query_text,
                    num_neighbors=self.num_neighbors,
                )
                logging.info(f"Retrieved {len(retrieved_chunks)} chunks for the query.")
            except Exception as e:
                logging.error(f"Vector search failed: {e}", exc_info=True)
                print("Chatbot: I'm sorry, I couldn't retrieve the necessary information to answer your query.\n")
                continue

            try:
                prompt_content = self.generate_prompt_content(query_text, retrieved_chunks, conversation_history)
                assistant_response = self.generate_response(prompt_content)
            except Exception as e:
                logging.error(f"Failed to generate assistant response: {e}", exc_info=True)
                print("Chatbot: I'm sorry, I encountered an error while generating a response.\n")
                continue

            print(f"Chatbot: {assistant_response}\n")

            conversation_history.append({
                'user': query_text,
                'assistant': assistant_response
            })

            try:
                self.save_conversation(session_id, conversation_history)
            except Exception as e:
                logging.error(f"Failed to save conversation history: {e}", exc_info=True)
                print("Chatbot: Warning - Failed to save conversation history.\n")
                continue