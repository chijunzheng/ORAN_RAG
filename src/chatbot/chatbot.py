import re
import uuid
import logging
import tiktoken
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
from src.chatbot.yang_processor import YangProcessor
from src.chatbot.rat_processor import RATProcessor



# ----------------------------------------------------------------------------------
# Chatbot Class
# ----------------------------------------------------------------------------------

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

            self.yang_generation_config = GenerationConfig(
                temperature=0,
                top_p=1,
                max_output_tokens=generation_max_output_tokens,
            )

            # Instantiate the YangProcessor
            self.yang_processor = YangProcessor(vector_searcher=vector_searcher)

        except KeyError as ke:
            logging.error(f"Missing configuration key: {ke}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Failed to initialize Chatbot: {e}", exc_info=True)
            raise

    # ----------------------------------------------------------------------------------
    # Conversation Management
    # ----------------------------------------------------------------------------------
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



    # ----------------------------------------------------------
    # YANG Analysis API (Query Translation, Filtering, Stitching)
    # ----------------------------------------------------------
    def get_yang_analysis(self, query: str, yang_chunks: List[Dict]) -> str:
        """
        Uses YangProcessor to parse the query, filter and stitch YANG chunks,
        then calls the generative model to produce an analysis.
        """
        processor = YangProcessor()
        return processor.get_analysis(query, yang_chunks, self.generative_model, self.yang_generation_config)


    # ----------------------------------------------------------------------------------
    # Vector Search & LLM Flow
    # ----------------------------------------------------------------------------------
    
    def _convert_to_search_format(self, chunk_data: Dict) -> Dict:
        """
        Converts a chunk record to a format suitable for reranking.
        """
        return {
            'id': chunk_data['id'],
            'content': chunk_data.get('content', "No content"),
            'document_name': chunk_data.get('document_name', "Unknown doc"),
            'page_number': chunk_data.get('page_number', "Unknown page"),
            'metadata': chunk_data.get('metadata', {}),
        }
    
    def generate_core_concept(self, user_query: str, conversation_history: List[Dict]) -> str:
        truncated_history = conversation_history[-3:] 
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

        concept_generation_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=128 
        )

        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=concept_generation_config,
            )
            core_concept = response.text.strip()
            return core_concept if core_concept else "O-RAN Architecture (General)"
        except Exception as e:
            logging.error(f"Error generating core concept: {e}", exc_info=True)
            return "O-RAN Architecture (General)"

    def generate_prompt_content(self, query: str, concept: str, chunks: List[Dict], conversation_history: List[Dict]) -> Content:
        """
        Builds the final user prompt. The prompt includes the ORAN context and—only if the query is
        explicitly YANG related—the YANG context block is inserted.
        """
        # (1) Build conversation history text from the last 5 turns.
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        history_text += f"User: {query}\n"

        # (2) Separate the chunks into ORAN and YANG chunks.
        oran_doc_chunks = []
        yang_doc_chunks = []
        for chunk in chunks:
            meta = chunk.get('metadata', {})
            doc_type = meta.get('doc_type', 'unknown')
            if doc_type == 'yang_model':
                yang_doc_chunks.append(chunk)
            else:
                oran_doc_chunks.append(chunk)

        # (3) Build the ORAN context from the ORAN document chunks.
        oran_context = "\n\n".join([
            f"Chunk {i+1} => "
            f"Document Name: {chunk.get('document_name','N/A')}\n"
            f"Version: {chunk.get('metadata', {}).get('version','unknown')}\n"
            f"Workgroup: {chunk.get('metadata', {}).get('workgroup','unknown')}\n"
            f"Subcategory: {chunk.get('metadata', {}).get('subcategory','unknown')}\n"
            f"Page: {chunk.get('page_number','N/A')}\n\n"
            f"{chunk.get('content','No content')}"
            for i, chunk in enumerate(chunks)
        ])
       

        # (4) Build the final prompt text with the reference rules intact.
        prompt_text = f"""
            <purpose>
                You are an expert in O-RAN systems. Always start by focusing on the "core concept" below 
                to keep the reasoning aligned. Then use conversation history and the context 
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
            {oran_context}
            </context>

            <conversation-history>
            {history_text}
            </conversation-history>

            <question>
            {query}
            </question>

            <sections>
                <answer-format>
                    Structure your answer with high-level headings (##) and subheadings (###). Present information in bullet points or numbered lists.
                    **Reference Rules**:
                    - **Do NOT** place references after individual bullets or sentences.
                    - **Do NOT** place references inline within paragraphs.                    
                    - Instead, gather all the references at the **end of the relevant heading** (## or ###) in **one combined block**.
                    - Format references in smaller font, using HTML `<small>` tags and indentation. For example:
                            <small>
                                &nbsp;&nbsp;*(Reference: [Document Name], page [Page Number(s)]; [Another Document], page [Page Number(s)])*
                            </small>
                </answer-format>
                <markdown-guidelines>
                    <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                    <markdown-guideline>Use bullet points for lists and maintain consistent indentation.</markdown-guideline>
                </markdown-guidelines>
                <important-notes>
                    <important-note>Focus on delivering a complete answer that fully addresses the query.</important-note>
                    <important-note>Be logical and concise, while providing as much detail as possible.</important-note>
                    <important-note>Ensure the explanation is presented step-by-step, covering relevant stages.</important-note>
                </important-notes>
                <audience>
                    Engineers new to O-RAN.
                </audience>
                <tone>
                    Professional and informative.
                </tone>
            </sections>

            <answer>

            </answer>
            """
        return Content(
            role="user",
            parts=[Part.from_text(prompt_text)]
        )

    def generate_response(self, prompt_content: Content) -> str:
        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config,
            )
            assistant_response = response.text.strip()
            # Remove code block markers if present.
            if assistant_response.startswith("```") and assistant_response.endswith("```"):
                lines = assistant_response.split("\n")
                if len(lines) >= 3:
                    assistant_response = "\n".join(lines[1:-1])
            logging.debug("Generated assistant response.")
            return assistant_response
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your request."



    # --------------------------------------------------------
    # Main Entry: get_response
    # --------------------------------------------------------
    def get_response(self, 
                     user_query: str, 
                     conversation_history: List[Dict],
                     use_cot: bool = False
                     ) -> str:
        """
        Main entry point. 
        Depending on user_query:
          - If listing YANG inventory, return a summary of modules.
          - If comparing vendor packages, call compare_vendor_packages.
          - If referencing a specific YANG file, reassemble it.
          - Otherwise, fallback to vector search + re-ranking + final generation.
        """
        try:

            # (2) Otherwise, use RATProcessor first to generate a preliminary answer.
            if use_cot:
                logging.info("CoT toggle is ON => using RATProcessor for chain-of-thought approach.")
                rat = RATProcessor(
                    vector_searcher=self.vector_searcher,
                    llm=self.generative_model,
                    generation_config=self.generation_config,
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    num_iterations=3
                )
                revised_cot = rat.process_query(user_query, conversation_history)
                logging.info(f"Preliminary answer (first 200 chars): {revised_cot[:200]}...")

                # (3) Build a final "refine" prompt that references the final CoT, query, and conversation history.
                #     We skip any extra vector search or reranking step here.
                # Gather last 5 conversation turns:
                history_text = ""
                for turn in conversation_history[-5:]:
                    history_text += f"User: {turn.get('user')}\nAssistant: {turn.get('assistant')}\n"
                history_text += f"User: {user_query}\n"

                refine_prompt = f"""
                <purpose>
                You are an advanced O-RAN expert. Refine the following chain-of-thought into a final, well-structured answer.
                Use the conversation history for context and clarity. Do not insert references inline after bullets; place them in a consolidated block at the end of each relevant section.
                </purpose>                
                <instructions>
                    <instruction>Structure the answer with high-level headings (##) and subheadings (###). Present information in bullet points or numbered lists.</instruction>
                    <instruction>Do NOT include reference citations immediately after individual bullet points or inline within paragraphs.</instruction>
                    <instruction>Instead, compile all references at the end of each section in one consolidated block formatted with HTML <small> tags.</instruction>
                    <instruction>Keep the tone professional and informative, suitable for engineers new to O-RAN systems.</instruction>
                </instructions>
                <sections>
                    <answer-format>
                        - Structure the answer with high-level headings (##) and subheadings (###). Present information in bullet points or numbered lists.
                        **Reference Rules**:
                        - **Do NOT** place references after individual bullets or sentences.
                        - **Do NOT** place references inline within paragraphs.                    
                        - Instead, gather all the references at the **end of the relevant heading** (## or ###) in **one combined block**.
                        - When listing references, include the actual document name and page number(s) as provided in the retrieved context.
                        - Format references in smaller font, using HTML `<small>` tags and indentation. For example:
                                <small>
                                    &nbsp;&nbsp;*(Reference: [Document Name], page [Page Number(s)]; [Another Document], page [Page Number(s)])*
                                </small>
                    </answer-format>
                    <markdown-guidelines>
                        <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                        <markdown-guideline>Use bullet points or numbered lists and maintain consistent indentation.</markdown-guideline>
                    </markdown-guidelines>
                    <important-notes>
                        <important-note>Focus on delivering a complete answer that fully addresses the query.</important-note>
                        <important-note>Be logical and concise, while providing as much detail as possible.</important-note>
                        <important-note>Ensure the explanation is presented step-by-step, covering relevant stages.</important-note>
                    </important-notes>
                    <audience>
                        Engineers new to O-RAN.
                    </audience>
                    <tone>
                        Professional and informative.
                    </tone>
                </sections>
                
                <Revised Chain-of-Thought>
                {revised_cot}
                </Revised Chain-of-Thought>

                <conversation-history>
                {conversation_history}
                </conversation-history>

                User Query:
                {user_query}

                <answer>

                </answer>
                """
                content = Content(role="user", parts=[Part.from_text(refine_prompt)])
                final_response = self.generative_model.generate_content(content, generation_config=self.generation_config)
                final_answer = final_response.text.strip() if final_response and final_response.text.strip() else revised_cot
            

                return final_answer
            
            # 1) YANG or vendor-specific or ORAN queries => YangProcessor
            if "yang" in user_query.lower() or "24a" in user_query.lower() or "24b" in user_query.lower() and "oran" in user_query.lower():
                # Use the same vector_searcher-based YangProcessor
                all_yang_chunks = list(self.vector_searcher.chunks.values())
                return self.yang_processor.get_analysis(
                    query=user_query,
                    yang_chunks=all_yang_chunks,
                    llm=self.generative_model,
                    generation_config=self.yang_generation_config
                )
            
            else:
                # Use the default step-back approach
                logging.info("CoT toggle is OFF => using step-back approach (core concept + chunk retrieval).")
                
                # (1) Get core concept
                concept = self.generate_core_concept(user_query, conversation_history)
                logging.info(f"Step-Back concept extracted: {concept}")

                # (2) Vector search
                search_query = f"User query: {user_query}\nCore concept: {concept}"
                retrieved_chunks = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=search_query,
                    num_neighbors=self.num_neighbors
                )
                if not retrieved_chunks:
                    logging.warning("No chunks retrieved for the query.")
                    return "I'm sorry, I couldn't find relevant information."

                # (3) Rerank
                reranked_chunks = self.reranker.rerank(query=search_query, records=retrieved_chunks)
                if not reranked_chunks:
                    logging.warning("Reranking returned no results.")
                    return "I'm sorry, I couldn't find relevant info after reranking."

                # (4) Build final prompt
                prompt_content = self.generate_prompt_content(
                    query=user_query,
                    concept=concept,
                    chunks=reranked_chunks,
                    conversation_history=conversation_history
                )

                # (5) Generate final response
                answer = self.generate_response(prompt_content)
                return answer.strip() if answer else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logging.error(f"Error in get_response: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."