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

    def force_yang_chunks_if_applicable(self, user_query: str) -> List[Dict]:
        """
        If user references '24A' or '24B' or 'yang', forcibly gather all YANG chunks from memory
        that have matching vendor_package or doc_type=yang_model, so user sees them.
        """
        # naive detection
        lower_q = user_query.lower()
        is_yang_query = any(k in lower_q for k in ["yang", "24a", "24b", "samsung"])

        if not is_yang_query:
            return []

        # We'll forcibly gather from self.vector_searcher.chunks
        # That dictionary is chunk_id -> { 'id', 'content', 'metadata': {...} }
        matched_chunks = []
        for chunk_id, chunk_data in self.vector_searcher.chunks.items():
            meta = chunk_data.get('metadata', {})
            doc_type = meta.get('doc_type', 'unknown')
            vendor_pkg = meta.get('vendor_package', 'unknown-package')

            # If doc_type is 'yang_model'
            if doc_type == 'yang_model':
                # Optionally check vendor_package. E.g. if user specifically says "24A" or "24B"
                # In real usage, you'd parse the query to see if it's "24A," "24B," or both.
                # We'll do a simple approach:
                if "24a" in lower_q and vendor_pkg.lower() == "24a":
                    matched_chunks.append(self._convert_to_search_format(chunk_data))
                elif "24b" in lower_q and vendor_pkg.lower() == "24b":
                    matched_chunks.append(self._convert_to_search_format(chunk_data))
                elif "yang" in lower_q and (vendor_pkg != 'unknown-package'):
                    # or we can just add any YANG chunk if they said "yang"
                    matched_chunks.append(self._convert_to_search_format(chunk_data))

        return matched_chunks

    def _convert_to_search_format(self, chunk_data: Dict) -> Dict:
        """
        The vector_search() returns a record format:
        { 'id': ..., 'content': ..., 'document_name':..., 'page_number':..., 'distance': ...}
        We want to produce a similar dict so we can pass it to reranker easily.
        """
        # We'll mimic the keys from the normal retrieval:
        return {
            'id': chunk_data['id'],
            'content': chunk_data.get('content', "No content"),
            'document_name': chunk_data.get('document_name', "Unknown doc"),
            'page_number': chunk_data.get('page_number', "Unknown page"),
            'metadata': chunk_data.get('metadata', {}),
            # no 'distance' since these are forcibly added. The reranker will handle scoring
        }
    
    # --------------------------------------------------------------------
    # (A) Detect if user query might be about YANG
    # --------------------------------------------------------------------
    def detect_yang_references(self, user_query: str) -> str:
        """
        A small utility to see if the user query references something like 'YANG', '24A', '24B', 'samsung', etc.
        If it does, we'll add some YANG-specific instructions in the final prompt.
        """
        lower_q = user_query.lower()
        # For example, just do a naive check
        if "yang" in lower_q or "24a" in lower_q or "24b" in lower_q or "samsung" in lower_q:
            return (
                "\n\n<yang-focus>\n"
                "Additionally, the user is referencing YANG files and/or vendor versions (e.g., '24A', '24B').\n"
                "Explain how these YANG model references differ or match up with O-RAN specs.\n"
                "Include any vendor-specific details if retrieved from the YANG chunks.\n"
                "</yang-focus>\n\n"
            )
        return ""

    # -------------------------------------------------------------------
    # 1) STEP-BACK METHOD: Identify the "Core ORAN Concept"
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # 2) PROMPT BUILDING: RETAIN your original large prompt for ORAN docs
    #    Then add a YANG block if the user query references YANG
    # -------------------------------------------------------------------
    def generate_prompt_content(self, query: str, concept: str, chunks: List[Dict], conversation_history: List[Dict]) -> Content:
        """
        Builds the final user prompt. We now split out YANG chunks from the main doc chunks.
        """
        # (1) Convert last 5 conversation turns into text (unchanged)
        history_text = ""
        for turn in conversation_history[-5:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        history_text += f"User: {query}\n"

        # (2) Separate the chunks into two lists:
        #     - ORAN doc chunks (or unknown doc_type)
        #     - YANG doc chunks
        oran_doc_chunks = []
        yang_doc_chunks = []
        for chunk in chunks:
            meta = chunk.get('metadata', {})
            doc_type = meta.get('doc_type', 'unknown')
            if doc_type == 'yang_model':
                yang_doc_chunks.append(chunk)
            else:
                oran_doc_chunks.append(chunk)

        # (3) Build the main 'context' as before from the ORAN doc chunks
        oran_context = "\n\n".join([
            f"Chunk {i+1} (File: {chunk.get('document_name','N/A')}, Page: {chunk.get('page_number','N/A')}):\n{chunk.get('content','No content')}"
            for i, chunk in enumerate(oran_doc_chunks)
        ])

        # (4) Build a separate 'yang_context' from the YANG doc chunks
        yang_context = "\n\n".join([
            f"YANG Chunk {i+1} (Module: {chunk.get('metadata', {}).get('module','unknown')}, Version: {chunk.get('metadata',{}).get('version','?')}):\n{chunk.get('content','No content')}"
            for i, chunk in enumerate(yang_doc_chunks)
        ])

        # (5) Check if the user query references YANG
        is_yang_query = self.detect_yang_references(query)

        # (6) Insert the YANG context block if the user query references YANG or vendor versions
        if is_yang_query and yang_context.strip():
            # We'll place it in a <yang-context> block below
            yang_context_block = f"""
            <yang-context>
            {yang_context}
            </yang-context>
            """
        else:
            yang_context_block = ""  # empty if no YANG reference

            # (7) Build final prompt text EXACTLY as before, but now we have {yang_context_block} inserted
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

            {yang_context_block}  <!-- ADDED for YANG specifically -->

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

    # -------------------------------------------------------------------
    # 3) Response Generation
    # -------------------------------------------------------------------
    def generate_response(self, prompt_content: Content) -> str:
        try:
            response = self.generative_model.generate_content(
                prompt_content,
                generation_config=self.generation_config,
            )
            assistant_response = response.text.strip()

            # Remove code blocks if present
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
    # 4) get_response 
    # -------------------------------------------------------------------
    def get_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        try:
            # (A) STEP-BACK concept
            core_concept = self.generate_core_concept(user_query, conversation_history)
            logging.info(f"Step-Back concept extracted: {core_concept}")

            # (B) Main vector search
            combined_query = f"User query: {user_query}\nCore concept: {core_concept}"
            retrieved_chunks = self.vector_searcher.vector_search(
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
                query_text=combined_query,
                num_neighbors=self.num_neighbors
            )
            logging.info(f"Retrieved {len(retrieved_chunks)} chunks for the query.")

            # (C) If user references YANG or '24A'/'24B', forcibly add any chunk with doc_type='yang_model' + matching vendor_package
            forced_yang_chunks = self.force_yang_chunks_if_applicable(user_query)
            if forced_yang_chunks:
                logging.info(f"Manually adding {len(forced_yang_chunks)} YANG chunks due to query references.")
                # Merge them with the normal vector results
                # Avoid duplicates by checking chunk IDs
                existing_ids = {c['id'] for c in retrieved_chunks}
                for fc in forced_yang_chunks:
                    if fc['id'] not in existing_ids:
                        retrieved_chunks.append(fc)

            if not retrieved_chunks:
                logging.warning("No chunks retrieved (or forced) for the query.")
                return "I'm sorry, I couldn't find relevant information to answer your question."

            # (D) Rerank the combined set
            reranked_chunks = self.reranker.rerank(
                query=combined_query,
                records=retrieved_chunks
            )
            logging.info(f"Reranked to {len(reranked_chunks)} top chunks.")
            if not reranked_chunks:
                logging.warning("Reranking returned no results.")
                return "I'm sorry, I couldn't find relevant information after reranking."

            # (E) Build final prompt
            prompt_content = self.generate_prompt_content(
                query=user_query,
                concept=core_concept,
                chunks=reranked_chunks,
                conversation_history=conversation_history
            )

            # (F) Generate final LLM response
            assistant_response = self.generate_response(prompt_content)
            return assistant_response.strip() if assistant_response else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logging.error(f"Error in get_response: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."