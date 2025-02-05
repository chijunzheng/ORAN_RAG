import re
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
from src.utils.token_utils import count_tokens

# Define a safe maximum context for summarization (for example, 900,000 tokens)
SAFE_SUMMARY_TOKEN_LIMIT = 900000
# Define a target token count for each summary segment (for example, 3000 tokens)
TARGET_SUMMARY_TOKENS = 3000

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

    # --- Helper Function for Hierarchical Summarization ---
    def summarize_text(self, text: str) -> str:
        """
        Uses the generative model to produce a summary of the provided text.
        Assumes that the text length is within a manageable token limit.
        """
        prompt = f"Summarize the following YANG file content while preserving all key details:\n\n{text}\n\nSummary:"
        prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
        try:
            response = self.generative_model.generate_content(prompt_content, generation_config=GenerationConfig(temperature=0.3, top_p=0.9, max_output_tokens=TARGET_SUMMARY_TOKENS))
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error summarizing text: {e}", exc_info=True)
            return text  # Fallback: return the original text if summarization fails
        
     # -----------------------
    # Retrieval, Reassembly, and Comparison Methods for YANG Files
    # -----------------------

    def reassemble_yang_files(self, yang_chunks: List[Dict]) -> Dict[str, Dict]:
        """
        Groups YANG chunks by file_name (from metadata), sorts them by chunk_index,
        and concatenates the chunk contents to reassemble the full file.
        Returns a mapping from file_name to a dict with full_text and metadata.
        """
        grouped = {}
        for chunk in yang_chunks:
            meta = chunk.get("metadata", {})
            file_name = meta.get("file_name", "unknown")
            grouped.setdefault(file_name, []).append(chunk)

        reassembled = {}
        for file_name, chunk_list in grouped.items():
            sorted_chunks = sorted(chunk_list, key=lambda x: int(x.get("metadata", {}).get("chunk_index", 0)))
            parts = [chunk.get("content", "") for chunk in sorted_chunks]
            full_text = "\n".join(parts)
            token_count = count_tokens(full_text)
            logging.info(f"Reassembled file '{file_name}' has {token_count} tokens.")
            # Optionally summarize if too long (if needed)
            if token_count > SAFE_SUMMARY_TOKEN_LIMIT:
                logging.info(f"File '{file_name}' exceeds safe token limit; summarizing in parts.")
                segments = []
                text_length = len(full_text)
                current_pos = 0
                while current_pos < text_length:
                    segment = full_text[current_pos: current_pos + (TARGET_SUMMARY_TOKENS * 5)]
                    segments.append(segment)
                    current_pos += len(segment)
                summarized_segments = [self.summarize_text(seg) for seg in segments]
                full_text = "\n\n".join(summarized_segments)
                logging.info(f"After summarization, file '{file_name}' reduced to {count_tokens(full_text)} tokens.")
            reassembled[file_name] = {
                "id": file_name,
                "full_text": full_text,
                "metadata": sorted_chunks[0].get("metadata", {})  # Use metadata from the first chunk
            }
        return reassembled

    # --- Retrieve YANG chunks based on vendor package filtering ---
    def retrieve_yang_chunks_by_vendor(self, vendor: str) -> List[Dict]:
        """
        Retrieves all YANG chunks whose metadata 'vendor_package' matches the vendor (e.g., "24A").
        """
        all_chunks = list(self.vector_searcher.chunks.values())
        filtered = [chunk for chunk in all_chunks if chunk.get("metadata", {}).get("vendor_package", "").upper() == vendor.upper()]
        logging.info(f"Retrieved {len(filtered)} YANG chunks for vendor package '{vendor}'.")
        return filtered
    
    # --- Determine if the query is a listing/inventory query for YANG models ---
    def is_yang_listing_query(self, query: str) -> bool:
        lower = query.lower()
        return (("list" in lower or "inventory" in lower) and "yang" in lower and ("24a" in lower or "24b" in lower))

    # --- Retrieval Shortcut for File-Specific Queries ---
    def retrieve_file_chunks(self, file_name_query: str) -> List[Dict]:
        """
        Directly retrieves all chunks for a specific YANG file by filtering on the file_name metadata.
        """
        # Access the preloaded chunks from the vector searcher.
        all_chunks = list(self.vector_searcher.chunks.values())
        # Use case-insensitive matching for file name.
        filtered = [chunk for chunk in all_chunks if file_name_query.lower() in chunk.get("metadata", {}).get("file_name", "").lower()]
        logging.info(f"Directly retrieved {len(filtered)} chunks for file query '{file_name_query}'.")
        return filtered
    
    def compare_vendor_packages(self, vendor_a: str, vendor_b: str) -> str:
        """
        Retrieves chunks for the two vendor packages (e.g. "24A" and "24B"), reassembles
        the full file texts per filename, and for each file common to both packages,
        constructs a prompt to compare them. Aggregates per-file comparisons into an overall report.
        """
        # Filter chunks by vendor package from the preloaded vector search chunks.
        # (Assumes self.vector_searcher.chunks contains all chunks keyed by id.)
        all_chunks = list(self.vector_searcher.chunks.values())
        chunks_a = [chunk for chunk in all_chunks if chunk.get("metadata", {}).get("vendor_package", "").upper() == vendor_a.upper()]
        chunks_b = [chunk for chunk in all_chunks if chunk.get("metadata", {}).get("vendor_package", "").upper() == vendor_b.upper()]

        # Reassemble files for each vendor.
        reassembled_a = self.reassemble_yang_files(chunks_a)
        reassembled_b = self.reassemble_yang_files(chunks_b)

        # Identify common files.
        common_files = set(reassembled_a.keys()) & set(reassembled_b.keys())
        if not common_files:
            return "No common YANG files found between vendor packages."

        comparisons = []
        for file_name in common_files:
            file_a = reassembled_a[file_name]
            file_b = reassembled_b[file_name]
            prompt = f"""
                You are a YANG model expert tasked with comparing two versions of the YANG file "{file_name}".
                Below are the two versions from different vendor packages:

                --- Vendor Package {vendor_a} ---
                {file_a["full_text"]}

                --- Vendor Package {vendor_b} ---
                {file_b["full_text"]}

                Provide a detailed analysis of the differences between these two versions. Focus on changes in structure, data types, features, revision dates, and any other relevant differences.
                """
            prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
            try:
                response = self.generative_model.generate_content(prompt_content, generation_config=self.generation_config)
                comparison_text = response.text.strip()
            except Exception as e:
                logging.error(f"Error comparing file {file_name}: {e}", exc_info=True)
                comparison_text = f"Comparison for {file_name} could not be generated due to an error."
            comparisons.append(f"Comparison for {file_name}:\n{comparison_text}\n")
        overall_report = "\n".join(comparisons)
        return overall_report

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

    def detect_yang_references(self, user_query: str) -> bool:
        """
        Returns True only if the query explicitly mentions 'yang' as a separate word
        or includes vendor package keywords ("24a", "24b", "samsung").
        """
        lower_q = user_query.lower()
        if re.search(r'\byang\b', lower_q):
            return True
        if "24a" in lower_q or "24b" in lower_q or "samsung" in lower_q:
            return True
        return False

    def detect_file_specific_query(self, query: str) -> str:
        """
        Checks if the query contains a specific .yang filename.
        Returns the filename if found, else returns an empty string.
        """
        match = re.search(r'([\w\-]+\.yang)', query, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""

    def detect_comparison_query(self, query: str) -> bool:
        """
        Returns True if the query indicates a desire to compare files, e.g., using terms like 'compare' or 'difference'.
        """
        lower = query.lower()
        return any(term in lower for term in ["compare", "difference", "vs", "versus", "change"])

    def extract_vendor_package_from_query(self, query: str) -> List[str]:
        """
        Extracts vendor package identifiers (e.g., "24A" or "24B") from the query.
        """
        targets = []
        lower = query.lower()
        if "24a" in lower:
            targets.append("24A")
        if "24b" in lower:
            targets.append("24B")
        return targets

    # --- Retrieval Shortcut for File-Specific Queries ---
    def retrieve_file_chunks(self, file_name_query: str) -> List[Dict]:
        all_chunks = list(self.vector_searcher.chunks.values())
        filtered = [chunk for chunk in all_chunks if file_name_query.lower() in chunk.get("metadata", {}).get("file_name", "").lower()]
        logging.info(f"Directly retrieved {len(filtered)} chunks for file query '{file_name_query}'.")
        return filtered
    
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
            f"Chunk {i+1} (File: {chunk.get('document_name','N/A')}, Version: {chunk.get('metadata', {}).get('version','unknown')}, "
            f"Workgroup: {chunk.get('metadata', {}).get('workgroup','unknown')}, Subcategory: {chunk.get('metadata', {}).get('subcategory','unknown')}, "
            f"Page: {chunk.get('page_number','N/A')}):\n{chunk.get('content','No content')}"
            for i, chunk in enumerate(oran_doc_chunks)
        ])

        
        # (4) For YANG chunks, group and reassemble full files.


        # If the query is YANG-related, reassemble full YANG files from chunks.
        yang_context_block = ""
        if self.detect_yang_references(query):
            reassembled_files = self.reassemble_yang_files(yang_doc_chunks)
            if reassembled_files:
                yang_parts = []
                for file_name, data in reassembled_files.items():
                    meta = data.get("metadata", {})
                    yang_parts.append(
                        f"File: {file_name}\n"
                        f"Module: {meta.get('module', 'unknown')}\n"
                        f"Revision: {meta.get('revision', 'unknown')}\n"
                        f"Namespace: {meta.get('namespace', 'unknown')}\n"
                        f"Yang Version: {meta.get('yang_version', 'unknown')}\n"
                        f"Organization: {meta.get('organization', 'unknown')}\n"
                        f"Description: {meta.get('description', 'No description')}\n\n"
                        f"{data.get('full_text','')}"
                    )
                yang_context_full = "\n\n".join(yang_parts)
                yang_context_block = f"<yang-context>\n{yang_context_full}\n</yang-context>"

        # (6) Build the final prompt text with the reference rules intact.
        prompt_text = f"""
            <purpose>
                You are an expert in O-RAN systems and YANG models. Always start by focusing on the "core concept" below 
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
                <instruction>Ensure that metadata (document name, version, workgroup, subcategory, page number) is referenced for ORAN documents.</instruction>
                <instruction>For YANG queries, analyze the reassembled full files provided in the <yang-context> block.</instruction>
            </instructions>

            <context>
            {oran_context}
            </context>

            {yang_context_block}  <!-- Included only if the query is yang-related -->

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
                    - Do NOT include references after each bullet or sentence.
                    - Instead, gather all module references at the end of the relevant section in one combined block.
                    - For "yang-context" or "yang_inventory_block", follow the reference rule: *Reference:()
                    - Format references in smaller font, using HTML `<small>` tags and indentation. For oran_context, use the following format:
                            <small>
                                &nbsp;&nbsp;*(Reference: [Document Name], page [Page Number(s)]; [Another Document], page [Page Number(s)])*
                            </small>
                    - For yang-context, use the following format:
                            <small>
                                &nbsp;&nbsp;*(Reference: [file_name], [vendor_package]; [another file_name], [vendor_package])*
                            </small>
                </answer-format>
                <markdown-guidelines>
                    <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                    <markdown-guideline>Use bullet points for lists and maintain consistent indentation.</markdown-guideline>
                </markdown-guidelines>
                <important-notes>
                    <important-note>Address the query fully by leveraging both ORAN and YANG context.</important-note>
                    <important-note>Provide a logical, step-by-step explanation.</important-note>
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

    def get_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        try:
            # Check if the query is a YANG model inventory (listing) query.
            if self.is_yang_listing_query(user_query):
                logging.info("Detected YANG model inventory listing query; bypassing reranking.")
                vendors = self.extract_vendor_package_from_query(user_query)
                if not vendors:
                    return "I'm sorry, I couldn't determine the vendor package from your query."
                inventory_results = {}
                for vendor in vendors:
                    chunks = self.retrieve_yang_chunks_by_vendor(vendor)
                    # Instead of reassembling full files, simply extract the unique set of module names
                    unique_modules = {}
                    for chunk in chunks:
                        metadata = chunk.get("metadata", {})
                        module_name = metadata.get("module", "unknown")
                        file_name = metadata.get("file_name", "unknown")
                        unique_modules[module_name] = file_name  # This ensures uniqueness.
                    inventory_results[vendor] = unique_modules

                report_lines = ["## Vendor Package YANG Model Inventory"]
                for vendor, modules in inventory_results.items():
                    report_lines.append(f"### Vendor Package {vendor}")
                    if modules:
                        for module_name, file_name in sorted(modules.items()):
                            report_lines.append(f"- {module_name} (File: {file_name})")
                    else:
                        report_lines.append("- No YANG models found.")
                return "\n".join(report_lines)
            
            # Check if the query targets a specific file by filename.
            file_specific = self.detect_file_specific_query(user_query)
            core_concept = self.generate_core_concept(user_query, conversation_history)
            logging.info(f"Step-Back concept extracted: {core_concept}")

            if file_specific:
                logging.info(f"File-specific query detected for '{file_specific}'. Retrieving chunks directly.")
                retrieved_chunks = self.retrieve_file_chunks(file_specific)
                if not retrieved_chunks:
                    logging.warning("No chunks found for the specified file.")
                    return "I'm sorry, no content could be retrieved for that file."
                reassembled_files = self.reassemble_yang_files(retrieved_chunks)
                if file_specific in reassembled_files:
                    file_data = reassembled_files[file_specific]
                    prompt = f"""
                        You are a YANG expert. Analyze the full contents of the YANG file "{file_specific}" and provide a detailed description of its structure, features, and any noteworthy details.

                        Full File Content:
                        {file_data['full_text']}
                        """
                    prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
                else:
                    logging.warning(f"Reassembled file for '{file_specific}' not found.")
                    return "I'm sorry, the specified file could not be reassembled."
            else:
                # For general queries, perform vector search and rerank.
                combined_query = f"User query: {user_query}\nCore concept: {core_concept}"
                retrieved_chunks = self.vector_searcher.vector_search(
                    index_endpoint_display_name=self.index_endpoint_display_name,
                    deployed_index_id=self.deployed_index_id,
                    query_text=combined_query,
                    num_neighbors=self.num_neighbors
                )
                logging.info(f"Retrieved {len(retrieved_chunks)} chunks for the query.")
                formatted_records = [self._convert_to_search_format(c) for c in retrieved_chunks]
                reranked_chunks = self.reranker.rerank(query=combined_query, records=formatted_records)
                logging.info(f"Reranked to {len(reranked_chunks)} top chunks.")
                if not reranked_chunks:
                    logging.warning("Reranking returned no results.")
                    return "I'm sorry, I couldn't find relevant information after reranking."
                retrieved_chunks = reranked_chunks

                prompt_content = self.generate_prompt_content(
                    query=user_query,
                    concept=core_concept,
                    chunks=retrieved_chunks,
                    conversation_history=conversation_history
                )
            assistant_response = self.generate_response(prompt_content)
            return assistant_response.strip() if assistant_response else "I'm sorry, I couldn't generate a response."
        except Exception as e:
            logging.error(f"Error in get_response: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."