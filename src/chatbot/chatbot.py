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

# ----------------------------------------------------------------------------------
# Constants and Helper
# ----------------------------------------------------------------------------------

SAFE_SUMMARY_TOKEN_LIMIT = 1_000_000
# We will do the first summarization pass in ~15k-token chunks, 
# which is big enough to reduce calls drastically vs. 3k chunks:
CHUNK_SIZE_FIRST_PASS = 15000
# Overlap can be small to reduce repetition and calls. 
CHUNK_OVERLAP = 50
# We allow multiple summarization passes if the final text is still huge:
MAX_SUMMARIZATION_PASSES = 3

def count_tokens(text: str) -> int:
    if tiktoken:
        try:
            encoding = tiktoken.encoding_for_model("text-embedding-005")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    else:
        return len(text.split())

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

    # --------------------------------------------------------
    # Multi-Pass Summarization Helpers
    # --------------------------------------------------------
    def _summarize_in_chunks(self, text: str, chunk_size: int, overlap: int) -> str:
        """
        Splits the text into ~chunk_size token segments (with 'overlap' tokens overlap),
        calls self.summarize_text() on each segment, then joins them.
        This is a SINGLE PASS only.
        """
        total_tokens = count_tokens(text)
        logging.info(f"Starting single-pass summarization: {total_tokens} tokens => chunk_size={chunk_size}, overlap={overlap}")

        if not tiktoken:
            # Fallback: approximate chunking by splitting on whitespace tokens
            words = text.split()
            segments = []
            i = 0
            n = len(words)
            while i < n:
                seg_words = words[i : i + chunk_size]
                seg_text = " ".join(seg_words)
                segments.append(seg_text)
                i += chunk_size - overlap
        else:
            # Token-level chunking using tiktoken
            try:
                encoding = tiktoken.encoding_for_model("text-embedding-005")
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            token_ids = encoding.encode(text)
            segments = []
            i = 0
            total_ids = len(token_ids)
            while i < total_ids:
                seg_ids = token_ids[i : i + chunk_size]
                seg_text = encoding.decode(seg_ids)
                segments.append(seg_text)
                i += (chunk_size - overlap)

        # Summarize each segment
        summarized_segments = []
        for idx, seg in enumerate(segments, start=1):
            logging.debug(f"Summarizing segment {idx}/{len(segments)} (length={count_tokens(seg)})")
            summary = self.summarize_text(seg)
            summarized_segments.append(summary)

        combined_summary = "\n\n".join(summarized_segments)
        new_tokens = count_tokens(combined_summary)
        logging.info(f"Single-pass summarization done. Combined summary size: {new_tokens} tokens.")
        return combined_summary

    def hierarchical_summarize(self, text: str, passes: int = MAX_SUMMARIZATION_PASSES) -> str:
        """
        Performs multi-pass hierarchical summarization on 'text' 
        until it's under SAFE_SUMMARY_TOKEN_LIMIT or we reach 'passes'.
        Each pass uses `_summarize_in_chunks(..., chunk_size=CHUNK_SIZE_FIRST_PASS, overlap=CHUNK_OVERLAP)`.

        Returns the final summary (which might still be above the limit if passes are exhausted).
        """
        current = text
        attempt = 1
        while True:
            current_tokens = count_tokens(current)
            if current_tokens <= SAFE_SUMMARY_TOKEN_LIMIT:
                return current  # Done
            if attempt > passes:
                logging.warning(f"Reached max summarization passes ({passes}). Returning partial summary of size {current_tokens}.")
                return current
            logging.info(
                f"File has {current_tokens} tokens (limit={SAFE_SUMMARY_TOKEN_LIMIT}). "
                f"Performing summarization pass {attempt} of {passes}..."
            )
            current = self._summarize_in_chunks(
                current, 
                chunk_size=CHUNK_SIZE_FIRST_PASS, 
                overlap=CHUNK_OVERLAP
            )
            attempt += 1

    # --------------------------------------------------------
    # Summarization API
    # --------------------------------------------------------
    def summarize_text(self, text: str) -> str:
        """
        Summarizes 'text' in one LLM call, assuming 'text' is already short enough 
        (e.g. < 15k tokens). If it's too big, we do multi-pass chunking first.
        """
        # A small safety check, in case the user calls this function directly:
        # If text is still huge, do a single chunk-based pass. Typically we do multi-pass 
        # using hierarchical_summarize() instead, but let's ensure we never pass enormous text to LLM in one go.
        if count_tokens(text) > CHUNK_SIZE_FIRST_PASS:
            logging.debug("summarize_text: input too large for single pass; chunking once now.")
            return self._summarize_in_chunks(
                text,
                chunk_size=CHUNK_SIZE_FIRST_PASS,
                overlap=CHUNK_OVERLAP
            )

        prompt = f"Summarize the following content while preserving key details:\n\n{text}\n\nSummary:"
        prompt_content = Content(role="user", parts=[Part.from_text(prompt)])
        try:
            # We use a smaller model config to avoid lengthy output:
            short_summary_config = GenerationConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=3000  # Enough for a short summary
            )
            response = self.generative_model.generate_content(
                prompt_content, 
                generation_config=short_summary_config
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error summarizing text: {e}", exc_info=True)
            return text  # fallback

    def summarize_large_file(self, file_text: str) -> str:
        """
        If the file_text is bigger than SAFE_SUMMARY_TOKEN_LIMIT, 
        do multi-pass hierarchical summarization. Otherwise return as-is.
        """
        total_tokens = count_tokens(file_text)
        if total_tokens <= SAFE_SUMMARY_TOKEN_LIMIT:
            return file_text
        logging.info(
            f"File exceeds safe token limit ({total_tokens} tokens); "
            "applying multi-pass hierarchical summarization."
        )
        return self.hierarchical_summarize(file_text, passes=MAX_SUMMARIZATION_PASSES)

    
     # -----------------------
    # YANG Processing
    # -----------------------
    def reassemble_yang_files(self, yang_chunks: List[Dict]) -> Dict[str, Dict]:
        """
        Groups YANG chunks by file name, sorts them by chunk_index,
        concatenates into full_text, then calls summarize_large_file(...) 
        if the reassembled text is too big.
        """
        grouped = {}
        for chunk in yang_chunks:
            meta = chunk.get("metadata", {})
            file_name = meta.get("file_name", "unknown").strip().lower()
            grouped.setdefault(file_name, []).append(chunk)

        reassembled = {}
        for file_name, chunks in grouped.items():
            sorted_chunks = sorted(chunks, key=lambda c: int(c.get("metadata", {}).get("chunk_index", 0)))
            parts = [c.get("content", "") for c in sorted_chunks]
            full_text = "\n".join(parts)
            token_count = count_tokens(full_text)
            logging.info(f"Reassembled file '{file_name}' has {token_count} tokens.")
            if token_count > SAFE_SUMMARY_TOKEN_LIMIT:
                # Summarize multi-pass
                full_text = self.summarize_large_file(full_text)
            reassembled[file_name] = {
                "full_text": full_text,
                "metadata": sorted_chunks[0].get("metadata", {})
            }
        return reassembled

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

    def handle_yang_inventory_listing(self, vendors: List[str]) -> str:
        """
        For each vendor in 'vendors', retrieve all YANG chunks and list out the module names + file names.
        Returns a markdown or text-based report of what's in each vendor's YANG inventory.
        """
        inventory_results = {}
        for vendor in vendors:
            chunks = self.retrieve_yang_chunks_by_vendor(vendor)
            unique_modules = {}
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                module_name = metadata.get("module", "unknown")
                file_name = metadata.get("file_name", "unknown")
                unique_modules[module_name] = file_name
            inventory_results[vendor] = unique_modules

        # Build a text response
        report_lines = ["## Vendor Package YANG Model Inventory"]
        for vendor in vendors:
            report_lines.append(f"### Vendor Package {vendor}")
            modules = inventory_results[vendor]
            if modules:
                for module_name, file_name in sorted(modules.items()):
                    report_lines.append(f"- {module_name} (File: {file_name})")
            else:
                report_lines.append("- No YANG models found.")
        return "\n".join(report_lines)
    
    # --------------------------------------------------------
    # Vendor Comparison
    # --------------------------------------------------------    
   
    def retrieve_yang_chunks_by_vendor(self, vendor: str) -> List[Dict]:
        """
        Retrieves YANG chunks from memory whose metadata 'vendor_package' matches vendor, e.g., "24A".
        """
        all_chunks = list(self.vector_searcher.chunks.values())
        filtered = [
            chunk for chunk in all_chunks 
            if chunk.get("metadata", {}).get("vendor_package", "").upper() == vendor.upper()
        ]
        logging.info(f"Retrieved {len(filtered)} YANG chunks for vendor package '{vendor}'.")
        return filtered
    
    def compare_vendor_packages(self, vendor1: str, vendor2: str, conversation_history: List[Dict]) -> str:
        """
        Compare all YANG models from vendor1 vs vendor2, summarizing large text so
        we don't overload the model. Then produce a final LLM answer describing their differences.
        """

        # 1) Retrieve the relevant chunks for each vendor.
        vendor1_chunks = self.retrieve_yang_chunks_by_vendor(vendor1)
        vendor2_chunks = self.retrieve_yang_chunks_by_vendor(vendor2)

        # 2) Reassemble the full text for each vendor's YANG files.
        #    If a file exceeds SAFE_SUMMARY_TOKEN_LIMIT, it is automatically summarized
        #    inside reassemble_yang_files() or summarize_large_file().
        vendor1_files = self.reassemble_yang_files(vendor1_chunks)  # {file_name: {"full_text", "metadata"}}
        vendor2_files = self.reassemble_yang_files(vendor2_chunks)

        # 3) Concatenate all vendor1 text into a single big string; then summarize if needed.
        vendor1_full_text = ""
        for fname, info in vendor1_files.items():
            # Optionally label each file so final summary references them
            vendor1_full_text += f"\n[File: {fname}]\n{info['full_text']}\n"
        vendor1_summary = self.summarize_large_file(vendor1_full_text)

        # 4) Same for vendor2.
        vendor2_full_text = ""
        for fname, info in vendor2_files.items():
            vendor2_full_text += f"\n[File: {fname}]\n{info['full_text']}\n"
        vendor2_summary = self.summarize_large_file(vendor2_full_text)

        # 5) Build a final comparison prompt with only the summary text from each vendor.
        comparison_prompt = f"""
        You are a YANG model expert. The user is comparing YANG modules from vendor package "{vendor1}" 
        and vendor package "{vendor2}". Below are concise summaries for each. Please analyze them and produce 
        a clear, concise comparison.

        ## Vendor {vendor1}
        {vendor1_summary}

        ## Vendor {vendor2}
        {vendor2_summary}

        ### Instructions:
        1. Identify **all notable differences** between the two sets of YANG modules:
        - Data definitions (typedefs, groupings, containers, leaves)
        - Naming conventions (module name, prefixes, leaf names)
        - Revisions or organization statements
        - Any newly added or removed statements, imports, or references
        2. Discuss **structural changes** (e.g., reorganized or renamed groupings, 
        new or removed containers, different data types, changed enumerations)
        3. Keep your explanation **concise** yet detailed enough to highlight important changes

        <comparison-format>
        - Use `##` for major sections (Differences in Data Definitions, Differences in Revisions, etc.)
        - Use bullet points for listing out specific changes
        - Maintain a professional, clear, step-by-step format
        </comparison-format>

        Please generate the final comparison now.
        """

        try:
            # 6) Use the generative model to answer. 
            response = self.generative_model.generate_content(
                comparison_prompt,
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error generating comparison response for vendor packages {vendor1}, {vendor2}: {e}")
            return "I'm sorry, an error occurred while comparing vendor packages."
    
  
    def list_modified_yang_files(self, vendor1: str, vendor2: str) -> str:
        """
        Returns a list of YANG files that appear to have different revisions or a different 
        chunk count between vendor1 and vendor2. This is a simpler approach than 
        comparing actual content. 
        """

        # 1) Retrieve chunks for each vendor
        vendor1_chunks = self.retrieve_yang_chunks_by_vendor(vendor1)
        vendor2_chunks = self.retrieve_yang_chunks_by_vendor(vendor2)

        # 2) Group by file_name
        v1_files_map = {}  # {file_name -> dict with 'max_revision', 'chunk_count', 'any metadata you want'}
        for c in vendor1_chunks:
            meta = c.get("metadata", {})
            fname = meta.get("file_name", "unknown")
            rev   = meta.get("revision", "unknown")
            # We can track the "max" revision lexicographically or store them all. 
            # For demonstration, just store the first non-unknown we see.
            if fname not in v1_files_map:
                v1_files_map[fname] = {
                    "file_name": fname,
                    "revisions": set(),
                    "chunk_count": 0
                }
            v1_files_map[fname]["revisions"].add(rev)
            v1_files_map[fname]["chunk_count"] += 1

        v2_files_map = {}
        for c in vendor2_chunks:
            meta = c.get("metadata", {})
            fname = meta.get("file_name", "unknown")
            rev   = meta.get("revision", "unknown")
            if fname not in v2_files_map:
                v2_files_map[fname] = {
                    "file_name": fname,
                    "revisions": set(),
                    "chunk_count": 0
                }
            v2_files_map[fname]["revisions"].add(rev)
            v2_files_map[fname]["chunk_count"] += 1

        # 3) Compare
        modified_files = []
        # We’ll consider all file names that appear in either vendor’s list
        all_fnames = set(list(v1_files_map.keys()) + list(v2_files_map.keys()))

        for fname in sorted(all_fnames):
            v1_info = v1_files_map.get(fname)
            v2_info = v2_files_map.get(fname)
            if not v1_info or not v2_info:
                # If a file only exists in one vendor, we can call that "modified" or "added/removed".
                modified_files.append(fname)
                continue

            # Compare revisions as sets
            if v1_info["revisions"] != v2_info["revisions"]:
                modified_files.append(fname)
                continue

            # Compare chunk counts
            if v1_info["chunk_count"] != v2_info["chunk_count"]:
                modified_files.append(fname)
                continue

            # Potentially compare other metadata (like total size, earliest revision date, etc.)

        if not modified_files:
            return (
                f"From metadata alone, it appears that no YANG files differ between vendor {vendor1} and {vendor2}."
            )

        # Build final output
        lines = [
            f"## YANG Files Modified Between Vendor {vendor1} and {vendor2}",
            "Based on the provided context, the following files show differences:"
        ]
        for f in modified_files:
            lines.append(f"- {f}")
        return "\n".join(lines)

    def compare_file_between_vendors(self, file_name: str, vendor1: str, vendor2: str) -> str:
        """
        Compare a specific YANG file (file_name) between vendor1 and vendor2.
        Only retrieve tailf-rollback.yang for each vendor, rather than pulling all YANG files.
        """
        try:
            # 1) Retrieve chunks for vendor1 *but only for the specified file*
            vendor1_chunks = [
                c for c in self.vector_searcher.chunks.values()
                if c.get("metadata", {}).get("vendor_package","").upper() == vendor1.upper()
                and file_name.lower() in c.get("metadata", {}).get("file_name","").lower()
            ]
            logging.info(f"Found {len(vendor1_chunks)} chunks for {file_name} in vendor '{vendor1}'.")

            # 2) Same for vendor2
            vendor2_chunks = [
                c for c in self.vector_searcher.chunks.values()
                if c.get("metadata", {}).get("vendor_package","").upper() == vendor2.upper()
                and file_name.lower() in c.get("metadata", {}).get("file_name","").lower()
            ]
            logging.info(f"Found {len(vendor2_chunks)} chunks for {file_name} in vendor '{vendor2}'.")

            if not vendor1_chunks and not vendor2_chunks:
                return f"I couldn't find {file_name} in either {vendor1} or {vendor2}."

            # 3) Reassemble them individually
            vendor1_files = self.reassemble_yang_files(vendor1_chunks)
            vendor2_files = self.reassemble_yang_files(vendor2_chunks)

            # 4) vendor1 text
            if file_name in vendor1_files:
                vendor1_text = vendor1_files[file_name]["full_text"]
            else:
                vendor1_text = "No content found."

            # 5) vendor2 text
            if file_name in vendor2_files:
                vendor2_text = vendor2_files[file_name]["full_text"]
            else:
                vendor2_text = "No content found."

            # 6) Summarize if huge (unlikely for tailf-rollback, but just in case)
            vendor1_summary = self.summarize_large_file(vendor1_text)
            vendor2_summary = self.summarize_large_file(vendor2_text)

            # 7) Build final comparison prompt
            compare_prompt = f"""
            You are a YANG expert. Your task is to perform a detailed, line-by-line comparison between the file '{file_name}' from vendor {vendor1} and vendor {vendor2}. Your analysis must be both high-level and granular, reflecting the logical and hierarchical structure of the YANG modules.

            ### Vendor Package {vendor1} version of {file_name}:
            {vendor1_summary}

            ### Vendor Package {vendor2} version of {file_name}:
            {vendor2_summary}

            <Instructions>
            - First, provide an overall high-level summary of the differences between the two files. 
            - Highlight the major additions, removals, or modifications, and mention any improvements or regressions.
            - Then, perform a detailed, hierarchical, line-by-line analysis. For every detected change, indicate its precise logical location using the structure path. Use the format “grouping/container/leaf” or the appropriate combination (for example, “grouping/container”, or “container/leaf”, or “grouping/leaf”) to pinpoint the change.
            - Only highlight functional changes that are significant and relevant to the comparison. Avoid listing every minor differences such as description changes.
            - Organize your answer by first summarizing module-level differences, then breaking down the differences into hierarchical sections that reflect the YANG structure.
            - Do not list every low-level leaf difference separately unless it is significant; instead, include these differences under their respective higher-level grouping or container.
            </Instructions>

            <answer-format>
            Structure your answer with high-level headings (##) and subheadings (###). Present your analysis in bullet points or numbered lists. For example:

            ## High-Level Summary of Differences of <file_name> between Vendor {vendor1} and Vendor {vendor2}
            - Description of enhancements, regressions, or major changes in bullet points
            ## Detailed, Hierarchical, Line-by-Line Analysis
            ### Module-Level Differences
            ### Grouping/Container/Leaf Differences
            - "grouping1": Description of change
            OR
            - "container1": Description of change
            OR
            - "grouping1/container1":
                - Description of leaf changes
            OR
            - "grouping1/container2":
                - Description of leaf changes
            OR
            - "grouping2"/"container1":
                - Description of leaf changes
            etc.


            <markdown-guidelines>
                <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                <markdown-guideline>Use bullet points for lists and maintain consistent indentation.</markdown-guideline>
            </markdown-guidelines>
            </answer-format>

            Answer:
            """

            response = self.generative_model.generate_content(
                compare_prompt,
                generation_config=self.yang_generation_config
            )
            return response.text.strip()

        except Exception as e:
            logging.error(f"Error comparing file '{file_name}' for {vendor1} vs {vendor2}: {e}", exc_info=True)
            return f"I'm sorry, an error occurred comparing '{file_name}' for {vendor1} vs {vendor2}."


    # ----------------------------------------------------------------------------------
    # Query Detection
    # ----------------------------------------------------------------------------------
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

    def is_yang_modified_files_query(self, query: str) -> bool:
        """
        Returns True if the query looks like a request for "which files were modified" 
        between 24A and 24B. Example triggers: 'which of the yang model files was modified', 
        'which yang files changed', etc., plus '24a' and '24b'.
        """
        lower = query.lower()
        if "modified" in lower or "changed" in lower:
            if "yang" in lower or ("24a" in lower and "24b" in lower):
                return True
        return False  
    
    def is_yang_listing_query(self, query: str) -> bool:
        lower = query.lower()
        return (("list" in lower or "inventory" in lower) and "yang" in lower and ("24a" in lower or "24b" in lower))

    def is_yang_comparison_query(self, query: str) -> bool:
        """
        Returns True if the query indicates a comparison between vendor packages.
        We now check if the query contains either 'compare' or 'difference' along with 'yang', '24a', and '24b'.
        """
        lower = query.lower()
        return (("compare" in lower or "difference" in lower) and "yang" in lower and "24a" in lower and "24b" in lower)
    


    def detect_file_specific_query(self, query: str) -> str:
        """
        Checks if the query contains a specific .yang filename.
        Returns the filename if found, else returns an empty string.
        """
        match = re.search(r'([\w\-]+\.yang)', query, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""



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

        # (5) Build the final prompt text with the reference rules intact.
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

    def explain_single_yang_file(self, file_specific: str, conversation_history: List[Dict]) -> str:
        """
        Retrieve only the chunks for `file_specific` .yang, reassemble them,
        and pass them directly to the LLM for an explanation.
        """
        logging.info(f"Single-file YANG query for '{file_specific}'. Skipping vector search and reranking.")
        # 1) Retrieve the relevant chunks for that .yang file (case-insensitive).
        retrieved_chunks = self.retrieve_file_chunks(file_specific)
        if not retrieved_chunks:
            return f"No content found for file '{file_specific}'."

        # 2) Reassemble them (ensuring grouping by normalized file name).
        reassembled_files = self.reassemble_yang_files(retrieved_chunks)

        norm_file = file_specific.strip().lower()
        if norm_file not in reassembled_files:
            return f"Could not reassemble the file '{file_specific}'."

        # 3) Build the direct prompt
        file_data = reassembled_files[norm_file]
        file_text = file_data['full_text']
        prompt_text = f"""
            You are an advanced YANG model specialist. Provide a thorough explanation of the YANG file '{file_specific}' 
            based on the provided contents. Your explanation must include:

            <answer-format>
            Structure your answer with high-level headings (##) and subheadings (###). Present information in bullet points or numbered lists.

            **Overview and Purpose**  
            - The module’s intended role in network management or O-RAN contexts.
            - How this file integrates with or depends on other modules.

            **Module-Level Details**  
            - Module name, prefix, namespace.
            - Any 'organization', 'contact', or 'description' statements.
            - Revision history (including revision dates and short descriptions).

            **Data Structures**  
            - Concise high level summaries of the typedef, grouping, container, list, leaf based on the description in each section.
            - Demonstrate and explain how these structures relate to one another and to the module’s overall purpose.

            **Imports and References**  
            - Which external YANG modules are imported or referenced, and why.
            - Any relevant RFC or O-RAN specification references within the file.

            **Usage and Implications**  
            - How a network operator or developer might configure or query devices with this module.
            - Notable use cases or deployment scenarios in an O-RAN architecture.

            **Conclusion**  
            - A concise summary highlighting key takeaways and the file’s significance.

            </answer-format>

            <markdown-guidelines>
                <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                <markdown-guideline>Use bullet points for lists and maintain consistent indentation.</markdown-guideline>
            </markdown-guidelines>

            <important-notes>
                    <important-note>Focus on delivering a complete answer that fully addresses the query.</important-note>
                    <important-note>Ensure the explanation is presented step-by-step, covering relevant stages.</important-note>
            </important-notes>

            ## File Content

            {file_text}

            <answer>

            </answer>
            """
        prompt_content = Content(role="user", parts=[Part.from_text(prompt_text)])

        try:
            # 4) Directly call LLM, no reranking or big comparison logic
            response = self.generative_model.generate_content(
                prompt_content, 
                generation_config=self.generation_config
            )
            return response.text.strip() if response else "No response generated."
        except Exception as e:
            logging.error(f"Error explaining single YANG file '{file_specific}': {e}", exc_info=True)
            return "Error explaining single-file YANG content."


    # --------------------------------------------------------
    # Main Entry: get_response
    # --------------------------------------------------------
    def get_response(self, user_query: str, conversation_history: List[Dict]) -> str:
        """
        Main entry point. 
        Depending on user_query:
          - If listing YANG inventory, return a summary of modules.
          - If comparing 24A vs 24B, call compare_vendor_packages.
          - If referencing a specific YANG file, reassemble it.
          - Otherwise, fallback to vector search + re-ranking + final generation.
        """
        try:
            lower_query = user_query.lower()

            # (1) Check YANG inventory listing
            if self.is_yang_listing_query(user_query):
                logging.info("Detected YANG inventory listing query. Bypassing reranking.")
                vendors = self.extract_vendor_package_from_query(user_query)
                if not vendors:
                    return "I'm sorry, I couldn't determine the vendor package(s) from your query."
                return self.handle_yang_inventory_listing(vendors)

            # 2) YANG "which files changed/modified" check
            if self.is_yang_modified_files_query(user_query):
                # If user specifically says "which files were modified between 24A and 24B"
                vendors = self.extract_vendor_package_from_query(user_query)
                if len(vendors) < 2:
                    return "Both vendor packages must be specified to list modified files."
                return self.list_modified_yang_files(vendors[0], vendors[1])
            
            # 3) YANG vendor package comparison (broad, or file-specific)
            if self.is_yang_comparison_query(user_query):
                # Check if there's a file name in the query
                file_specific = self.detect_file_specific_query(user_query)
                vendors = self.extract_vendor_package_from_query(user_query)
                # If exactly 2 vendor packages and 1 file name => do *file-specific* vendor compare
                if file_specific and len(vendors) == 2:
                    return self.compare_file_between_vendors(file_specific, vendors[0], vendors[1])

            
                # Otherwise do existing broad vendor package comparison
                if len(vendors) < 2:
                    return "Both vendor packages (e.g. 24A and 24B) must be specified for comparison."
                return self.compare_vendor_packages(vendors[0], vendors[1], conversation_history)


            #4) Single .yang file check
            file_specific = self.detect_file_specific_query(user_query)
            if file_specific:
                # If user explicitly names "something.yang"
                return self.explain_single_yang_file(file_specific, conversation_history)
        
            # 5) Otherwise, fallback to vector search + re-ranking
            core_concept = self.generate_core_concept(user_query, conversation_history)
            combined_query = f"User query: {user_query}\nCore concept: {core_concept}"
            retrieved_chunks = self.vector_searcher.vector_search(
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
                query_text=combined_query,
                num_neighbors=self.num_neighbors
            )
            logging.info(f"Retrieved {len(retrieved_chunks)} chunks for the query.")
            if not retrieved_chunks:
                return "I'm sorry, I couldn't find relevant information."

            # Rerank
            formatted_records = [self._convert_to_search_format(c) for c in retrieved_chunks]
            reranked_chunks = self.reranker.rerank(query=combined_query, records=formatted_records)
            logging.info(f"Reranked to top {len(reranked_chunks)} chunks.")
            if not reranked_chunks:
                return "I'm sorry, no relevant information after reranking."

            # Build final prompt
            prompt_content = self.generate_prompt_content(
                query=user_query,
                concept=core_concept,
                chunks=reranked_chunks,
                conversation_history=conversation_history
            )
            response = self.generative_model.generate_content(prompt_content, generation_config=self.generation_config)
            return response.text.strip() if response else "I'm sorry, I couldn't generate a response."

        except Exception as e:
            logging.error(f"Error in get_response: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing your request."