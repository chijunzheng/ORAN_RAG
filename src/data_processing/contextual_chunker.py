import logging
import json
import uuid
import os
import requests
from typing import List, Dict
from langchain.schema import Document
from google.cloud import storage
from google.oauth2.credentials import Credentials
from src.data_processing.document_chunker import DocumentChunker, extract_oran_metadata_from_filename

# Try to load environment variables from .env file using absolute path
try:
    from dotenv import load_dotenv
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    dotenv_path = os.path.join(project_root, '.env')
    
    # Check if .env file exists
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logging.info(f"Loaded .env file from {dotenv_path}")
    else:
        logging.warning(f".env file not found at {dotenv_path}")
except ImportError:
    logging.warning("python-dotenv not installed. Reading .env file manually.")
    try:
        # Define possible locations for the .env file
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        possible_paths = [
            '.env',  # Current directory
            os.path.join(project_root, '.env'),  # Project root
        ]
        
        # Try each path
        for env_path in possible_paths:
            if os.path.exists(env_path):
                logging.info(f"Found .env file at {env_path}")
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                            logging.info(f"Manually set {key} from .env file")
                break  # Exit loop if a file was found and processed
        else:
            logging.warning("No .env file found in expected locations")
    except Exception as e:
        logging.error(f"Error manually reading .env file: {e}")
        logging.warning("Falling back to environment variables")

class ContextualChunker(DocumentChunker):
    """
    Extends DocumentChunker to implement contextual retrieval as described in Anthropic's article.
    
    This approach improves retrieval accuracy by adding document-level context to each chunk
    before embedding it. Context is generated using Gemini-1.5-flash to situate the chunk 
    within the entire document.
    """
    
    def __init__(
        self,
        chunk_size: int = 1536,
        chunk_overlap: int = 256,
        separators: List[str] = None,
        gcs_bucket_name: str = None,
        gcs_embeddings_path: str = "embeddings/",
        credentials: Credentials = None,
        min_char_count: int = 100
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            gcs_bucket_name=gcs_bucket_name,
            gcs_embeddings_path=gcs_embeddings_path,
            credentials=credentials,
            min_char_count=min_char_count
        )
        
        self.credentials = credentials
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            logging.warning("GEMINI_API_KEY not found in environment variables. Context generation will be limited.")
        
        logging.info("ContextualChunker initialized with Gemini-1.5-flash for context generation.")
    
    def generate_chunk_context(self, document_text: str, chunk_text: str) -> str:
        """
        Generates context for a chunk using Gemini-1.5-flash.
        
        Args:
            document_text (str): The full document text.
            chunk_text (str): The content of the chunk.
            
        Returns:
            str: Generated context that situates the chunk within the document.
        """
        if not self.api_key:
            return "This is part of a document."
            
        try:
            # Truncate document if it's too large
            max_doc_length = 10000  # Arbitrary limit to avoid context window issues
            if len(document_text) > max_doc_length:
                # Take beginning and end of document
                half_length = max_doc_length // 2
                truncated_doc = document_text[:half_length] + "\n...\n" + document_text[-half_length:]
                document_text = truncated_doc
            
            # Set up the API request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"
            
            # Create the prompt
            prompt = f"""
<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_text}
</chunk>
Please give a detailed succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
            
            # Set up the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 300
                }
            }
            
            # Make the API request
            response = requests.post(url, json=payload)
            
            if response.status_code != 200:
                logging.error(f"API request failed with status code {response.status_code}: {response.text}")
                return self._generate_simple_context(document_text, chunk_text)
            
            # Parse the response
            response_json = response.json()
            
            if not response_json.get("candidates") or not response_json["candidates"][0].get("content"):
                return self._generate_simple_context(document_text, chunk_text)
                
            context = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            logging.debug(f"Generated context: {context}")
            return context
            
        except Exception as e:
            logging.error(f"Failed to generate context with Gemini: {e}", exc_info=True)
            return self._generate_simple_context(document_text, chunk_text)
    
    def _generate_simple_context(self, document_text: str, chunk_text: str) -> str:
        """
        Fallback method to generate a simple context without using the Gemini model.
        
        Args:
            document_text (str): The full document text.
            chunk_text (str): The content of the chunk.
            
        Returns:
            str: A simple context based on document metadata.
        """
        # Get first 100 characters of document as context
        doc_start = document_text[:100].replace("\n", " ").strip()
        if len(doc_start) == 100:
            doc_start += "..."
            
        # Calculate approximate position
        chunk_position = document_text.find(chunk_text[:50])
        if chunk_position == -1:
            position_desc = "part of the document"
        else:
            total_length = len(document_text)
            position_percent = (chunk_position / total_length) * 100
            if position_percent < 20:
                position_desc = "beginning section of the document"
            elif position_percent < 40:
                position_desc = "early section of the document"
            elif position_percent < 60:
                position_desc = "middle section of the document"
            elif position_percent < 80:
                position_desc = "later section of the document"
            else:
                position_desc = "final section of the document"
        
        # Extract section headings before the chunk
        section_context = self._extract_section_headings(document_text, chunk_position)
        if section_context:
            return f"This content appears in the {position_desc}, under the section '{section_context}'. Document begins with: {doc_start}"
        else:
            return f"This content appears in the {position_desc}. Document begins with: {doc_start}"
    
    def _extract_section_headings(self, document_text: str, chunk_position: int) -> str:
        """
        Extract section headings that appear before the chunk.
        
        Args:
            document_text (str): The full document text.
            chunk_position (int): Position of the chunk in the document.
            
        Returns:
            str: Section heading if found, empty string otherwise.
        """
        # Only look at the text before the chunk
        text_before_chunk = document_text[:chunk_position]
        
        # Look for patterns that might indicate section headings
        import re
        
        # Try to find numbered headings (e.g., "1.2.3 Section Name")
        section_pattern = r'\n(\d+(\.\d+)* +[A-Z][^\n]+)\n'
        sections = re.findall(section_pattern, text_before_chunk)
        
        # Also look for capitalized headings
        if not sections:
            caps_pattern = r'\n([A-Z][A-Z\s]+[A-Z])\n'
            sections = re.findall(caps_pattern, text_before_chunk)
        
        if sections:
            # Return the last (most recent) section heading
            return sections[-1][0].strip() if isinstance(sections[-1], tuple) else sections[-1].strip()
        return ""
    
    def assign_ids(self, split_docs: List[Document]) -> List[Dict]:
        """
        Overrides the parent method to add context generation.
        Assigns unique IDs to chunks and adds context based on the full document.
        
        Args:
            split_docs (List[Document]): List of split Document objects.
            
        Returns:
            List[Dict]: List of chunk dictionaries with IDs and context.
        """
        logging.info("Assigning UUIDs and embedding context into chunks.")
        
        # Group documents by name to reconstruct full documents
        doc_content_map = {}
        for doc in split_docs:
            doc_name = doc.metadata.get('document_name', 'Unknown Document')
            if doc_name not in doc_content_map:
                doc_content_map[doc_name] = {
                    'chunks': [],
                    'content': ""
                }
            doc_content_map[doc_name]['chunks'].append(doc)
            doc_content_map[doc_name]['content'] += doc.page_content + "\n\n"
        
        chunks_with_ids = []
        
        for doc_name, doc_data in doc_content_map.items():
            document_content = doc_data['content']
            
            for doc in doc_data['chunks']:
                char_count = len(doc.page_content)
                if char_count < self.min_char_count:
                    logging.debug(f"Skipped chunk due to low character count: {char_count} < {self.min_char_count}.")
                    continue
                
                chunk_uuid = str(uuid.uuid4())
                document_name = doc.metadata.get('document_name', 'Unknown Document')
                page_number = doc.metadata.get('page_number', 'Unknown Page')
                
                # Extract additional ORAN metadata from the document file name
                extracted_metadata = extract_oran_metadata_from_filename(document_name)
                version = doc.metadata.get('version', extracted_metadata.get('version'))
                workgroup = doc.metadata.get('workgroup', extracted_metadata.get('workgroup'))
                subcategory = doc.metadata.get('subcategory', extracted_metadata.get('subcategory'))
                
                # Generate context for this chunk
                context = self.generate_chunk_context(document_content, doc.page_content)
                
                # Create the content with context but without metadata header
                # The metadata is still preserved in the metadata field
                full_content = f"Context: {context}\n\n{doc.page_content}"
                
                chunk_dict = {
                    'id': chunk_uuid,
                    'content': full_content,
                    'document_name': document_name,
                    'page_number': page_number,
                    'char_count': char_count,
                    'metadata': {
                        "document_name": document_name,
                        "version": version,
                        "workgroup": workgroup,
                        "subcategory": subcategory,
                        "page_number": page_number,
                        "context": context
                    }
                }
                chunks_with_ids.append(chunk_dict)
        
        logging.info("Completed assigning UUIDs and embedding context into chunks.")
        return chunks_with_ids
    
    def split_documents(self, documents: List[Document]) -> List[Dict]:
        """
        Splits a list of Document objects into chunks and assigns unique IDs with context.
        This method overrides the parent method but uses its functionality.
        
        Args:
            documents (List[Document]): List of Document objects.
            
        Returns:
            List[Dict]: List of dictionaries containing chunk information with context.
        """
        logging.info("Starting to split documents into chunks with context.")
        split_docs = self.splitter.split_documents(documents)
        logging.info(f"Total chunks created before filtering: {len(split_docs)}")
        
        chunks_with_ids = self.assign_ids(split_docs)
        logging.info(f"Total chunks after filtering and adding context: {len(chunks_with_ids)}")
        return chunks_with_ids
    
    def save_chunks_to_json(self, chunks: List[Dict], file_path: str = "chunks.json"):
        """
        Saves the chunks to a JSON file format, which is compatible with the Embedder.
        Each line is a valid JSON object with 'id' and 'content' fields.
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries.
            file_path (str): Path to save the JSON file.
            
        Raises:
            Exception: If saving fails for any reason.
        """
        logging.info(f"Saving {len(chunks)} chunks to JSON format at {file_path}")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write directly to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    # Extract only the fields needed for embedding
                    line_dict = {
                        'id': chunk.get('id'),
                        'content': chunk.get('content', ""),
                        'metadata': chunk.get('metadata', {})
                    }
                    json_record = json.dumps(line_dict, ensure_ascii=False)
                    f.write(json_record + "\n")
            logging.info(f"Chunk data saved in JSON format to {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Failed to save chunks to JSON at {file_path}: {e}", exc_info=True)
            raise
    
