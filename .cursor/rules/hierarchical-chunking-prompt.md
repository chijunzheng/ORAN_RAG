# Simplified Hierarchical Chunking Implementation

## Task Overview

Implement a hierarchical chunking system for ORAN technical documents that preserves document structure by adding parent summaries to child chunks. This approach must work within the existing 768 token limit while maintaining compatibility with the current codebase.

## Existing Codebase Context

The implementation will build on these existing components:

1. `TextFormatter` - Currently cleans headers, footers, and TOC pages 
2. `Embedder` - Generates embeddings with 768 token limit
3. `VectorIndexer` - Creates and deploys vector search indexes
4. `VectorSearcher` - Performs vector search queries
5. `Reranker` - Reranks search results

## Implementation Requirements

### 1. Extend TextFormatter for Structure Detection

Create a new `HierarchicalTextFormatter` class that extends the current `TextFormatter`:

```python
# src/data_processing/hierarchical_text_formatter.py

import re
import logging
from typing import List, Dict, Tuple
from langchain.schema import Document
from collections import defaultdict

from src.data_processing.text_formatter import TextFormatter

class HierarchicalTextFormatter(TextFormatter):
    def __init__(self):
        """
        Extends TextFormatter with document structure detection capabilities.
        """
        super().__init__()
        
        # Define patterns to identify section headings at different levels
        self.heading_patterns = [
            # Level 1 headings (main sections)
            r'^(\d+\.)\s+([A-Z][A-Za-z\s]+)$',  # "1. INTRODUCTION"
            r'^([A-Z][A-Z\s]+)$',               # "INTRODUCTION"
            
            # Level 2 headings (subsections)
            r'^(\d+\.\d+\.?)\s+([A-Za-z][\w\s]+)$',  # "1.1 System Overview"
            
            # Level 3 headings (sub-subsections)
            r'^(\d+\.\d+\.\d+\.?)\s+([A-Za-z][\w\s]+)$'  # "1.1.1 Component Details"
        ]
    
    def detect_document_structure(self, pages: List[Dict]) -> List[Dict]:
        """
        Detects document structure by identifying section headings in page content.
        
        Args:
            pages: List of page dictionaries with text content
            
        Returns:
            List of pages with added structure information
        """
        structured_pages = []
        current_section = {"level": 0, "title": "", "id": ""}
        current_subsection = {"level": 0, "title": "", "id": ""}
        current_subsubsection = {"level": 0, "title": "", "id": ""}
        
        for page in pages:
            text = page['text']
            lines = text.split('\n')
            page_structure = {
                "headings": [],
                "sections": {
                    "section": current_section.copy(),
                    "subsection": current_subsection.copy(),
                    "subsubsection": current_subsubsection.copy()
                }
            }
            
            # Identify headings in this page
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Try to match against each heading pattern
                for level, pattern in enumerate(self.heading_patterns, 1):
                    match = re.match(pattern, line)
                    if match:
                        # Extract section ID and title
                        section_id = match.group(1) if len(match.groups()) > 0 else ""
                        title = match.group(2) if len(match.groups()) > 1 else line
                        
                        heading = {
                            "level": level,
                            "title": title,
                            "id": section_id,
                            "line_index": i
                        }
                        page_structure["headings"].append(heading)
                        
                        # Update current section tracking
                        if level == 1:
                            current_section = heading.copy()
                            # Reset lower levels when a new section is found
                            current_subsection = {"level": 0, "title": "", "id": ""}
                            current_subsubsection = {"level": 0, "title": "", "id": ""}
                        elif level == 2:
                            current_subsection = heading.copy()
                            # Reset lowest level
                            current_subsubsection = {"level": 0, "title": "", "id": ""}
                        elif level == 3:
                            current_subsubsection = heading.copy()
                        
                        # Update the current sections in the page structure
                        page_structure["sections"] = {
                            "section": current_section.copy(),
                            "subsection": current_subsection.copy(),
                            "subsubsection": current_subsubsection.copy()
                        }
                        
                        break  # Stop checking patterns once a match is found
            
            # Add structure info to the page
            page_with_structure = page.copy()
            page_with_structure['structure'] = page_structure
            structured_pages.append(page_with_structure)
        
        return structured_pages
    
    def format_documents_with_structure(self, documents: Dict[str, List[Dict]]) -> List[Document]:
        """
        Process documents to clean text and extract structure.
        
        Args:
            documents: Dictionary mapping document names to lists of page dictionaries
            
        Returns:
            List of Document objects with cleaned text and structure metadata
        """
        all_cleaned_documents = []
        
        for doc_name, pages in documents.items():
            logging.info(f"Processing document with structure: {doc_name}")
            
            # First detect headers and footers (using parent class method)
            first_lines = [page.get('first_line', '') for page in pages]
            last_lines = [page.get('last_line', '') for page in pages]
            headers, footers = self.detect_headers_footers(first_lines, last_lines)
            
            # Then detect document structure
            structured_pages = self.detect_document_structure(pages)
            
            # Process each page
            for i, page in enumerate(structured_pages):
                # Clean the text using parent class method
                cleaned_text, exclude_page = self.text_formatter(page['text'], headers, footers)
                
                if exclude_page or not cleaned_text:
                    logging.info(f"Excluded page {i+1} from {doc_name}")
                    continue
                
                # Extract structure information
                structure = page.get('structure', {})
                sections = structure.get('sections', {})
                
                # Create enhanced metadata
                metadata = {
                    'document_name': doc_name,
                    'page_number': i+1,
                    'section': sections.get('section', {}).get('title', ''),
                    'section_id': sections.get('section', {}).get('id', ''),
                    'subsection': sections.get('subsection', {}).get('title', ''),
                    'subsection_id': sections.get('subsection', {}).get('id', ''),
                    'subsubsection': sections.get('subsubsection', {}).get('title', ''),
                    'subsubsection_id': sections.get('subsubsection', {}).get('id', ''),
                    'headings': structure.get('headings', [])
                }
                
                # Create document with cleaned text and enhanced metadata
                cleaned_document = Document(
                    page_content=cleaned_text,
                    metadata=metadata
                )
                all_cleaned_documents.append(cleaned_document)
        
        logging.info(f"Processed {len(all_cleaned_documents)} pages with structure")
        return all_cleaned_documents
```

### 2. Implement Document Tree Construction

Create a document tree structure to track parent-child relationships:

```python
# src/data_processing/document_tree.py

import uuid
from typing import List, Dict, Optional
from collections import defaultdict
from langchain.schema import Document

class DocumentNode:
    """
    Represents a node in the document hierarchy (document, section, subsection).
    """
    def __init__(self, title: str, level: int, section_id: str = ""):
        self.id = str(uuid.uuid4())
        self.title = title
        self.level = level  # 0=document, 1=section, 2=subsection, 3=subsubsection
        self.section_id = section_id
        self.content = ""
        self.children = []
        self.parent = None
        self.pages = []
        self.summary = None
    
    def add_child(self, child_node):
        """Add a child node to this node"""
        self.children.append(child_node)
        child_node.parent = self
    
    def get_path(self) -> List[str]:
        """Get full path from root to this node"""
        if not self.parent:
            return [self.title]
        return self.parent.get_path() + [self.title]

class DocumentTree:
    """
    Tree structure representing document hierarchy.
    """
    def __init__(self, document_name: str):
        self.root = DocumentNode(document_name, 0)
        self.nodes_by_id = {self.root.id: self.root}
    
    def build_from_documents(self, documents: List[Document]) -> None:
        """
        Build document tree from processed documents with structure metadata.
        
        Args:
            documents: List of Document objects with structure metadata
        """
        # Group documents by section structure
        section_groups = defaultdict(list)
        for doc in documents:
            if doc.metadata['document_name'] != self.root.title:
                continue  # Skip documents that don't match this tree
            
            # Create key based on section hierarchy
            key = (
                doc.metadata.get('section_id', ''),
                doc.metadata.get('subsection_id', ''),
                doc.metadata.get('subsubsection_id', '')
            )
            section_groups[key].append(doc)
        
        # Process each section group
        for (section_id, subsection_id, subsubsection_id), docs in section_groups.items():
            # Skip if no section information
            if not section_id and not subsection_id and not subsubsection_id:
                continue
            
            # Get section information from first document
            first_doc = docs[0].metadata
            
            # Find or create section node
            section_node = self.root
            if section_id:
                section_node = self._find_or_create_child(
                    self.root,
                    first_doc.get('section', ''),
                    1,
                    section_id
                )
            
            # Find or create subsection node
            subsection_node = section_node
            if subsection_id and section_node:
                subsection_node = self._find_or_create_child(
                    section_node,
                    first_doc.get('subsection', ''),
                    2,
                    subsection_id
                )
            
            # Find or create subsubsection node
            target_node = subsection_node
            if subsubsection_id and subsection_node:
                target_node = self._find_or_create_child(
                    subsection_node,
                    first_doc.get('subsubsection', ''),
                    3,
                    subsubsection_id
                )
            
            # Combine content from all documents in this section
            combined_content = "\n\n".join([doc.page_content for doc in docs])
            
            # Add content to target node
            if target_node.content:
                target_node.content += f"\n\n{combined_content}"
            else:
                target_node.content = combined_content
            
            # Track page numbers
            for doc in docs:
                page_num = doc.metadata.get('page_number')
                if page_num and page_num not in target_node.pages:
                    target_node.pages.append(page_num)
    
    def _find_or_create_child(self, parent: DocumentNode, title: str, level: int, section_id: str) -> DocumentNode:
        """Find existing child by section ID or create a new one"""
        # Try to find existing child with matching section ID
        for child in parent.children:
            if child.section_id == section_id:
                return child
        
        # Create new child node
        child = DocumentNode(title, level, section_id)
        parent.add_child(child)
        self.nodes_by_id[child.id] = child
        return child
    
    def generate_summaries(self, max_length: int = 100) -> None:
        """
        Generate summaries for all nodes in the tree.
        Simple version: use first paragraph as summary.
        
        Args:
            max_length: Maximum length of summary
        """
        for node_id, node in self.nodes_by_id.items():
            if not node.content:
                continue
                
            # Use first paragraph as simple summary
            first_para = node.content.split('\n\n')[0] if '\n\n' in node.content else node.content
            if len(first_para) > max_length:
                node.summary = first_para[:max_length] + "..."
            else:
                node.summary = first_para
```

### 3. Implement Hierarchical Chunker with Parent Summaries

Create a chunker that adds parent summaries to chunks:

```python
# src/data_processing/hierarchical_chunker.py

import logging
from typing import List, Dict, Optional
import uuid

def count_tokens(text: str) -> int:
    """
    Count tokens in text (approximate).
    For precise counting, use tiktoken or another tokenizer.
    """
    return len(text.split())

class HierarchicalChunker:
    """
    Creates chunks with parent summary context.
    """
    def __init__(self, max_tokens: int = 768, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # Reserve tokens for context preamble
        self.reserved_context_tokens = 100  # ~100 tokens for context
        self.available_content_tokens = max_tokens - self.reserved_context_tokens
    
    def create_chunks_with_parent_context(self, document_tree):
        """
        Create chunks from document tree with parent context.
        
        Args:
            document_tree: DocumentTree instance
            
        Returns:
            List of chunks with parent context
        """
        all_chunks = []
        
        # Process all nodes except root
        for node_id, node in document_tree.nodes_by_id.items():
            if node.level == 0 or not node.content.strip():
                continue  # Skip root node and empty nodes
            
            # Collect parent context
            context = self._collect_parent_context(node)
            
            # Create chunks for this node
            node_chunks = self._create_node_chunks(node, context)
            all_chunks.extend(node_chunks)
        
        logging.info(f"Created {len(all_chunks)} hierarchical chunks")
        return all_chunks
    
    def _collect_parent_context(self, node):
        """
        Collect context from parent node only.
        
        Args:
            node: DocumentNode to get parent context for
            
        Returns:
            Dictionary with parent context information
        """
        context = {
            "path": [],
            "summary": "",
            "section_id": "",
            "pages": []
        }
        
        # Get parent information
        if node.parent:
            parent = node.parent
            
            # Add parent summary
            if parent.summary:
                context["summary"] = parent.summary
            
            # Add parent section ID
            context["section_id"] = parent.section_id
            
            # Add page numbers
            context["pages"] = sorted(parent.pages) if parent.pages else []
            
            # Build section path
            current = node
            while current.parent:
                if current.parent.title:
                    context["path"].insert(0, current.parent.title)
                current = current.parent
        
        return context
    
    def _format_context_preamble(self, context, node):
        """
        Format parent context as a preamble for the chunk.
        
        Args:
            context: Parent context dictionary
            node: Current node
            
        Returns:
            Formatted context preamble string
        """
        lines = []
        
        # Section title
        if node.section_id:
            lines.append(f"SECTION: {node.section_id} {node.title}")
        else:
            lines.append(f"SECTION: {node.title}")
        
        # Parent context/summary
        if context["summary"]:
            lines.append(f"CONTEXT: {context['summary']}")
        
        # Section path (breadcrumb)
        if context["path"]:
            path_str = " > ".join(context["path"])
            lines.append(f"PATH: {path_str}")
        
        # Page range
        if context["pages"]:
            pages_str = ", ".join(str(p) for p in context["pages"])
            lines.append(f"PAGES: {pages_str}")
        
        # Separator
        lines.append("---")
        
        return "\n".join(lines)
    
    def _create_node_chunks(self, node, context):
        """
        Create chunks for node content with parent context.
        
        Args:
            node: DocumentNode to create chunks for
            context: Parent context dictionary
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Format context preamble
        context_preamble = self._format_context_preamble(context, node)
        context_tokens = count_tokens(context_preamble)
        
        # Verify context size is within reserved limit
        if context_tokens > self.reserved_context_tokens:
            logging.warning(f"Context preamble is larger than reserved size: {context_tokens} tokens > {self.reserved_context_tokens}")
        
        # Split content to fit within available tokens
        content_chunks = self._split_text(node.content, self.available_content_tokens, self.overlap_tokens)
        
        # Create chunks with context preamble
        for i, content in enumerate(content_chunks):
            chunk_id = f"{node.id}_chunk_{i}"
            
            # Combine context and content
            full_content = f"{context_preamble}\n{content}"
            
            # Check final token count
            total_tokens = count_tokens(full_content)
            if total_tokens > self.max_tokens:
                logging.warning(f"Chunk {chunk_id} exceeds token limit: {total_tokens} > {self.max_tokens}")
                # Trim content to fit (simple approach)
                words = content.split()
                reduced_words = words[:-(total_tokens - self.max_tokens)]
                content = " ".join(reduced_words)
                full_content = f"{context_preamble}\n{content}"
            
            # Create chunk object
            chunk = {
                'id': chunk_id,
                'content': full_content,
                'metadata': {
                    'document_node_id': node.id,
                    'level': node.level,
                    'title': node.title,
                    'section_id': node.section_id,
                    'chunk_index': i,
                    'total_chunks': len(content_chunks),
                    'parent_id': node.parent.id if node.parent else None,
                    'has_children': len(node.children) > 0,
                    'pages': node.pages,
                    'path': " > ".join(node.get_path()) if node.parent else node.title
                }
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text, max_tokens, overlap_tokens):
        """
        Split text into chunks of specified token size with overlap.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        if not text or count_tokens(text) <= max_tokens:
            return [text]
        
        chunks = []
        words = text.split()
        start_idx = 0
        
        while start_idx < len(words):
            # Take a chunk of max_tokens or remaining words
            end_idx = min(start_idx + max_tokens, len(words))
            chunk = " ".join(words[start_idx:end_idx])
            chunks.append(chunk)
            
            # Move start position for next chunk with overlap
            start_idx += max_tokens - overlap_tokens
            
            # If we're near the end, break to avoid tiny chunks
            if len(words) - start_idx <= overlap_tokens:
                break
        
        return chunks
```

### 4. Integration with Embedding Pipeline

Create a function to integrate the hierarchical chunking with the existing embedding pipeline:

```python
# src/processing/hierarchical_pipeline.py

import os
import json
import logging
from typing import Dict, List, Tuple
from google.cloud import storage

from src.data_processing.hierarchical_text_formatter import HierarchicalTextFormatter
from src.data_processing.document_tree import DocumentTree
from src.data_processing.hierarchical_chunker import HierarchicalChunker
from src.data_processing.loaders import PDFLoader
from src.embeddings.embedder import Embedder

def process_with_hierarchical_chunking(config: Dict) -> Tuple[List[Dict], str]:
    """
    Process documents using hierarchical chunking and generate embeddings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (chunks list, embeddings file path)
    """
    # 1. Load PDFs
    loader = PDFLoader(pdf_directory=config['paths']['documents'])
    documents = loader.load_multiple_pdfs()
    logging.info(f"Loaded {len(documents)} documents")
    
    # 2. Process with hierarchical text formatter
    formatter = HierarchicalTextFormatter()
    
    # First handle document cleaning and structure detection
    structured_documents = {}
    for doc_name, pages in documents.items():
        # Extract structure for this document
        structured_pages = formatter.detect_document_structure(pages)
        
        # Get headers and footers
        first_lines = [page.get('first_line', '') for page in pages]
        last_lines = [page.get('last_line', '') for page in pages]
        headers, footers = formatter.detect_headers_footers(first_lines, last_lines)
        
        # Clean each page
        cleaned_pages = []
        for i, page in enumerate(structured_pages):
            cleaned_text, exclude_page = formatter.text_formatter(page['text'], headers, footers)
            if exclude_page or not cleaned_text:
                continue
                
            # Add cleaned text back to page
            cleaned_page = page.copy()
            cleaned_page['text'] = cleaned_text
            cleaned_pages.append(cleaned_page)
        
        structured_documents[doc_name] = cleaned_pages
    
    # Convert to Document objects with metadata
    formatted_documents = formatter.format_documents_with_structure(structured_documents)
    logging.info(f"Created {len(formatted_documents)} document objects with structure")
    
    # 3. Build document trees
    document_trees = {}
    for doc in formatted_documents:
        doc_name = doc.metadata['document_name']
        if doc_name not in document_trees:
            document_trees[doc_name] = DocumentTree(doc_name)
        
    # Add documents to trees
    for doc_name, tree in document_trees.items():
        docs_for_tree = [d for d in formatted_documents if d.metadata['document_name'] == doc_name]
        tree.build_from_documents(docs_for_tree)
        
        # Generate summaries for all nodes
        tree.generate_summaries(max_length=100)
    
    logging.info(f"Built {len(document_trees)} document trees")
    
    # 4. Create hierarchical chunks
    chunker = HierarchicalChunker(max_tokens=768, overlap_tokens=50)
    all_chunks = []
    
    for doc_name, tree in document_trees.items():
        chunks = chunker.create_chunks_with_parent_context(tree)
        all_chunks.extend(chunks)
    
    logging.info(f"Created {len(all_chunks)} hierarchical chunks")
    
    # 5. Save chunks to file
    chunks_path = os.path.join(config['paths']['embeddings_save_path'], 'hierarchical_chunks.json')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    # Upload to GCS
    storage_client = storage.Client(project=config['gcp']['project_id'])
    bucket = storage_client.bucket(config['gcp']['bucket_name'])
    blob = bucket.blob(f"{config['gcp']['embeddings_path']}hierarchical_chunks.json")
    blob.upload_from_filename(chunks_path)
    
    logging.info(f"Saved chunks to {chunks_path} and uploaded to GCS")
    
    # 6. Generate embeddings
    embedder = Embedder(
        project_id=config['gcp']['project_id'],
        location=config['gcp']['location'],
        bucket_name=config['gcp']['bucket_name'],
        embeddings_path=config['gcp']['embeddings_path'],
        credentials=None  # This will use default credentials
    )
    
    embeddings_path = os.path.join(config['paths']['embeddings_save_path'], 'hierarchical_embeddings.jsonl')
    embedder.generate_and_store_embeddings(
        chunks=all_chunks,
        local_jsonl_path=embeddings_path,
        batch_size=9  # Adjust as needed
    )
    
    logging.info(f"Generated and stored embeddings at {embeddings_path}")
    
    return all_chunks, embeddings_path
```

### 5. Create and Deploy New Vector Index

Create a function to set up a new index for the hierarchical chunks:

```python
# src/vector_search/hierarchical_indexing.py

import logging
from typing import Dict, Tuple
from google.cloud import aiplatform

from src.vector_search.indexer import VectorIndexer

def create_hierarchical_index(config: Dict) -> Tuple[str, str]:
    """
    Create and deploy a new vector index for hierarchical chunks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (endpoint name, deployed index ID)
    """
    # Create a copy of the config for the hierarchical index
    hierarchical_config = config.copy()
    
    # Modify index and endpoint names to distinguish from original
    hierarchical_config['vector_search']['index_display_name'] += "_hierarchical"
    hierarchical_config['vector_search']['endpoint_display_name'] += "_hierarchical"
    hierarchical_config['vector_search']['deployed_index_id'] += "_hierarchical"
    
    # Initialize indexer with new config
    indexer = VectorIndexer(hierarchical_config)
    
    # Create the index
    try:
        index = indexer.create_index()
        logging.info(f"Created hierarchical index: {index.display_name}")
        
        # Deploy the index
        endpoint, deployed_index_id = indexer.deploy_index(index)
        logging.info(f"Deployed hierarchical index with ID: {deployed_index_id} to endpoint: {endpoint.display_name}")
        
        return endpoint.display_name, deployed_index_id
    except Exception as e:
        logging.error(f"Failed to create or deploy hierarchical index: {e}")
        raise
```

### 6. Update Main Pipeline

Modify the main pipeline to include hierarchical chunking:

```python
# src/main.py (modifications)

# Add imports
from src.processing.hierarchical_pipeline import process_with_hierarchical_chunking
from src.vector_search.hierarchical_indexing import create_hierarchical_index

# Add hierarchical chunking option
def parse_arguments():
    # Add to existing arguments
    parser.add_argument(
        '--use-hierarchical',
        action='store_true',
        help='Use hierarchical chunking with parent summaries.'
    )
    
    # Existing code...
    return parser.parse_args()

# In main function
def main():
    args = parse_arguments()
    
    # Load config and set up logging...
    
    # If hierarchical chunking is requested
    if args.use_hierarchical:
        logging.info("Using hierarchical chunking with parent summaries")
        
        # Process documents with hierarchical chunking
        hierarchical_chunks, embeddings_path = process_with_hierarchical_chunking(config)
        
        # Create and deploy hierarchical index
        endpoint_name, deployed_index_id = create_hierarchical_index(config)
        
        # Store the hierarchical endpoint and index for later use
        config['hierarchical'] = {
            'endpoint_display_name': endpoint_name,
            'deployed_index_id': deployed_index_id,
        }
        
        # Update config file to save hierarchical settings
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    # Continue with rest of pipeline...
```

### 7. Update VectorSearcher for Hierarchical Expansion (Optional)

Extend VectorSearcher to support hierarchical expansion for better context:

```python
# src/vector_search/hierarchical_searcher.py

from typing import List, Dict
import logging

from src.vector_search.searcher import VectorSearcher

class HierarchicalVectorSearcher(VectorSearcher):
    """
    Extends VectorSearcher with hierarchical expansion capabilities.
    """
    
    def hierarchical_search(
        self,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        query_text: str,
        num_neighbors: int = 5,
        expand_parents: bool = True
    ) -> List[Dict]:
        """
        Perform vector search with hierarchical expansion.
        
        Args:
            index_endpoint_display_name: Name of index endpoint
            deployed_index_id: ID of deployed index
            query_text: Query text
            num_neighbors: Number of neighbors to retrieve
            expand_parents: Whether to include parent chunks
            
        Returns:
            List of retrieved chunks with parent expansion
        """
        # Perform base vector search
        results = self.vector_search(
            index_endpoint_display_name,
            deployed_index_id,
            query_text,
            num_neighbors
        )
        
        if not expand_parents:
            return results
        
        # Collect unique parent IDs
        parent_ids = set()
        for result in results:
            parent_id = result.get('metadata', {}).get('parent_id')
            if parent_id:
                parent_ids.add(parent_id)
        
        # Add parent chunks if not already included
        expanded_results = list(results)  # Make a copy
        result_ids = {r['id'] for r in results}
        
        for parent_id in parent_ids:
            # Look for any chunk from this parent
            for chunk_id, chunk_data in self.chunks.items():
                if chunk_id.startswith(f"{parent_id}_chunk_") and chunk_id not in result_ids:
                    # Found a parent chunk, add it to results
                    expanded_results.append(chunk_data)
                    result_ids.add(chunk_id)
                    # Only need one chunk per parent
                    break
        
        logging.info(f"Expanded results from {len(results)} to {len(expanded_results)} chunks")
        return expanded_results
```

## Implementation Steps

1. **Setup**: Create the new Python files in their respective directories
2. **Implement Hierarchy Detection**: Start with the `HierarchicalTextFormatter`
3. **Build Tree Structure**: Implement the `DocumentTree` class
4. **Create Chunker**: Implement the `HierarchicalChunker` with parent summaries
5. **Integration**: Set up the integration pipeline and create new index
6. **Test**: Validate token counts and system behavior with real documents

## Key Considerations

1. **Token Counting**: Keep track of token counts to ensure no chunk exceeds 768 tokens
2. **Context Size**: Limit parent summary to ~100 tokens to leave room for content
3. **Testing**: Verify that the parent context is helpful and not introducing noise
4. **Performance**: Check memory usage when building document trees for large documents
5. **Error Handling**: Add robust error handling, especially for token limit enforcement

## Conclusion

This simplified approach focuses on adding valuable parent summaries to child chunks while maintaining minimal overhead. By starting with this approach, you can achieve most of the benefits of hierarchical chunking without excessive complexity.

The implementation builds upon your existing codebase and can be activated as an optional feature by adding the `--use-hierarchical` flag to your existing pipeline.
