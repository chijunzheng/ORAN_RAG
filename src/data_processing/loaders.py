# src/data_processing/loaders.py

import os
import fitz
import logging
from collections import defaultdict, Counter
from typing import Dict, List
from langchain.schema import Document

class PDFLoader:
    def __init__(self, pdf_directory: str):
        """
        Initializes the PDFLoader with the directory containing PDF files.
        
        Args:
            pdf_directory (str): Path to the directory containing .pdf files.
        """
        self.pdf_directory = pdf_directory

    def load_multiple_pdfs(self) -> Dict[str, List[Dict]]:
        """
        Loads multiple PDFs and extracts text per page.
        
        Returns:
            Dict[str, List[Dict]]: A dictionary mapping document names to their pages' data.
        """
        documents = defaultdict(list)
        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)
                try:
                    doc = fitz.open(pdf_path)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text = page.get_text("text")
                        metadata = {
                            'document_name': filename,
                            'page_number': page_num + 1
                        }
                        lines = text.split('\n') if text else []
                        first_line = lines[0].strip() if lines else ''
                        last_line = lines[-1].strip() if lines else ''

                        documents[filename].append({
                            'text': text,
                            'metadata': metadata,
                            'first_line': first_line,
                            'last_line': last_line
                        })
                    logging.info(f"Loaded {filename} with {len(documents[filename])} pages.")
                except Exception as e:
                    logging.error(f"Failed to load {filename}: {e}")

        total_pages = sum(len(pages) for pages in documents.values())
        logging.info(f"Total Documents Loaded: {len(documents)} with {total_pages} pages.")
        return documents