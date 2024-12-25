# src/data_processing/text_formatter.py

import logging
import re
from typing import List, Tuple, Dict
from langchain.schema import Document
from collections import Counter, defaultdict

class TextFormatter:
    def __init__(self):
        """
        Initializes the TextFormatter with comprehensive header and footer patterns.
        """
        # Define known footer patterns
        self.footer_patterns = [
            r'^Page\s+\d+\s+of\s+\d+$',
            r'^\d+\s+/\s+\d+$',
            r'^Page\s+\d+$',
            r'^O-RAN\.\w+\.\w+\.\d+-v\d+\.\d+$',
            r'^©\s+\d{4}\s+by\s+the\s+O-RAN\s+ALLIANCE\s+e\.V\..*$',
        ]

        # Define patterns to exclude TOC pages
        self.toc_pattern = re.compile(r'\.{10,}')

    def detect_headers_footers(self, first_lines: List[str], last_lines: List[str], min_occurrence: float = 0.5) -> Tuple[List[str], List[str]]:
        """
        Detects headers and footers based on the frequency of first and last lines.

        Args:
            first_lines (List[str]): List of first lines from each page.
            last_lines (List[str]): List of last lines from each page.
            min_occurrence (float, optional): Minimum fraction of pages a line must appear in to be considered a header/footer. Defaults to 0.5.

        Returns:
            Tuple[List[str], List[str]]: Detected headers and footers.
        """
        num_pages = len(first_lines)
        logging.info(f"Total pages for header/footer detection: {num_pages}")

        # Count occurrences of first and last lines
        header_counter = Counter(first_lines)
        footer_counter = Counter(last_lines)

        # Detect header candidates
        header_candidates = [
            line for line, count in header_counter.items() if count / num_pages >= min_occurrence
        ]

        # Detect footer candidates based on patterns and frequency
        footer_candidates = []
        for line, count in footer_counter.items():
            if any(re.match(pattern, line.strip()) for pattern in self.footer_patterns):
                if count / num_pages >= min_occurrence:
                    footer_candidates.append(line)

        logging.info(f"Detected {len(header_candidates)} header candidates.")
        logging.info(f"Detected {len(footer_candidates)} footer candidates.")

        return header_candidates, footer_candidates

    def text_formatter(self, text: str, headers: List[str] = None, footers: List[str] = None) -> Tuple[str, bool]:
        """
        Cleans text by removing headers, footers, and excludes TOC pages.

        Args:
            text (str): The text to be cleaned.
            headers (List[str], optional): Identified headers to remove. Defaults to None.
            footers (List[str], optional): Identified footers to remove. Defaults to None.

        Returns:
            Tuple[str, bool]: Cleaned text and a boolean indicating whether to exclude the page.
        """
        if not text:
            logging.debug("Empty text received for formatting.")
            return "", False

        lines = text.split('\n')

        # 1. Exclude TOC pages based on dotted lines
        has_toc = any(self.toc_pattern.search(line) for line in lines)
        if has_toc:
            logging.info("Excluded page due to TOC detection.")
            return "", True

        # 2. Remove identified headers
        if headers and lines:
            if lines[0].strip() in headers:
                logging.debug("Header detected and removed.")
                lines = lines[1:]

        # 3. Remove identified footers
        if footers and lines:
            if lines[-1].strip() in footers:
                logging.debug("Footer detected and removed.")
                lines = lines[:-1]

        # 4. Remove lines matching known footer patterns
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            if any(re.match(pattern, line_stripped) for pattern in self.footer_patterns):
                logging.debug(f"Removed footer line: {line_stripped}")
                continue
            cleaned_lines.append(line)

        # 5. Remove dynamic header/footer patterns
        dynamic_patterns = [
            r'^[\s_]+$',  # Lines with only spaces or underscores
            r'^[\s_]+\d+\s+O-RAN\.[A-Za-z0-9\.\-]+(?:-v\d+\.\d+)?$',
            r'^\d+\s+O-RAN\.[A-Za-z0-9\.\-]+(?:-v\d+\.\d+)?$',
            r'^\d+$',
            r'.*O-RAN\.\w+\.\w+\.\d+-v\d+\.\d+$',
            r'^O-RAN\.[A-Za-z0-9.\-]+$',
        ]

        final_lines = []
        for line in cleaned_lines:
            line_stripped = line.strip()
            if any(re.match(pattern, line_stripped) for pattern in dynamic_patterns):
                logging.debug(f"Removed dynamic header/footer line: {line_stripped}")
                continue
            final_lines.append(line)

        # 6. Rejoin lines and clean up whitespace
        cleaned_text = ' '.join(final_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        logging.debug("Text formatted successfully.")

        return cleaned_text, False

    def format_documents(
        self,
        documents: Dict[str, List[Dict]]
    ) -> List[Document]:
        """
        Processes and cleans all documents by removing headers/footers and excluding TOC pages.

        Args:
            documents (Dict[str, List[Dict]]): A dictionary where keys are document names and values are lists of pages.
                                               Each page is a dict with 'text' and 'metadata'.

        Returns:
            List[Document]: A list of cleaned Document objects.
        """
        all_cleaned_documents = []

        for doc_name, pages in documents.items():
            logging.info(f"Processing document: {doc_name}")

            # Extract first and last lines for header/footer detection
            first_lines = [page['first_line'] for page in pages]
            last_lines = [page['last_line'] for page in pages]

            # Detect headers and footers
            headers, footers = self.detect_headers_footers(first_lines, last_lines, min_occurrence=0.5)

            logging.info(f"Detected headers for {doc_name}: {headers}")
            logging.info(f"Detected footers for {doc_name}: {footers}")

            # Clean each page
            for i, page in enumerate(pages, start=1):
                cleaned_text, exclude_page = self.text_formatter(page['text'], headers, footers)

                if exclude_page:
                    logging.info(f"Excluded TOC page {i} for {doc_name}")
                    continue  # Skip this page

                if cleaned_text:
                    metadata = page.get('metadata', {})
                    cleaned_document = Document(
                        page_content=cleaned_text,
                        metadata={
                            'document_name': doc_name,
                            'page_number': metadata.get('page_number')
                        }
                    )
                    all_cleaned_documents.append(cleaned_document)
                    logging.debug(f"Added cleaned page {i} for {doc_name}")
                else:
                    logging.debug(f"No content after cleaning for page {i} in {doc_name}")

            logging.info(f"Total cleaned pages for {doc_name}: {len([doc for doc in all_cleaned_documents if doc.metadata['document_name'] == doc_name])}")

        logging.info(f"Total cleaned documents: {len(all_cleaned_documents)}")
        return all_cleaned_documents