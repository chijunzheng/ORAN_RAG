# src/data_processing/converters.py

import os
import subprocess
import logging
import shutil
from typing import Optional

class DocumentConverter:
    def __init__(self, directory_path: str, soffice_path: Optional[str] = None):
        """
        Initializes the DocumentConverter with the target directory path and the path to 'soffice'.
        
        Args:
            directory_path (str): Path to the directory containing .docx files.
            soffice_path (str, optional): Path to the 'soffice' executable.
                                         If None, it will be searched in the system's PATH.
        """
        self.directory_path = directory_path
        self.soffice_path = soffice_path or shutil.which('soffice')
        
        if not self.soffice_path:
            logging.error(
                "'soffice' command not found. Please install LibreOffice and ensure 'soffice' is in your PATH."
            )
            raise FileNotFoundError(
                "'soffice' command not found. Please install LibreOffice and ensure 'soffice' is in your PATH."
            )
        else:
            logging.debug(f"'soffice' found at: {self.soffice_path}")

        # Verify that the directory exists
        if not os.path.isdir(self.directory_path):
            logging.error(f"The directory '{self.directory_path}' does not exist.")
            raise NotADirectoryError(f"The directory '{self.directory_path}' does not exist.")
        else:
            logging.debug(f"DocumentConverter initialized for directory: {self.directory_path}")

    def convert_docx_to_pdf(self):
        """
        Converts all .docx files in the specified directory to .pdf using LibreOffice.
        """
        docx_files = [f for f in os.listdir(self.directory_path) if f.lower().endswith('.docx')]
        
        if not docx_files:
            logging.warning(f"No .docx files found in directory: {self.directory_path}")
            return

        for filename in docx_files:
            docx_path = os.path.join(self.directory_path, filename)
            try:
                logging.info(f"Starting conversion of '{filename}'...")
                result = subprocess.run([
                    self.soffice_path,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", self.directory_path,
                    docx_path
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120  # Timeout after 2 minutes
                )
                output_pdf = os.path.splitext(filename)[0] + ".pdf"
                logging.info(f"Successfully converted '{filename}' to '{output_pdf}' using LibreOffice.")
            except subprocess.TimeoutExpired:
                logging.error(f"Conversion of '{filename}' timed out after 120 seconds.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to convert '{filename}' to PDF. Error: {e.stderr.decode().strip()}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while converting '{filename}': {e}")