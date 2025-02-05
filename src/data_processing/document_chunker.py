import logging
import json
import uuid
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from google.cloud import storage
import os
from google.oauth2.credentials import Credentials

# Mapping dictionaries for ORAN metadata extraction from document file names.
WG_MAPPING = {
    "WG1": "WG1: Use Cases and Overall Architecture Workgroup",
    "WG2": "WG2: Non-real-time RAN Intelligent Controller and A1 Interface Workgroup",
    "WG3": "WG3: Near-real-time RIC and E2 Interface Workgroup",
    "WG4": "WG4: Open Fronthaul Interfaces Workgroup",
    "WG5": "WG5: Open F1/W1/E1/X2/Xn Interface Workgroup",
    "WG6": "WG6: Cloudification and Orchestration Workgroup",
    "WG7": "WG7: White-box Hardware Workgroup",
    "WG8": "WG8: Stack Reference Design Workgroup",
    "WG9": "WG9: Open X-haul Transport Workgroup",
    "WG10": "WG10: OAM for O-RAN",
    "WG11": "WG11: Security Work Group",
    "TIFG": "TIFG: Test & Integration Focus Group",
    "SUFG": "SuFG: Sustainability Focus Group"
}

SUBCATEGORY_MAPPING = {
    "MP": "M-plane",
    "CUS": "Control, User, and Synchronization Planes",
    "CONF": "Conformance Test",
    "IOT": "Interoperability Test",
    "CTI-TCP": "Cooperative Transport Interface Transport Control Plane",
    "CTI-TMP": "Cooperative Transport Interface Transport Management Procedure",
    "CCIN": "Communication and Computing Integrated Networks",
    "U": "U-plane",
    "C": "C-plane",
    "OAD": "O-RAN Architecture Description",
    "OAM": "Operations and Maintenance",
    "UCR": "Use Cases and Requirements",
    "GAP": "General Aspects and Principles",
    "AP": "Application Protocol",
    "SM": "Service Model",
    "ARCH": "Architecture",
    "A1GAP": "A1 interface: General Aspects and Principles",
    "A1UCR": "A1 interface: Use Cases and Requirements",
    "A1AP": "A1 interface: Application Protocol",
    "A1TD": "A1 interface: Type Definitions",
    "A1TS": "A1 interface: Test Specification",
    "A1TP": "A1 interface: Transport Protocol",
    "TS": "Technical Specification",
    "TR": "Technical Report",
    "R1GAP": "R1 interface: General Aspects and Principles",
    "R1UCR": "R1 interface: Use Cases and Requirements",
    "R1AP": "R1 interface: Application Protocol",
    "R1TD": "R1 interface: Type Definitions",
    "R1TP": "R1 interface: Transport Protocol",
    "E2GAP": "E2 interface: General Aspects and Principles",
    "E2UCR": "E2 interface: Use Cases and Requirements",
    "E2AP": "E2 interface: Application Protocol",
    "E2TD": "E2 interface: Type Definitions",
    "E2TS": "E2 interface: Test Specification",
    "E2TP": "E2 interface: Transport Protocol",
    "Y1GAP": "Y1 interface: General Aspects and Principles",
    "Y1UCR": "Y1 interface: Use Cases and Requirements",
    "Y1AP": "Y1 interface: Application Protocol",
    "Y1TD": "Y1 interface: Type Definitions",
    "Y1TS": "Y1 interface: Test Specification",
    "Y1TP": "Y1 interface: Transport Protocol",
    "CADS": "Cloud Architecture and Deployment Scenarios",
    "O2-GA&P": "O2 interface: General Aspects and Principles",
    "ORCH": "Orchestration",
    "AAL": "Acceleration Abstraction Layer",
    "ASD": "Application Service Descriptor",
    "AppLCM": "Application Lifecycle Management",
    "IPC": "Indoor Picocell",
    "NES": "Network Energy Savings",
    "OMAC": "Outdoor Macrocell",
    "OMC": "Outdoor Microcell",
    "EMC": "Enterprise Microcell",
    "DSC": "Deployment Scenarios and Base Station Classes",
    "FHGW": "Fronthaul Gateway",
    "AAD": "Architecture and API",
    "XTRP-MGT": "X-haul Transport Management",
    "XPSAAS": "X-haul Packet Switched Architecture and Solutions",
    "XTRP-SYN": "X-haul Transport Synchronization",
    "XTRP-TST": "X-haul Transport Test",
    "TE&IV-CIMI": "Topology Exposure & Inventory Common Information Models and Interface",
    "TE&IV-UCR": "Topology Exposure and Inventory Management Services Use Cases and Requirement",
    "O1PMeas": "O1 Performance Measurements",
    "O1NRM": "O1 Network Resource Model",
    "E2E-Test": "End-to-End Test",
    "CGofOTIC": "Criteria and Guidelines of Open Testing and Integration Centre",
    "OTIC": "Open Testing and Integration Centre",
    "E2ETSTFWK": "End-to-End System Testing Framework",
    "AIML": "Artificial Intelligence and Machine Learning",
    "ZTA": "Zero Trust Architecture",
}

def extract_oran_metadata_from_filename(document_name: str) -> dict:
    """
    Extracts ORAN-specific metadata from the document file name.
    This function normalizes the delimiters (converting '-' to '.') and
    then splits the name to extract the workgroup, subcategory, and version.
    
    Examples:
      - "O-RAN.WG4.MP.0-R004-v16.01" 
            -> workgroup: WG4: Open Fronthaul Interfaces Workgroup,
               subcategory: M-plane,
               version: 16.01
      - "O-RAN.WG2.A1UCR-v04.00"
            -> workgroup: WG2: Non-real-time RAN Intelligent Controller and A1 Interface Workgroup,
               subcategory: A1 interface: Use Cases and Requirements,
               version: 4.0
    """
    metadata = {
        "version": "unknown",
        "workgroup": "unknown-workgroup",
        "subcategory": "unknown-subcategory"
    }
    # Normalize the file name: replace hyphens with dots to standardize delimiters.
    normalized_name = document_name.replace("-", ".")
    # Remove any leading "O.RAN." or "O-RAN." (case-insensitive).
    normalized_name = re.sub(r'(?i)^O[.-]?RAN[.-]?', '', normalized_name)
    parts = normalized_name.split(".")
    if len(parts) >= 2:
        wg = parts[0].strip().upper()
        metadata["workgroup"] = WG_MAPPING.get(wg, wg)
        subcat = parts[1].strip().upper()
        metadata["subcategory"] = SUBCATEGORY_MAPPING.get(subcat, subcat)
    # Extract version using a regex pattern (e.g., v16.01)
    version_match = re.search(r'v(\d+(?:\.\d+)*)', document_name, re.IGNORECASE)
    if version_match:
        ver = version_match.group(1)
        try:
            normalized_ver = str(float(ver))
        except Exception:
            normalized_ver = ver
        metadata["version"] = normalized_ver
    return metadata

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1536,
        chunk_overlap: int = 256,
        separators: List[str] = None,
        gcs_bucket_name: str = None,
        gcs_embeddings_path: str = "embeddings/",
        credentials: Credentials = None,
        min_char_count: int = 100  # Minimum characters per chunk
    ):
        if separators is None:
            separators = [". ", "? ", "! ", "\n\n"]
        
        # Use word count as a proxy for token count.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=lambda text: len(text.split())
        )
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_embeddings_path = gcs_embeddings_path
        self.credentials = credentials
        self.min_char_count = min_char_count

        logging.info(
            f"DocumentChunker initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}, separators={separators}, "
            f"gcs_bucket_name={gcs_bucket_name}, gcs_embeddings_path={gcs_embeddings_path}, "
            f"min_char_count={min_char_count}"
        )
    
    def split_documents(self, documents: List[Document]) -> List[Dict]:
        """
        Splits a list of Document objects into chunks and assigns unique IDs.
        Returns:
            List[Dict]: List of dictionaries containing chunk information.
        """
        logging.info("Starting to split documents into chunks.")
        split_docs = self.splitter.split_documents(documents)
        logging.info(f"Total chunks created before filtering: {len(split_docs)}")
        
        chunks_with_ids = self.assign_ids(split_docs)
        logging.info(f"Total chunks after filtering: {len(chunks_with_ids)}")
        return chunks_with_ids
    
    def assign_ids(self, split_docs: List[Document]) -> List[Dict]:
        """
        Assigns a unique UUID to each chunk, embeds a metadata header into the chunk content,
        and stores the metadata separately.
        
        The metadata header for ORAN documents now includes:
          - Document Name
          - Revision Data (extracted from the file name)
          - Workgroup (with full name)
          - Subcategory (interpreted from abbreviation)
          - Page Number
        
        Returns:
            List[Dict]: List of chunk dictionaries with full metadata.
        """
        logging.info("Assigning UUIDs and embedding ORAN metadata into chunks.")
        chunks_with_ids = []
        for doc in split_docs:
            char_count = len(doc.page_content)
            if char_count < self.min_char_count:
                logging.debug(f"Skipped chunk due to low character count: {char_count} < {self.min_char_count}.")
                continue
            
            chunk_uuid = str(uuid.uuid4())
            document_name = doc.metadata.get('document_name', 'Unknown Document')
            page_number = doc.metadata.get('page_number', 'Unknown Page')
            
            # Extract additional ORAN metadata from the document file name.
            extracted_metadata = extract_oran_metadata_from_filename(document_name)
            version = doc.metadata.get('version', extracted_metadata.get('version'))
            workgroup = doc.metadata.get('workgroup', extracted_metadata.get('workgroup'))
            subcategory = doc.metadata.get('subcategory', extracted_metadata.get('subcategory'))
            
            metadata_header = (
                f"Document Name: {document_name}; "
                f"Version: {version}; "
                f"Workgroup: {workgroup}; "
                f"Subcategory: {subcategory}; "
                f"Page: {page_number};"
            )
            
            full_content = metadata_header + "\n\n" + doc.page_content
            
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
                    "page_number": page_number
                }
            }
            chunks_with_ids.append(chunk_dict)
        
        logging.info("Completed assigning UUIDs and embedding ORAN metadata into chunks.")
        return chunks_with_ids
    
    def save_chunks_to_json(self, chunks: List[Dict], file_path: str = "chunks.json"):
        """
        Saves the chunk mappings to a JSON Lines (.jsonl) file.
        """
        logging.info(f"Saving {len(chunks)} chunks to {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    line_dict = {
                        'id': chunk.get('id'),
                        'content': chunk.get('content', ""),
                        'document_name': chunk.get('document_name'),
                        'page_number': chunk.get('page_number'),
                        'char_count': chunk.get('char_count'),
                        'metadata': chunk.get('metadata', {})
                    }
                    json_record = json.dumps(line_dict, ensure_ascii=False)
                    f.write(json_record + "\n")
            logging.info(f"Chunk data with IDs and metadata saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save chunks to {file_path}: {e}", exc_info=True)
            raise
    
    def upload_to_gcs(self, file_path: str, overwrite: bool = True):
        """
        Uploads a file to the specified Google Cloud Storage bucket.
        """
        if not self.gcs_bucket_name:
            logging.error("GCS bucket name not provided. Cannot upload file.")
            raise ValueError("GCS bucket name not provided.")
        
        logging.info(f"Uploading {file_path} to GCS bucket {self.gcs_bucket_name} at {self.gcs_embeddings_path}")
        try:
            if self.credentials:
                storage_client = storage.Client(credentials=self.credentials, project=self.credentials.project_id)
            else:
                storage_client = storage.Client()
            
            bucket = storage_client.bucket(self.gcs_bucket_name)
            destination_blob_name = os.path.join(self.gcs_embeddings_path, os.path.basename(file_path))
            blob = bucket.blob(destination_blob_name)
            
            if blob.exists() and not overwrite:
                logging.warning(f"Blob {destination_blob_name} already exists and overwrite is set to False.")
                return
            
            blob.upload_from_filename(file_path, content_type="application/json")
            logging.info(f"Uploaded {file_path} to GCS: {destination_blob_name}")
        except Exception as e:
            logging.error(f"Failed to upload {file_path} to GCS: {e}", exc_info=True)
            raise