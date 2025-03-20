#!/usr/bin/env python
# Simple test script to verify the ContextualChunker implementation

import os
import sys
import logging
from src.data_processing.contextual_chunker import ContextualChunker

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Successfully loaded environment variables from .env file")
except ImportError:
    logging.warning("python-dotenv not installed. Will use environment variables directly.")

def test_contextual_chunker():
    """Test the ContextualChunker with a sample document."""
    try:
        # Create a contextual chunker instance
        chunker = ContextualChunker(
            chunk_size=500,
            chunk_overlap=50,
            min_char_count=50
        )
        print("✅ Successfully instantiated ContextualChunker")
        
        # Sample document text
        document_text = """
        O-RAN ALLIANCE
        O-RAN Working Group 1
        O-RAN Architecture Description v05.00
        
        1 Introduction
        The O-RAN Alliance was founded in February 2018 by AT&T, China Mobile, Deutsche Telekom, 
        NTT DOCOMO and Orange. The Alliance is a world-wide, carrier-led effort to drive new levels of 
        openness in the radio access network of next generation wireless systems.
        
        2 O-RAN Architecture
        The O-RAN architecture is a decoupled RAN architecture that is designed to deliver an open, 
        interoperable, virtualized, and fully programmable RAN. The O-RAN architecture includes 
        decomposed RAN components with open interfaces between them.
        
        2.1 O-RAN Logical Architecture
        The O-RAN logical architecture defines several key functional splits and interfaces:
        - Split 7-2x between O-DU and O-RU
        - E2 interface between O-RAN components and E2 Node
        - O1 interface for management plane
        - Open Fronthaul interface
        
        3 Key Components
        The key components of O-RAN architecture include:
        - O-CU (O-RAN Central Unit)
        - O-DU (O-RAN Distributed Unit)
        - O-RU (O-RAN Radio Unit)
        - RIC (RAN Intelligent Controller)
        - SMO (Service Management and Orchestration)
        """
        
        # Sample chunk to test context generation
        chunk_text = "The O-RAN logical architecture defines several key functional splits and interfaces:\n- Split 7-2x between O-DU and O-RU\n- E2 interface between O-RAN components and E2 Node\n- O1 interface for management plane\n- Open Fronthaul interface"
        
        # Test context generation with Gemini API
        print("\nTesting Gemini API context generation...")
        context = chunker.generate_chunk_context(document_text, chunk_text)
        print(f"Generated context: {context}")
        
        # Test fallback context generation
        print("\nTesting fallback context generation...")
        fallback_context = chunker._generate_simple_context(document_text, chunk_text)
        print(f"Fallback context: {fallback_context}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_contextual_chunker() 