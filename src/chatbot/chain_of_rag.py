import logging
from typing import List, Dict, Tuple, Any
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part
)

class ChainOfRagProcessor:
    """
    Implements Chain of RAG for the ORAN RAG system.
    
    Chain of RAG breaks down complex queries into follow-up questions and 
    gradually builds up an answer through iterative search and information gathering.
    
    It is particularly effective for:
    1. Complex factual queries
    2. Multi-hop reasoning questions
    3. Questions requiring information from multiple documents
    """
    def __init__(
        self,
        vector_searcher,
        llm,
        generation_config: GenerationConfig,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        max_iterations: int = 4,
        early_stopping: bool = True
        ):
        self.vector_searcher = vector_searcher
        self.llm = llm  # generative model (e.g., gemini-1.5-flash-002)
        self.generation_config = generation_config
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        self.index_endpoint_display_name = index_endpoint_display_name
        self.deployed_index_id = deployed_index_id

        # Use specific generation config for Chain of RAG
        self.chain_generation_config = GenerationConfig(
            temperature=0,
            top_p=1,
            max_output_tokens=8192,
        )

    def _generate_follow_up_query(self, main_query: str, intermediate_context: List[str]) -> str:
        """
        Generates a follow-up query based on the main query and intermediate context.
        
        Args:
            main_query: Original user query
            intermediate_context: List of previous query-answer pairs
            
        Returns:
            Follow-up query
        """
        prompt = f"""You are using a search tool to answer the main query by iteratively searching the database. Given the following intermediate queries and answers, generate a new simple follow-up question that can help answer the main query. You may rephrase or decompose the main query when previous answers are not helpful. Ask simple follow-up questions only as the search tool may not understand complex questions.

Guidelines for generating the follow-up question:
1. The initial follow-up question should be exactly the same as the main query.
2. Review previous answers carefully and assess whether it is relevant to the main query - build upon it to dig deeper
3. Keep questions simple and specific - avoid compound questions
4. Stay within the scope of the main query. Always assess the relevance of the follow-up question to the main query.
5. Keep digging deeper into the main query, intermediate queries and answers until the answer satisfies the main query.
6. Do not ask follow-up questions that are not relevant to the main query.
7. If intermediate answer produces "No relevant information found", you can ask a new follow-up question based on the main query and the previous answers.

## Previous intermediate queries and answers
{intermediate_context}

## Main query to answer
{main_query}

Respond with a simple follow-up question that will help answer the main query, do not explain yourself or output anything else.
"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
        follow_up_query = response.text.strip() if response and response.text.strip() else ""
        logging.info(f"Follow-up Query: {follow_up_query}")
        return follow_up_query

    def _retrieve_context(self, query: str) -> List[Dict]:
        """
        Retrieves ORAN document chunks relevant to the query.
        
        Args:
            query: The search query
            
        Returns:
            List of retrieved document chunks
        """
        if not self.vector_searcher:
            logging.error("No vector_searcher available for retrieval.")
            return []
        
        retrieved = self.vector_searcher.vector_search(
            query_text=query, 
            num_neighbors=5,
            index_endpoint_display_name=self.index_endpoint_display_name,
            deployed_index_id=self.deployed_index_id,
        )
        
        logging.info(f"Retrieved {len(retrieved)} documents for query: {query}")
        return retrieved

    def _generate_intermediate_answer(self, query: str, retrieved_documents: List[Dict]) -> str:
        """
        Generates an intermediate answer based on retrieved documents.
        
        Args:
            query: The query to answer
            retrieved_documents: List of retrieved document chunks
            
        Returns:
            Intermediate answer
        """
        # Format retrieved documents
        context_text = ""
        for i, chunk in enumerate(retrieved_documents, start=1):
            doc_meta = chunk.get("metadata", {})
            doc_name = chunk.get("document_name", "N/A")
            version = doc_meta.get("version", "unknown")
            workgroup = doc_meta.get("workgroup", "unknown")
            subcat = doc_meta.get("subcategory", "unknown")
            page = chunk.get("page_number", "N/A")

            context_text += (
                f"Document {i}:\n"
                f"Source: {doc_name}, page {page}\n\n"
                f"{chunk.get('content', 'No content')}\n\n"
            )

        prompt = f"""Given the following documents, generate an appropriate answer for the query. DO NOT hallucinate any information, only use the provided documents to generate the answer. Respond "No relevant information found" if the documents do not contain useful information.

## Documents
{context_text}

## Query
{query}

### Response Guidelines:
1. Provide a detailed step-by-step answer based on the documents.
2. For each significant statement or fact, include a reference to the source document using the following format:
   - After each statement, add the document name and page number from the "Source:" field. For example: "O-RAN.WG4.MP, page 20" or "(O-RAN.WG4.MP, page 20)"
   - DO NOT use generic references like "Document 0" or "Document 1" - always use the actual document name
3. When documents have conflicting information, cite all relevant sources and explain the differences.
4. If citing multiple documents for one statement, list them all: "(O-RAN.WG4.MP, page 20; O-RAN.WG7, page 96)"
5. Only use information from the provided documents - do not hallucinate.
6. Extract the document name and page number from the "Source:" field that appears at the end of each document.

Respond with a detailed and well-referenced answer that will be valuable for generating the final response.
"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
        intermediate_answer = response.text.strip() if response and response.text.strip() else "No relevant information found."
        logging.info(f"Intermediate Answer (first 200 chars): {intermediate_answer[:200]}...")
        return intermediate_answer

    def _check_has_enough_info(self, query: str, intermediate_contexts: List[str]) -> bool:
        """
        Checks if we have enough information to answer the main query.
        
        Args:
            query: Main query
            intermediate_contexts: List of intermediate query-answer pairs
            
        Returns:
            Whether we have enough information
        """
        if not intermediate_contexts:
            return False

        prompt = f"""Given the following intermediate queries and answers, carefully assess whether you have SUFFICIENT information to provide a COMPREHENSIVE answer to the main query.

## Intermediate queries and answers
{intermediate_contexts}

## Main query
{query}

Before responding, consider the following criteria:

1. COMPREHENSIVENESS: Do you have detailed information covering ALL key aspects of the query?
2. TECHNICAL DEPTH: Have you gathered specific technical details, protocols, specifications, and implementation details?
3. MULTIPLE SOURCES: Has the information been verified from multiple different documents or sources?
4. COMPLETENESS: Can you provide a complete answer that doesn't leave significant gaps in understanding?
5. SPECIFICITY: Do you have specific document references for all major claims you would make?
6. SUFFICIENT CONTEXT: Do you understand the broader context of the topic to provide accurate information?

IMPORTANT: Be CONSERVATIVE in your assessment. It is better to gather MORE information than to stop early with an incomplete answer.

Set an extremely high standard - only say "Yes" if you have TRULY COMPREHENSIVE information about ALL aspects of the query.

If ANYTHING is missing, respond with "No".

Respond with "Yes" or "No" ONLY - do not explain your reasoning or output anything else.
"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
        has_enough_info = response.text.strip().lower() == "yes"
        return has_enough_info

    def _generate_final_answer(self, query: str, intermediate_context: List[str], retrieved_documents: List[Dict]) -> str:
        """
        Generates the final answer based on all the intermediate context.
        
        Args:
            query: Main query
            intermediate_context: List of intermediate query-answer pairs
            retrieved_documents: All retrieved documents
            
        Returns:
            Final answer
        """
        # Format all retrieved documents for reference
        all_docs_text = ""
        doc_references = []
        for i, chunk in enumerate(retrieved_documents, start=1):
            doc_meta = chunk.get("metadata", {})
            doc_name = chunk.get("document_name", "N/A")
            version = doc_meta.get("version", "unknown")
            workgroup = doc_meta.get("workgroup", "unknown")
            subcat = doc_meta.get("subcategory", "unknown")
            page = chunk.get("page_number", "N/A")
            
            doc_references.append(f"Document {i}: {doc_name}, page {page}")
            
            all_docs_text += (
                f"Document {i}:\n"
                f"Source: {doc_name}, page {page}\n\n"
                f"{chunk.get('content', 'No content')}\n\n"
            )

        prompt = f"""Given the following intermediate queries and answers, generate a final answer for the main query by combining relevant information. Note that intermediate answers are generated by an LLM and may not always be accurate.

## Documents
{all_docs_text}

## Intermediate queries and answers
{intermediate_context}

## Main query
{query}

Please provide a comprehensive answer following these specific formatting guidelines:

1. Structure:
   - Use '##' for main sections and '###' for subsections
   - Present information in bullet points or numbered lists
   - Maintain consistent indentation for nested lists
   - Each section should be logically organized and flow naturally

2. Content Guidelines:
   - Focus on delivering a complete answer that fully addresses the query
   - Be logical and concise while providing detailed information
   - Present explanations in a step-by-step manner
   - Write in a professional and informative tone
   - Target the explanation for engineers new to O-RAN
   - CONSOLIDATE information from multiple sources into single, comprehensive bullet points

3. Reference Formatting (EXTREMELY IMPORTANT):
   - Add reference numbers in square brackets ONLY at the END of each bullet point: [1], [2], etc.
   - NEVER put references within sentences or in the middle of bullet points
   - NEVER use comma-separated references like [1,2,3] - this is incorrect
   - Place references as separate adjacent brackets at the end: [1][2][3]
   - References must be placed IMMEDIATELY after the text they support (no space between text and reference)
   - Aim to use FEWER reference numbers overall - group related information together
   - Never reference the same document multiple times within a single bullet point
   - INCORRECT: "The O-RAN architecture [1, 2, 3] includes multiple components."
   - CORRECT: "The O-RAN architecture includes multiple components.[1][2][3]"

4. Reference Mapping Section:
   - At the very end of your answer, include a "## References" section
   - List ALL referenced documents with their assigned numbers
   - Format as:
     - [1] Document Name, page Page Number
     - [2] Another Document, page Page Number
     - etc.
   - Use actual document names (e.g., "O-RAN.WG4.MP.0-R004-v16.01.pdf") rather than generic "Document X" references

5. Example Format:
   ## Main Section Title
   - Solution 1 of the O-RAN RIC API uses JSON for message encoding. This is the data interchange format used for transporting data between the Near-RT RIC and the consumer.[1][2][3]
   - The complete protocol stack for Solution 1 includes TCP for the transport layer, TLS for secure HTTP connections (optional but supported), and HTTP as the application-level protocol.[4]

   ## References
   - [1] O-RAN.WG1.OAD-R003-v12.00.pdf, page 15
   - [2] O-RAN.WG4.MP.0-R004-v16.01.pdf, page 20
   - [3] O-RAN.WG7.IPC-HRD-Opt8.0-v03.00.pdf, page 96
   - [4] O-RAN.WG10.OAM-Architecture-R004-v13.00.pdf, page 43
   - [5] O-RAN.WG4.CONF.0-R004-v11.00.pdf, page 42
   - [6] O-RAN.WG8.AAD.0-R004-v13.00.pdf, page 156

IMPORTANT: Keep your writing style fluid and natural by placing references ONLY at the end of complete statements or bullet points. DO NOT interrupt the flow of information with references throughout the text. Consolidate information from multiple sources into comprehensive, well-written bullet points with fewer reference numbers overall. The goal is to provide a clean, readable answer with minimal reference intrusion.

Your answer should be well-structured, detailed, and easy to read. Focus on answering the query accurately and comprehensively, using only the information from the provided documents.
"""
        content = Content(role="user", parts=[Part.from_text(prompt)])
        response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
        final_answer = response.text.strip() if response and response.text.strip() else ""
        return final_answer

    def process_query(self, query: str, conversation_history: List[Dict] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Processes the user query using Chain of RAG.
        
        Args:
            query: User query
            conversation_history: Conversation history (not used for Chain of RAG but kept for API compatibility)
            
        Returns:
            Tuple of (final answer, additional info)
        """
        intermediate_contexts = []
        all_retrieved_documents = []
        debug_info = {
            "iterations": [],
            "total_documents": 0,
            "early_stopped": False
        }
        
        logging.info(f"Processing query with Chain of RAG: {query}")
        
        # Initial follow-up query is the same as the main query
        for iter_idx in range(self.max_iterations):
            logging.info(f"Chain of RAG iteration {iter_idx + 1}/{self.max_iterations}")
            
            # 1. Generate follow-up query
            intermediate_context_str = "\n".join(intermediate_contexts)
            follow_up_query = self._generate_follow_up_query(query, intermediate_context_str)
            
            # 2. Retrieve context for the follow-up query
            retrieved_documents = self._retrieve_context(follow_up_query)
            all_retrieved_documents.extend(retrieved_documents)
            
            # 3. Generate intermediate answer
            intermediate_answer = self._generate_intermediate_answer(follow_up_query, retrieved_documents)
            
            # 4. Format and add to intermediate context
            context_entry = f"Intermediate query{len(intermediate_contexts) + 1}: {follow_up_query}\nIntermediate answer{len(intermediate_contexts) + 1}: {intermediate_answer}"
            intermediate_contexts.append(context_entry)
            
            # Store iteration debug info
            debug_info["iterations"].append({
                "query": follow_up_query,
                "num_documents": len(retrieved_documents),
                "answer_preview": intermediate_answer[:100] + "..." if len(intermediate_answer) > 100 else intermediate_answer
            })
            
            # 5. Check if we have enough information (early stopping)
            if self.early_stopping and iter_idx < self.max_iterations - 1:  # Don't check on the last iteration
                if self._check_has_enough_info(query, intermediate_context_str):
                    logging.info(f"Early stopping after iteration {iter_idx + 1}: Have enough information")
                    debug_info["early_stopped"] = True
                    break
        
        # Deduplicate retrieved documents based on document_name and page_number
        unique_docs = {}
        for doc in all_retrieved_documents:
            key = f"{doc.get('document_name', 'unknown')}_{doc.get('page_number', 'unknown')}"
            if key not in unique_docs:
                unique_docs[key] = doc
        
        deduplicated_docs = list(unique_docs.values())
        debug_info["total_documents"] = len(deduplicated_docs)
        
        # Generate final answer
        final_answer = self._generate_final_answer(
            query=query,
            intermediate_context=intermediate_contexts,
            retrieved_documents=deduplicated_docs
        )
        
        logging.info(f"Generated final answer with Chain of RAG (first 200 chars): {final_answer[:200]}...")
        
        return final_answer, debug_info 