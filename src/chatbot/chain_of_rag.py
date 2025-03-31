import logging
import os
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
        
        # Check if we're using the API key integration (google.generativeai package)
        self.using_api_key = 'google.generativeai' in str(type(self.llm))
        
        if self.using_api_key:
            # For google.generativeai, we'll set up generation parameters
            self.chain_generation_params = {
                "temperature": 0,
                "top_p": 1,
                "max_output_tokens": 8192,
            }
            # Import the generativeai module for configuration
            try:
                import google.generativeai as genai
                self.genai = genai
                self.chain_generation_config = genai.GenerationConfig(**self.chain_generation_params)
            except ImportError:
                logging.error("Failed to import google.generativeai module")
                self.chain_generation_config = None
        else:
            # Use specific generation config for Chain of RAG with Vertex AI
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
        prompt = f"""<|system|>
**Your Role:** You are an Expert Research Strategist and Decomposer, guiding a RAG system with a fixed, internal knowledge base. Your mission is to iteratively formulate the *single most strategically valuable* follow-up question to build a *comprehensive, accurate, and deeply informative* answer to the main query, using only the available internal documents.

**Your Goal:** Generate the *next specific query* for the internal semantic search tool. This query should target the most critical information gap, considering previous findings, the required level of detail, and explicitly acknowledging limitations of the knowledge base revealed by prior searches.

**CRITICAL CONSTRAINTS:**
1.  **Internal Knowledge Only:** The search tool accesses ONLY information within the provided context (previous answers) and the underlying indexed documents. Assume no external knowledge or access.
2.  **No External Resource Requests:** NEVER ask for external URLs, websites, document repositories, or information outside the system's scope.
3.  **Acknowledge Unavailability:** If previous searches for a specific detail (e.g., "internal algorithm X", "specific parameter value Y", "exact log format Z") consistently resulted in "No relevant information found" or explicit statements of absence, treat that specific level of granular detail as **confirmed unavailable**.
4.  **Avoid Futile Repetition:** DO NOT re-ask queries (even rephrased) targeting information already confirmed as unavailable (Constraint 3). This wastes resources. If a specific approach failed, pivot.

<|user|>
**Context:**
<main_query>
{main_query}
</main_query>

<previous_context>
{intermediate_context}
</previous_context>
*(If empty, this is the first iteration)*

**Task:**
Generate the *single best* follow-up question by executing the following reasoning process:

**Reasoning Process:**

1.  **Analyze Situation & Goal:**
    *   **Main Query Objective & Depth:** Re-read the `<main_query>`. What specific information (e.g., steps, components, definitions, relationships, configurations, *mechanisms*, *parameters*, *reasons*) is ultimately required for a *complete and technically deep* answer? What level of detail seems necessary (overview vs. specific interactions)? What would a good structure for the final answer look like?
    *   **Previous Context Review:** Examine `<previous_context>`:
        *   **Successes:** What parts of the main query have been addressed? What key O-RAN terms, procedures (e.g., "M-Plane startup", "NETCONF RPCs", "PTP sync states"), entities, parameters, or document sections were successfully identified and described? Was the *required depth* achieved for these parts?
        *   **Failures & Gaps:** What parts of the main query remain unanswered or lack sufficient depth? Crucially, identify any specific information targeted by previous queries that resulted in "No relevant information found" or explicit confirmation of absence. Note the *semantic core* and *level of detail* of what was missing (e.g., "specific internal O-RU processing logic", "exact value for parameter X").
        *   **Implicit Structure:** Note how the gathered information is starting to form parts of the potential final answer structure.

2.  **Identify Most Critical Gap & Likely Availability:**
    *   Based on the analysis (including required depth), what is the *most significant* remaining gap in information needed to fulfill the `<main_query>` objective comprehensively and accurately?
    *   Critically assess: Is information likely to exist in the documents to fill this *specific* gap at the *required level of detail*, given previous successes and failures? Avoid pursuing details previously shown to be unavailable.

3.  **Select Query Strategy (Choose ONE):**
    *   **(A) Drill-Down for Depth/Mechanism:** If a previous answer successfully identified a relevant concept/procedure but lacked the depth, *parameters*, *mechanisms*, *specific steps*, or *reasons* required by the `<main_query>` objective, formulate a question asking for these *specific aspects about that concept AS DESCRIBED IN THE DOCUMENTS*. **Use this if the concept is central and needs deeper explanation.** (e.g., "What specific parameters and mechanisms does `o-ran-fm.yang` define for fault reporting according to the documents?" or "What are the detailed steps and reasons for the O-RU stopping transmission during FREERUN state based on the sources?")
    *   **(B) Explore Unaddressed Component:** If a distinct, significant component or aspect of the `<main_query>` has not been addressed *and* information for it hasn't been previously shown to be unavailable, formulate a question targeting *that specific component*. **Use this to ensure breadth of coverage.** (e.g., If the main query covers C/U/S plane impacts and S-plane hasn't been detailed, ask: "How specifically does jitter impact S-Plane PTP synchronization accuracy according to the documents?")
    *   **(C) Strategic Pivot after Failure:** If the *most recent* query failed (returned "No relevant information" or similar) for a specific detail:
        *   **Option C1 (Alternative Component):** If other significant components of the main query remain unexplored, switch focus to one of them (Use Strategy B).
        *   **Option C2 (Higher-Level View/Purpose):** If the failed query sought very specific details, try asking about the *overall process*, *purpose*, or *key principles* of that concept *as described in the documents*, if that higher level view is still relevant and wasn't covered, and might yield useful context. (e.g., If "specific calculation for `ta3`" failed, try "What is the purpose and factors influencing the `ta3` processing deadline in the O-RU based on the documents?")
        *   **Option C3 (Related Concept):** If a related, relevant concept was mentioned in *earlier successful* answers, consider exploring that instead (Use Strategy A on that concept, focusing on depth/mechanism).
        *   **Crucially:** *Do not* simply rephrase the failed query. Pick a genuinely different angle or target based on C1, C2, or C3.

4.  **Craft the Follow-up Question:**
    *   Formulate a *simple, unambiguous question* based *strictly* on the chosen strategy (A, B, or C).
    *   Use precise O-RAN terminology identified from the `<main_query>` or successful `<previous_context>`.
    *   Frame the question to elicit *specific details, mechanisms, parameters, steps, or reasons* contained within the documents (e.g., "What parameters define...", "Describe the mechanism for X...", "What steps are involved in Y...", "How is Z configured/handled according to the sources?").
    *   Ensure the question focuses on a *single*, well-defined target.

5.  **Final Check:**
    *   Does the question directly advance the goal of answering the `<main_query>` comprehensively and accurately?
    *   Does it seek the appropriate *level of detail*?
    *   Does it strictly adhere to ALL constraints (Internal Knowledge, No External Links, Avoid Futile Repetition)?
    *   Is it the most strategic choice based on the current context and reasoning steps?

**Output Format:**
Respond ONLY with the text of the follow-up question. Do NOT include explanations, apologies, introductions, or any other surrounding text.

**Follow-up Question:**
<|assistant|>
"""
        # Handle different API types
        if self.using_api_key:
            # For google.generativeai, we can directly pass the text
            response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
        else:
            # For vertexai.generative_models, we need to create a Content object
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
            num_neighbors=10,
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

        prompt = f"""<|system|>
**Your Role:** You are a Critical Information Synthesizer and Relevance Assessor. Your primary task is to determine if the provided documents *directly and specifically* answer the given query, including necessary details like mechanisms or parameters, and if so, synthesize a concise and accurate answer using *only* that relevant information.

**Your Goal:** Create a trustworthy intermediate answer that:
    1.  Rigorously assesses if the provided documents contain information that *specifically addresses the core question and required level of detail* asked in the `<query>`.
    2.  If relevant information exists, synthesizes it accurately, including key *mechanisms, parameters, steps, reasons, relationships, or contrasts* found in the text, relying solely on the evidence within the supplied snippets.
    3.  Clearly indicates when the documents, despite potentially containing related keywords, *do not actually answer the specific question asked* or provide the *required level of detail*.
    4.  Avoids hallucination and external knowledge.
<|user|>
**Context:**
<query>
{query}
</query>

<documents>
{context_text}
</documents>
*(Each document block `<Document i>` contains 'Content' and 'Metadata' including 'Document Name' and 'Page'.)*

**Task:**
1.  **Critically Analyze Relevance & Detail:** Carefully read the `<query>`. Understand the specific question, the *type of information needed* (e.g., process, definition, parameter list, comparison, mechanism), and the *implied level of detail*. Then, examine each document in `<documents>`. Determine if the document's content provides a **direct and specific answer** to the question, including the necessary details (e.g., does it explain *how* or *why*? Does it list the *parameters* involved? Does it detail the *steps*? Does it provide the *contrast* asked for?). Do not rely on keyword matching alone; assess semantic meaning, intent, and the presence of required specifics.

2.  **Synthesis Assignment:**
    * If relevant information is found that specifically addresses the query at the needed level of detail:
        * Synthesize a clear, accurate answer that preserves key terminology, parameters, and system mechanics from the source text.
        * Focus on the precise question asked and only include relevant information.
        * Ensure clarity; do not maintain ambiguities from the source text (e.g., be specific about which system is doing what).
        * Clearly label information drawn from different documents if they represent different perspectives, e.g., [From Doc 1: ...] vs [From Doc 3: ...] 
    * If no relevant information is found (content is unrelated to query) or information only partially addresses the query (relevant topic but inadequate detail):
        * Respond with "No relevant information found."
        * Briefly explain why the documents don't satisfy the query (e.g., "Documents discuss concept X but provide no details on steps/parameters/reasons," or "Documents mention system Y but not in relation to process Z").

**Prohibited Strategies:**
* Do NOT hallucinate or add external knowledge beyond what is in the documents.
* Do NOT rely solely on keyword matching. A document mentioning terms like those in the query is not necessarily answering the question's specific intent.
* Do NOT provide lengthy summaries of all documents. Focus on directly addressing the query.

**Output Format:**
If the documents contain relevant information: Provide the synthesized answer. Clearly cite document numbers [Doc X].
If not: State "No relevant information found." followed by a brief explanation of the mismatch.

**Response:**
<|assistant|>
"""
        # Handle different API types
        if self.using_api_key:
            # For google.generativeai, we can directly pass the text
            response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
        else:
            # For vertexai.generative_models, we need to create a Content object
            content = Content(role="user", parts=[Part.from_text(prompt)])
            response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
            
        intermediate_answer = response.text.strip() if response and response.text else ""
        logging.info(f"Intermediate Answer (first 200 chars): {intermediate_answer[:200]}...")
        return intermediate_answer

    def _check_has_enough_info(self, query: str, intermediate_contexts: List[str]) -> bool:
        """
        Checks if we have gathered enough information to answer the query.
        
        Args:
            query: Original user query
            intermediate_contexts: List of intermediate Q&A pairs
            
        Returns:
            True if we have enough information, False otherwise
        """
        if not intermediate_contexts:
            return False
            
        # Join intermediate contexts
        intermediate_context_text = "\n\n".join(intermediate_contexts)
        
        prompt = f"""<|system|>
**Your Role:** You are an Information Sufficiency Assessor helping a Chain-of-Thought system decide when to stop gathering information. Your job is to determine if enough relevant information has been collected to completely and accurately answer a user's query, or if more specific information is still needed.

**Your Goal:** Provide a binary assessment (YES/NO) of whether the collected information adequately covers all aspects of the query at the level of detail implicitly required.

<|user|>
**Context:**
<original_query>
{query}
</original_query>

<collected_information>
{intermediate_context_text}
</collected_information>

**Task:**
Analyze the original query and all collected information to decide if:

1. The query has been addressed **completely** - all aspects and sub-components of the question have information addressing them
2. The information is **sufficiently detailed** - the appropriate level of detail (e.g., mechanisms, parameters, reasons, steps) implied by the original query has been met
3. The information is **directly relevant** - the collected information directly pertains to the specific details asked in the query

Consider:
* Does the collected information directly address each aspect of the original query?
* Is the level of detail sufficient for the query's apparent purpose?
* Would a domain expert consider this information sufficient to give a complete, accurate answer?
* Are there any significant aspects, steps, mechanisms, or parameters mentioned in the query that remain unaddressed?

Respond with ONLY:
* "YES" if the collected information is sufficient to provide a complete, accurate, and appropriate answer to the query.
* "NO" if more information is needed on any significant aspect of the query or if the current information lacks the appropriate level of detail.

**Response (YES/NO only):**
<|assistant|>
"""
        try:
            # Handle different API types
            if self.using_api_key:
                # For google.generativeai, we can directly pass the text
                response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
            else:
                # For vertexai.generative_models, we need to create a Content object
                content = Content(role="user", parts=[Part.from_text(prompt)])
                response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
            
            result = response.text.strip().upper()
            has_enough = result.startswith("YES")
            
            if has_enough:
                logging.info("Early stopping after iteration {}: Have enough information")
            
            return has_enough
        except Exception as e:
            logging.error(f"Error checking if we have enough info: {e}")
            return False

    def _generate_final_answer(self, query: str, intermediate_context: List[str], retrieved_documents: List[Dict]) -> str:
        """
        Generates the final answer based on all the gathered information.
        
        Args:
            query: Original user query
            intermediate_context: List of intermediate Q&A pairs
            retrieved_documents: All retrieved document chunks across iterations
            
        Returns:
            Final answer
        """
        # Join intermediate contexts
        intermediate_context_text = "\n\n".join(intermediate_context)
        
        # Format document chunks for reference
        doc_references = ""
        for i, chunk in enumerate(retrieved_documents[:10], start=1):  # Limit to first 10 chunks for prompt size
            doc_meta = chunk.get("metadata", {})
            doc_name = chunk.get("document_name", "N/A")
            page = chunk.get("page_number", "N/A")
            version = doc_meta.get("version", "unknown")
            
            doc_references += f"[{i}] {doc_name}, page {page}, version {version}\n"
            
        prompt = f"""<|system|>
**Your Role:** You are a Senior Technical Writer for O-RAN Alliance who specializes in producing comprehensive, well-structured answers from collected research. Your job is to transform research notes into a definitive, readable, technically-rich answer to a user query.

**Your Goal:** Create a clear, structured, and thorough answer using only the collected research notes, ensuring it fully addresses all aspects of the original query.

<|user|>
**Context:**
<original_query>
{query}
</original_query>

<collected_information>
{intermediate_context_text}
</collected_information>

<document_references>
{doc_references}
</document_references>

**Task:**
Synthesize a comprehensive, authoritative answer to the original query using only the collected information. Your answer should:

1. **Structure:** Create a well-organized response, potentially including sections, bullet points, or step-by-step formats when appropriate for clarity. Complex topics deserve well-structured explanations.

2. **Depth:** Provide a thorough explanation that covers all aspects of the query, including technical details, parameters, mechanisms, reasons, or contrasts identified in the collected information.

3. **Precision:** Use precise O-RAN terminology and technical language found in the collected information. Maintain technical accuracy while ensuring readability.

4. **References:** Where helpful for credibility, include numbered references to source documents from <document_references> using [1], [2], etc.

5. **Completeness:** Ensure all aspects of the original query are addressed completely.

**Answer Format:**
- Start with "## " followed by a concise title summarizing the topic
- Include clear section headings where appropriate for complex topics
- Use bulleted lists or numbered steps where helpful for clarity
- Include appropriate technical details: parameters, methodologies, procedures

**Response:**
<|assistant|>
"""
        # Handle different API types
        if self.using_api_key:
            # For google.generativeai, we can directly pass the text
            response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
        else:
            # For vertexai.generative_models, we need to create a Content object
            content = Content(role="user", parts=[Part.from_text(prompt)])
            response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
            
        final_answer = response.text.strip() if response and response.text else ""
        logging.info(f"Generated final answer with Chain of RAG (first 200 chars): {final_answer[:200]}...")
        return final_answer

    def process_query(self, query: str, conversation_history: List[Dict] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Processes a user query using the Chain of RAG approach.
        
        Args:
            query: User query
            conversation_history: Optional conversation history
            
        Returns:
            Tuple containing the final answer and a debug context dictionary
        """
        if conversation_history is None:
            conversation_history = []
            
        try:
            # Initialize tracking
            all_retrieved_docs = []
            intermediate_contexts = []
            follow_up_queries = []
            iteration_details = []
            
            # Main Chain of RAG loop
            for i in range(1, self.max_iterations + 1):
                logging.info(f"Chain of RAG iteration {i}/{self.max_iterations}")
                
                # Step 1: Generate follow-up query
                follow_up_query = self._generate_follow_up_query(query, intermediate_contexts)
                follow_up_queries.append(follow_up_query)
                
                # Step 2: Retrieve relevant documents
                retrieved_docs = self._retrieve_context(follow_up_query)
                all_retrieved_docs.extend(retrieved_docs)
                
                # Step 3: Generate intermediate answer
                intermediate_answer = self._generate_intermediate_answer(follow_up_query, retrieved_docs)
                
                # Format the context entry
                context_entry = f"Follow-up Query: {follow_up_query}\n\nIntermediate Answer: {intermediate_answer}"
                intermediate_contexts.append(context_entry)
                
                # Track iteration details for debugging
                iteration_details.append({
                    "iteration": i,
                    "follow_up_query": follow_up_query,
                    "num_docs_retrieved": len(retrieved_docs),
                    "intermediate_answer": intermediate_answer[:500] + ("..." if len(intermediate_answer) > 500 else "")
                })
                
                # Step 4: Check if we have enough information
                if self.early_stopping and i >= 2:  # Need at least 2 iterations
                    if self._check_has_enough_info(query, intermediate_contexts):
                        logging.info(f"Early stopping after iteration {i}: Have enough information")
                        break
            
            # Step 5: Generate final answer
            final_answer = self._generate_final_answer(query, intermediate_contexts, all_retrieved_docs)
            
            # Prepare debug context
            debug_context = {
                "original_query": query,
                "iterations": iteration_details,
                "num_iterations": len(iteration_details),
                "early_stopped": len(iteration_details) < self.max_iterations and self.early_stopping,
                "total_docs_retrieved": len(all_retrieved_docs)
            }
            
            return final_answer, debug_context
            
        except Exception as e:
            # Log the error
            logging.error(f"Error in Chain of RAG processing: {e}", exc_info=True)
            
            # Check if this is a Content object related error that might be happening with google.generativeai
            if 'Content' in str(e) and self.using_api_key:
                error_message = (
                    "## Error Processing Query\n\n"
                    "An error occurred while processing your query with the Chain of RAG pipeline. "
                    "There seems to be an issue with the API integration. Please try using the standard RAG pipeline instead.\n\n"
                    f"Error details: {str(e)}"
                )
            else:
                # Generic error message
                error_message = (
                    "## Error Processing Query\n\n"
                    "An error occurred while processing your query with the Chain of RAG pipeline. "
                    "Please try rephrasing your question or try again later.\n\n"
                    f"Error details: {str(e)}"
                )
            
            return error_message, {"error": str(e)} 