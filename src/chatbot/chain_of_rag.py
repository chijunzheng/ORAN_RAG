import logging
import os
import ast  # Import ast for literal_eval
from typing import List, Dict, Tuple, Any
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part
)

# --- Prompt Templates ---

FOLLOWUP_QUERY_PROMPT = """<|system|>
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
{query}
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
<|assistant|>"""

INTERMEDIATE_ANSWER_PROMPT = """<|system|>
**Your Role:** You are a Critical Information Synthesizer and Relevance Assessor. Your primary task is to determine if the provided documents *directly and specifically* answer the given query, including necessary details like mechanisms or parameters, and if so, synthesize a concise and accurate answer using *only* that relevant information.

**Your Goal:** Create a trustworthy intermediate answer that:
    1.  Rigorously assesses if the provided documents contain information that *specifically addresses the core question and required level of detail* asked in the `<query>`.
    2.  If relevant information exists, synthesizes it accurately, including key *mechanisms, parameters, steps, reasons, relationships, or contrasts* found in the text, relying solely on the evidence within the supplied snippets.
    3.  Clearly indicates when the documents, despite potentially containing related keywords, *do not actually answer the specific question asked* or provide the *required level of detail*.
    4.  Avoids hallucination and external knowledge.
<|user|>
**Context:**
<query>
{sub_query}
</query>

<documents>
{retrieved_documents}
</documents>
*(Each document block `<Document i>` contains 'Content' and 'Metadata' including 'Document Name' and 'Page'.)*

**Task:**
1.  **Critically Analyze Relevance & Detail:** Carefully read the `<query>`. Understand the specific question, the *type of information needed* (e.g., process, definition, parameter list, comparison, mechanism), and the *implied level of detail*. Then, examine each document in `<documents>`. Determine if the document's content provides a **direct and specific answer** to the question, including the necessary details (e.g., does it explain *how* or *why*? Does it list the *parameters* involved? Does it detail the *steps*? Does it provide the *contrast* asked for?). Do not rely on keyword matching alone; assess semantic meaning, intent, and the presence of required specifics.
2.  **Generate Answer OR Indicate Irrelevance:**
    *   **If relevant documents ARE found:** Synthesize an answer based *exclusively* on the portions of those documents that *directly* answer the `<query>` with the appropriate detail. Follow the Response Guidelines below.
    *   **If NO documents directly answer the `<query>` OR provide the necessary level of detail:** (Even if they discuss related topics or contain keywords) Respond *only* with the exact phrase: `No relevant information found`

**Response Guidelines (Only applies if relevant information is found):**

1.  **Strict Source Adherence:** Base your entire response *only* on the specific information within the `Content:` sections that *directly answers the `<query>`*. Exclude tangential information. DO NOT HALLUCINATE.
2.  **Focus on Query & Detail:** Ensure the synthesized answer squarely addresses the specific question asked in the `<query>`, including its nuances and required level of detail (mechanisms, parameters, steps, etc.) *if present in the source text*.
3.  **Content Synthesis:** Combine the *strictly relevant* information into a clear, coherent answer.
    *   Prioritize extracting information that explains *how* or *why*, includes specific *parameters* or *mechanisms* mentioned, details *process steps*, explains *relationships* between components, or provides *contrasts/comparisons* if requested by the query.
    *   If the query asks for a process or steps, provide only the steps *relevant to the query* found in the documents.
4.  **Referencing (CRITICAL):**
    *   Cite the source for your statements using the `Document Name` and `Page` from the `Metadata:` associated with the relevant `<Document i>` block.
    *   Format citations as: `(Document Name, page Page Number)`. Example: `(O-RAN.WG4.MP.0-R004-v16.01.pdf, page 20)`
    *   Place citations *immediately after* the specific information they support.
    *   If multiple documents support the *exact same* specific statement, list citations sequentially: `(DocA, page 10)(DocB, page 5)`.
5.  **Handling Conflicts:** If different documents provide conflicting information *specifically regarding the answer to the query*, state the conflict clearly and cite all involved sources.
6.  **Clarity and Conciseness:** Provide a direct answer. Avoid jargon unless defined or essential *and present in the source text*.

**Output:**
Provide the synthesized answer based *only* on directly relevant information (including mechanisms, parameters, steps etc. if found), properly referenced according to the guidelines. If no documents directly answer the specific question asked or provide the necessary detail, output only `No relevant information found`.

<|assistant|>"""

FINAL_ANSWER_PROMPT = """Given the following intermediate queries and answers, generate a final answer for the main query by combining relevant information. Note that intermediate answers are generated by an LLM and may aim to extract specific details like mechanisms or parameters, but may not always be fully comprehensive on their own.

## Documents
{retrieved_documents}

## Intermediate queries and answers
{intermediate_context}

## Main query
{query}

Please provide a comprehensive and technically accurate answer following these specific formatting guidelines:

1. Structure:
   - Use '##' for main sections and '###' for subsections
   - Present information in bullet points or numbered lists where appropriate for clarity
   - Maintain consistent indentation for nested lists
   - Each section should be logically organized and flow naturally

2. Content Guidelines:
   - Focus on delivering a complete answer that fully addresses **all aspects and the required depth** of the original `<main_query>`. Refer back to the `<main_query>` to ensure full coverage.
   - Synthesize information logically, explaining **how** components interact, **why** certain procedures are followed, and detailing relevant **mechanisms** or **parameters** identified in the intermediate answers. Draw connections between different pieces of information.
   - Be logical and concise while providing necessary technical detail. Ensure accuracy, paying attention to **subtle technical nuances** (e.g., correct parameter names, precise definitions, distinctions between similar concepts) identified in the intermediate answers.
   - Present explanations clearly, often step-by-step for processes.
   - Write in a professional and informative tone, targeted for engineers potentially new to the specific O-RAN topic but capable of understanding technical detail.
   - **CONSOLIDATE** information from multiple intermediate answers/sources into single, comprehensive bullet points where appropriate, **without losing critical details, parameters, or mechanism explanations**.

3. Reference Formatting (EXTREMELY IMPORTANT):
   - Add reference numbers in square brackets ONLY at the END of each complete bullet point or statement: [1], [2], etc.
   - NEVER put references within sentences or in the middle of bullet points.
   - NEVER use comma-separated references like [1,2,3] - this is incorrect.
   - Place references as separate adjacent brackets at the end: [1][2][3]
   - References must be placed IMMEDIATELY after the text they support (no space between text and reference).
   - Aim to use FEWER reference numbers overall by grouping related information from the *same logical point* in the intermediate answers together under consolidated final points.
   - Never reference the same source document multiple times for the *same synthesized statement* within a single bullet point.
   - INCORRECT: "The O-RAN architecture [1, 2, 3] includes multiple components."
   - CORRECT: "The O-RAN architecture includes multiple components, such as the O-CU and O-DU, which communicate via the F1 interface.[1][2][3]"

4. Reference Mapping Section:
   - At the very end of your answer, include a "## References" section.
   - List ALL unique referenced documents/pages with their assigned numbers based on their appearance in the final text.
   - Format as:
     - [1] Document Name, page Page Number
     - [2] Another Document, page Page Number
     - etc.
   - Use actual document names (e.g., "O-RAN.WG4.MP.0-R004-v16.01.pdf") rather than generic "Document X" references.

5. Example Format:
   ## Main Section Title
   - Solution 1 of the O-RAN RIC API uses JSON for message encoding. This format defines the structure for data interchange between the Near-RT RIC platform and consuming applications or services.[1][2][3]
   - The protocol stack supporting Solution 1 typically involves HTTP as the application protocol, potentially secured using TLS over a TCP transport layer connection.[4]

   ## References
   - [1] O-RAN.WG1.OAD-R003-v12.00.pdf, page 15
   - [2] O-RAN.WG4.MP.0-R004-v16.01.pdf, page 20
   - [3] O-RAN.WG7.IPC-HRD-Opt8.0-v03.00.pdf, page 96
   - [4] O-RAN.WG10.OAM-Architecture-R004-v13.00.pdf, page 43

IMPORTANT: Keep your writing style fluid and natural. Place references ONLY at the end of complete statements or bullet points. Consolidate information effectively, but ensure that critical technical details, parameters, mechanisms, and explanations identified in the intermediate answers are preserved and accurately integrated into the final response. The goal is a clean, readable, accurate, and *comprehensively detailed* answer that fully addresses the user's original query."""


REFLECTION_PROMPT = """<|system|>
**Your Role:** You are a Meticulous Quality Assurance Analyst. Your task is to determine if the information gathered so far is *sufficiently comprehensive* to fully answer the main query to a high standard.

**Your Goal:** Make a conservative judgment ("Yes" or "No") on information sufficiency. It is crucial to avoid stopping the research process prematurely. Err on the side of gathering more information if *any* doubt exists.
<|user|>
**Context:**
<main_query>
{query}
</main_query>

<intermediate_context>
{intermediate_context}
</intermediate_context>
*(This contains the sequence of sub-queries asked and the answers generated so far.)*

**Task:**
Assess whether the `<intermediate_context>` provides *truly comprehensive* information to fully answer all aspects of the `<main_query>`.

**Assessment Checklist (Mentally perform these checks):**

1.  **Query Coverage:** Does the gathered information address *all* key aspects and sub-questions implied by the `<main_query>`? Are there any obvious parts of the query left untouched?
2.  **Depth & Specificity:** Is the information detailed enough? Does it include specific technical details, names, protocols, values, process steps, etc., as appropriate for the query, or is it still too general?
3.  **Consistency & Verification:** Does the information seem consistent across different intermediate answers? Has key information been potentially corroborated by different documents (judging by the nature of the intermediate answers)?
4.  **Completeness:** Could a final answer synthesized *only* from this context be considered complete by someone knowledgeable in the domain? Or would it leave significant questions unanswered?
5.  **Ambiguity:** Are there remaining ambiguities or vague statements from the intermediate answers that need clarification before generating a final answer?

**Decision Rule:**
Set an *extremely high bar* for sufficiency. Only respond "Yes" if you are highly confident that the information is comprehensive, detailed, specific, and covers *all* facets of the main query, leaving no significant gaps or ambiguities. If *any* criterion above raises doubt, or if you feel more detail *could* be beneficial, you MUST respond "No".

**Output Format:**
Respond ONLY with the word `Yes` or `No`. Do NOT provide explanations, justifications, or any other text.

<|assistant|>"""

GET_SUPPORTED_DOCS_PROMPT = """<|system|>
**Your Role:** You are a Document-Answer Matcher. Your task is to identify which of the provided documents contain information that directly supports the given Answer to the Question.

**Your Goal:** Produce a precise list of indices corresponding to the supporting documents.
<|user|>
**Context:**
<documents>
{retrieved_documents}
</documents>
*(Each document block starts with `<Document i>` where 'i' is the index, followed by 'Content' and 'Metadata'.)*

<qa_pair>
  <query>
  {query}
  </query>
  <answer>
  {answer}
  </answer>
</qa_pair>

**Task:**
Examine the `Content:` of each `<Document i>` in the `<documents>` section. Identify which documents contain text that directly states, confirms, or elaborates on the information presented in the `<answer>` provided for the `<query>`.

**Instructions:**

1.  For each `<Document i>`, compare its `Content:` against the statements made in the `<answer>`.
2.  If a document's content provides evidence for the answer, note its index `i`.
3.  Compile a list of these indices `i`.

**Output Format:**
Respond ONLY with a valid Python list of integers representing the indices (`i`) of the supporting documents. The list should look like `[0, 2, 5]`. If no documents support the answer, respond with an empty list `[]`. Do not include any other text, explanation, or formatting.

<|assistant|>"""

# --- Class Definition ---

class ChainOfRagProcessor:
    """
    Implements Chain of RAG for the ORAN RAG system following a specific logic
    including document support validation.
    """
    def __init__(
        self,
        vector_searcher,
        llm,
        generation_config: GenerationConfig,
        index_endpoint_display_name: str,
        deployed_index_id: str,
        reranker=None,  # Add reranker parameter
        max_iterations: int = 4,
        early_stopping: bool = True,
        search_neighbors: int = 10,  # Add parameter for initial search results count
        rerank_top_n: int = 5,  # Add parameter for number of reranked results to keep
    ):
        self.vector_searcher = vector_searcher
        self.llm = llm  # generative model (e.g., gemini-1.5-flash-002)
        self.generation_config = generation_config
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        self.index_endpoint_display_name = index_endpoint_display_name
        self.deployed_index_id = deployed_index_id
        self.reranker = reranker  # Store reranker
        self.search_neighbors = search_neighbors  # Store search_neighbors
        self.rerank_top_n = rerank_top_n  # Store rerank_top_n

        # Check if we're using the API key integration (google.generativeai package)
        self.using_api_key = 'google.generativeai' in str(type(self.llm))

        logging.info(f"ChainOfRagProcessor initialized with max_iterations={max_iterations}, early_stopping={early_stopping}")
        logging.info(f"Reranker {'is available' if reranker else 'is not available'}")
        logging.info(f"Using API key integration: {self.using_api_key}")

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
                logging.info("Successfully initialized genai.GenerationConfig")
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
            logging.info("Successfully initialized VertexAI GenerationConfig")

    def _format_retrieved_results(self, retrieved_documents: List[Dict]) -> str:
        """
        Formats retrieval results (list of dictionaries) for prompts.
        """
        formatted_documents = []
        for i, doc in enumerate(retrieved_documents):
            content = doc.get('content', 'No content')
            metadata = doc.get('metadata', {})
            doc_name = metadata.get("document_name", doc.get("document_name", "Unknown document"))
            version = metadata.get("version", "Unknown version")
            workgroup = metadata.get("workgroup", "Unknown workgroup")
            subcategory = metadata.get("subcategory", "Unknown subcategory")
            page_num = metadata.get("page_number", doc.get("page_number", "N/A"))
            context_meta = metadata.get("context", "") # Assuming context might be in metadata

            # Format page number display
            if page_num is None or page_num == "Unknown Page":
                page_num = "N/A"
            
            # Format context string if it exists
            context_str = f"\n- Context: {context_meta}" if context_meta else ""

            formatted_documents.append(
                f"<Document {i}>\n"
                f"Content: {content}\n\n"
                f"Metadata:\n"
                f"- Document Name: {doc_name}\n"
                f"- Version: {version}\n"
                f"- Workgroup: {workgroup}\n"
                f"- Subcategory: {subcategory}\n"
                f"- Page: {page_num}{context_str}"
                # Removed the duplicate </Document {i}> tag here
            )
        return "\n\n".join(formatted_documents)

    def _get_supported_docs(self, retrieved_documents: List[Dict], query: str, answer: str) -> List[Dict]:
        """
        Filters retrieved documents to keep only those supporting the answer.
        Uses GET_SUPPORTED_DOCS_PROMPT and ast.literal_eval for safe parsing.
        """
        if not retrieved_documents or "No relevant information found" in answer:
            return retrieved_documents # Return all if no answer or no docs initially
        
        formatted_docs = self._format_retrieved_results(retrieved_documents)
        prompt = GET_SUPPORTED_DOCS_PROMPT.format(
            retrieved_documents=formatted_docs,
            query=query,
            answer=answer
        )
        
        logging.debug(f"Get Supported Docs Prompt (start):\n{prompt[:300]}...")
        logging.debug(f"Get Supported Docs Prompt (end):\n...{prompt[-300:]}")
        
        try:
            if self.using_api_key:
                response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
            else:
                content = Content(role="user", parts=[Part.from_text(prompt)])
                response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
            
            response_text = response.text.strip() if response and response.text else '[]'
            logging.info(f"Get Supported Docs LLM Response: {response_text}")
            
            # Safely parse the string representation of the list
            try:
                supported_indices = ast.literal_eval(response_text)
                if not isinstance(supported_indices, list):
                    raise ValueError("LLM response is not a list")
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Failed to parse LLM response for supported docs: '{response_text}'. Error: {e}. Using all retrieved docs.")
                return retrieved_documents
            
            # Filter documents based on valid indices
            valid_indices = {idx for idx in supported_indices if isinstance(idx, int) and 0 <= idx < len(retrieved_documents)}
            supported_docs = [retrieved_documents[i] for i in sorted(list(valid_indices))]
            logging.info(f"Identified {len(supported_docs)} supporting documents out of {len(retrieved_documents)}.")
            return supported_docs

        except Exception as e:
            logging.error(f"Error in _get_supported_docs LLM call: {e}", exc_info=True)
            # Fallback to returning all documents in case of error
            return retrieved_documents
            
    def _generate_follow_up_query(self, query: str, intermediate_context: List[str]) -> str:
        """
        Generates a follow-up query using FOLLOWUP_QUERY_PROMPT.
        """
        prompt = FOLLOWUP_QUERY_PROMPT.format(
            query=query,
            intermediate_context="\n".join(intermediate_context)
        )
        # Log prompt for debugging
        logging.debug(f"Follow-up Query Prompt (start):\n{prompt[:300]}...")
        logging.debug(f"Follow-up Query Prompt (end):\n...{prompt[-300:]}")
        
        try:
            if self.using_api_key:
                response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
            else:
                content = Content(role="user", parts=[Part.from_text(prompt)])
                response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
            
            follow_up_query = response.text.strip() if response and response.text.strip() else ""
            logging.info(f"Generated Follow-up Query: {follow_up_query}")
            return follow_up_query
        except Exception as e:
            logging.error(f"Error generating follow-up query: {e}", exc_info=True)
            return "" # Return empty string on error

    def _generate_intermediate_answer(self, sub_query: str, retrieved_documents: List[Dict]) -> str:
        """
        Generates an intermediate answer using INTERMEDIATE_ANSWER_PROMPT.
        """
        if not retrieved_documents:
             logging.warning("No documents provided for intermediate answer generation.")
             return "No relevant information found because the retrieval step yielded no documents."

        formatted_docs = self._format_retrieved_results(retrieved_documents)
        prompt = INTERMEDIATE_ANSWER_PROMPT.format(
            retrieved_documents=formatted_docs,
            sub_query=sub_query
        )
        
        # Log the beginning and end of the prompt for debugging
        prompt_preview_len = 300
        logging.debug(f"Intermediate Answer Prompt (start):\n{prompt[:prompt_preview_len]}...")
        logging.debug(f"Intermediate Answer Prompt (end):\n...{prompt[-prompt_preview_len:]}")
        logging.debug(f"Intermediate Answer Prompt Length: {len(prompt)} characters")
        
        try:
            if self.using_api_key:
                response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
            else:
                content = Content(role="user", parts=[Part.from_text(prompt)])
                response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
                
            intermediate_answer = response.text.strip() if response and response.text else ""
            # Handle potential empty response explicitly
            if not intermediate_answer:
                 intermediate_answer = "No relevant information found" # Default if LLM returns empty
                 logging.warning("Intermediate answer LLM returned empty response. Defaulting to 'No relevant information found'.")
            
            logging.info(f"Intermediate Answer generated (first 100 chars): {intermediate_answer[:100]}...")
            return intermediate_answer
        except Exception as e:
            logging.error(f"Error during intermediate answer generation LLM call: {e}", exc_info=True)
            return "Error: Failed to generate intermediate answer due to an LLM error."

    def _check_has_enough_info(self, query: str, intermediate_contexts: List[str]) -> bool:
        """
        Checks if enough info has been gathered using REFLECTION_PROMPT.
        """
        if not intermediate_contexts:
            return False
            
        intermediate_context_text = "\n".join(intermediate_contexts)
        prompt = REFLECTION_PROMPT.format(
            query=query,
            intermediate_context=intermediate_context_text
        )
        
        logging.debug(f"Check Enough Info Prompt (start):\n{prompt[:300]}...")
        logging.debug(f"Check Enough Info Prompt (end):\n...{prompt[-300:]}")

        try:
            if self.using_api_key:
                response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
            else:
                content = Content(role="user", parts=[Part.from_text(prompt)])
                response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
            
            result = response.text.strip().upper()
            has_enough = result.startswith("YES")
            logging.info(f"Check Has Enough Info result: {result} -> {has_enough}")
            return has_enough
        except Exception as e:
            logging.error(f"Error checking if we have enough info: {e}", exc_info=True)
            return False # Assume not enough info on error

    def _generate_final_answer(self, query: str, intermediate_context: List[str], supported_documents: List[Dict]) -> str:
        """
        Generates the final answer using FINAL_ANSWER_PROMPT and supported documents.
        """
        intermediate_context_text = "\n\n".join(intermediate_context)
        formatted_docs = self._format_retrieved_results(supported_documents)
        
        prompt = FINAL_ANSWER_PROMPT.format(
            retrieved_documents=formatted_docs, # Using supported docs
            intermediate_context=intermediate_context_text,
            query=query
        )
        
        logging.debug(f"Final Answer Prompt (start):\n{prompt[:300]}...")
        logging.debug(f"Final Answer Prompt (end):\n...{prompt[-300:]}")
        logging.debug(f"Final Answer Prompt Length: {len(prompt)} characters")
        
        try:
            if self.using_api_key:
                response = self.llm.generate_content(prompt, generation_config=self.chain_generation_config)
            else:
                content = Content(role="user", parts=[Part.from_text(prompt)])
                response = self.llm.generate_content(content, generation_config=self.chain_generation_config)
            
            final_answer = response.text.strip() if response and response.text else ""
            logging.info(f"Final Answer generated (first 100 chars): {final_answer[:100]}...")
            return final_answer
        except Exception as e:
            logging.error(f"Error generating final answer: {e}", exc_info=True)
            return "Error: Failed to generate the final answer."

    def _retrieve_context(self, query: str) -> List[Dict]:
        """
        Retrieves ORAN document chunks relevant to the query.
        If a reranker is available, reranks the top results.
        """
        if not self.vector_searcher:
            logging.error("No vector_searcher available for retrieval.")
            return []
        
        # Retrieve documents using vector search
        try:
            retrieved = self.vector_searcher.vector_search(
                query_text=query, 
                num_neighbors=self.search_neighbors,
                index_endpoint_display_name=self.index_endpoint_display_name,
                deployed_index_id=self.deployed_index_id,
            )
        except Exception as e:
            logging.error(f"Vector search failed: {e}", exc_info=True)
            return [] # Return empty list if search fails
        
        logging.info(f"Retrieved {len(retrieved)} documents initially for query: '{query}'")
        
        # Apply reranking if reranker is available
        if self.reranker and retrieved:
            try:
                logging.info(f"Applying reranking to the {len(retrieved)} retrieved documents")
                reranked_docs = self.reranker.rerank(query=query, records=retrieved)
                
                # Limit to top N results
                reranked_results = reranked_docs[:self.rerank_top_n]
                logging.info(f"Reranked to top {len(reranked_results)} documents")
                
                # Log top scores for debugging
                if reranked_results:
                    top_scores = [f"{doc.get('rank_score', 0):.4f}" for doc in reranked_results[:3]]
                    logging.info(f"Top reranking scores: {', '.join(top_scores)}")
                
                return reranked_results
            except Exception as e:
                logging.error(f"Error during reranking: {e}. Using original vector search results.", exc_info=True)
                # Return original results if reranking fails
                return retrieved[:self.rerank_top_n] # Apply top_n even if rerank fails
        
        # If no reranker, return the top_n results from vector search
        return retrieved[:self.rerank_top_n]

    def process_query(self, query: str, conversation_history: List[Dict] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Processes a user query using the refined Chain of RAG approach.
        """
        if conversation_history is None:
            conversation_history = []
            
        try:
            # Initialize tracking
            all_supported_docs = [] # Store only supported documents across iterations
            intermediate_contexts = []
            follow_up_queries = []
            iteration_details = []
            
            logging.info(f"Starting Chain of RAG processing for query: '{query}'")
            logging.info(f"Conversation history has {len(conversation_history)} turns")
            
            # Main Chain of RAG loop
            for i in range(1, self.max_iterations + 1):
                logging.info(f"Chain of RAG iteration {i}/{self.max_iterations}")
                
                # Step 1: Generate follow-up query
                logging.info(f"Generating follow-up query #{i}...")
                follow_up_query = self._generate_follow_up_query(query, intermediate_contexts)
                if not follow_up_query:
                    logging.warning(f"Failed to generate follow-up query for iteration {i}. Ending process.")
                    break # Stop if query generation fails
                follow_up_queries.append(follow_up_query)
                logging.info(f"Follow-up query #{i}: '{follow_up_query}'")
                
                # Step 2: Retrieve relevant documents (initial retrieval + reranking)
                logging.info(f"Retrieving documents for follow-up query #{i}...")
                retrieved_docs = self._retrieve_context(follow_up_query)
                logging.info(f"Retrieved {len(retrieved_docs)} documents for follow-up query #{i} (after potential reranking)")
                
                # Step 3: Generate intermediate answer using retrieved docs
                logging.info(f"Generating intermediate answer for follow-up query #{i}...")
                intermediate_answer = self._generate_intermediate_answer(follow_up_query, retrieved_docs)
                logging.info(f"Intermediate answer #{i} (first 100 chars): {intermediate_answer[:100]}...")
                
                # Step 4: Get documents supporting the intermediate answer
                logging.info(f"Identifying documents supporting intermediate answer #{i}...")
                supported_docs = self._get_supported_docs(retrieved_docs, follow_up_query, intermediate_answer)
                all_supported_docs.extend(supported_docs)
                # Deduplicate based on 'id' field
                seen_ids = set()
                unique_supported_docs = []
                for doc in all_supported_docs:
                    if doc['id'] not in seen_ids:
                        unique_supported_docs.append(doc)
                        seen_ids.add(doc['id'])
                all_supported_docs = unique_supported_docs
                logging.info(f"Total unique supported documents accumulated: {len(all_supported_docs)}")
                
                # Format the context entry for reflection and final answer
                context_entry = f"Intermediate query{i}: {follow_up_query}\nIntermediate answer{i}: {intermediate_answer}"
                intermediate_contexts.append(context_entry)
                
                # Track iteration details for debugging
                iteration_details.append({
                    "iteration": i,
                    "follow_up_query": follow_up_query,
                    "num_docs_retrieved": len(retrieved_docs), # Initial retrieved count
                    "num_docs_supported": len(supported_docs), # Supported count for this iteration
                    "intermediate_answer": intermediate_answer[:500] + ("..." if len(intermediate_answer) > 500 else "")
                })
                
                # Step 5: Check if we have enough information (Reflection)
                if self.early_stopping and i < self.max_iterations: # Don't check on last iteration
                    logging.info(f"Checking if we have enough information after iteration {i}...")
                    if self._check_has_enough_info(query, intermediate_contexts):
                        logging.info(f"Early stopping after iteration {i}: Have enough information")
                        break
            
            # Step 6: Generate final answer using accumulated supported documents
            logging.info(f"Generating final answer using {len(all_supported_docs)} accumulated supported documents...")
            final_answer = self._generate_final_answer(query, intermediate_contexts, all_supported_docs)
            
            # Prepare debug context
            debug_context = {
                "original_query": query,
                "iterations": iteration_details,
                "num_iterations": len(iteration_details),
                "early_stopped": len(iteration_details) < self.max_iterations and self.early_stopping,
                "total_supported_docs_accumulated": len(all_supported_docs)
            }
            
            logging.info(f"Chain of RAG processing completed with {len(iteration_details)} iterations")
            return final_answer, debug_context
            
        except Exception as e:
            logging.error(f"Error in Chain of RAG processing: {e}", exc_info=True)
            error_message = (
                "## Error Processing Query\n\n"
                "An error occurred while processing your query with the Chain of RAG pipeline. "
                "Please try rephrasing your question or try again later.\n\n"
                f"Error details: {str(e)}"
            )
            return error_message, {"error": str(e)} 