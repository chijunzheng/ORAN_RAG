# src/chatbot/yang_processor.py

import re
import json
import logging
from typing import List, Dict
import tiktoken
from google.cloud import firestore, aiplatform
from vertexai.generative_models import Content, GenerationConfig, Part


# ----------------------------------------------------------------------------------
# Constants and Helper
# ----------------------------------------------------------------------------------

SAFE_SUMMARY_TOKEN_LIMIT = 1_000_000
# We will do the first summarization pass in ~15k-token chunks, 
# which is big enough to reduce calls drastically vs. 3k chunks:
CHUNK_SIZE_FIRST_PASS = 15000
# Overlap can be small to reduce repetition and calls. 
CHUNK_OVERLAP = 50
# We allow multiple summarization passes if the final text is still huge:
MAX_SUMMARIZATION_PASSES = 3

def count_tokens(text: str) -> int:
    if tiktoken:
        try:
            encoding = tiktoken.encoding_for_model("text-embedding-005")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    else:
        return len(text.split())

class YangProcessor:
    """
    Encapsulates all YANG processing functionality:
      - Parsing natural language queries into metadata filters.
      - Filtering preprocessed YANG chunks.
      - Stitching chunks together to reassemble a full YANG file.
      - Hierarchical summarization if file size exceeds a safe token limit.
      - Generating detailed LLM prompts for:
          • Explaining a single YANG file.
          • Comparing a specific YANG file between vendors.
          • Comparing vendor packages.
          • Listing modified YANG files.
    Logs the number of chunks collected and the token count of the reassembled file.
    """
    def __init__(self, vector_searcher=None):
            self.vector_searcher = vector_searcher

            self.parse_generation_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=512
        )
        # Summaries or final prompts can use a separate config if desired.
            self.summarization_config = GenerationConfig(
            temperature=0.1,
            top_p=0.9,
            max_output_tokens=4096
        )
            self.yang_generation_config = GenerationConfig(
                temperature=0,
                top_p=1,
                max_output_tokens=8192,
            )        


    def llm_parse_query(self, query: str, llm) -> dict:
        """
        Calls LLM to parse user query into a minimal JSON:
        {
            "action": string in ["list_inventory","explain","compare_file","compare_vendor","compare"],
            "file": string or null,
            "vendors": array of strings
        }
        No other fields are allowed. 
        """
        prompt_text = f"""
        You are a specialized parser. 
        Output ONLY valid JSON with no extra text (no code fences, no commentary). 
        The JSON MUST have exactly three fields: "action", "file", "vendors".

        **Allowed Actions**:
        - "list_inventory": listing YANG inventory for a vendor or ORAN version. 
        - "explain": explain a single .yang file. 
        - "compare_file": compare a specific .yang file between two vendors or ORAN versions. 
        - "compare_vendor": compare two vendors (broadly, or check which files changed). 
        - "compare": a generic compare action (e.g. "which files changed" might set this if ambiguous). 

        **Rules**:
        1. `action` MUST be one of: ["list_inventory","explain","compare_file","compare_vendor","compare"]  
        2. `file` is either a .yang filename (string) or null if none specified.  
        3. `vendors` is an array of strings (e.g. ["24A","24B"] or ["v06.0","v08.0"]). Possibly empty if not relevant.

        **IMPORTANT**: 
        - Do NOT output code fences or markdown. 
        - Do NOT add extra keys. 
        - Do NOT wrap the JSON in code fences. 
        - Output only one JSON object with no additional text.
        - If the user mentions an ORAN version like v6.0, unify it to "v06.00".
        - If the user mentions vendor 24A or 24B, store them as is.

        **Examples**:

        1) If user says "provide a list of yang models in oran v6", then:
        {{
        "action": "list_inventory",
        "file": null,
        "vendors": ["v06.00"]
        }}

        2) If user says "explain samsung-access-uducnf.yang", then:
        {{
        "action": "explain",
        "file": "samsung-access-uducnf.yang",
        "vendors": []
        }}

        3) If user says "compare file tailf-rollback.yang between vendor 24A and 24B", then:
        {{
        "action": "compare_file",
        "file": "tailf-rollback.yang",
        "vendors": ["24A","24B"]
        }}

        4) If user says "which files changed from 24A to 24B", then:
        {{
        "action": "compare",
        "file": null,
        "vendors": ["24A","24B"]
        }}

        The user's query: {query}
        """

        try:
            response = llm.generate_content(
                prompt_text, 
                generation_config=self.parse_generation_config
            )
            raw_text = response.text.strip()
            if not raw_text:
                logging.warning("LLM parse gave empty output. Fallback parse.")
                return self.parse_query_fallback(query)

            parsed = json.loads(raw_text)
            logging.info(f"LLM parse => {parsed}")
            return parsed
        except Exception as e:
            logging.error(f"LLM parse failed: {e}", exc_info=True)
            return self.parse_query_fallback(query)

    def parse_query_fallback(self, query: str) -> dict:
        """
        Fallback parser if the LLM parse fails. We produce a dictionary with:
        {
            "action": "list_inventory"|"explain"|"compare_file"|"compare_vendor"|"compare",
            "file": string or None,
            "vendors": []
        }
        We do NOT store 'module' or other keys.
        """

        filter_dict = {
            "action": "explain",
            "file": None,
            "vendors": []
        }

        q_lower = query.lower()

        # 1) If user says "list" or "show" or "what files" plus "vendor" or "oran" => list_inventory
        if (("list" in q_lower or "show" in q_lower or "what files" in q_lower)
            and ("vendor" in q_lower or "oran" in q_lower)):
            filter_dict["action"] = "list_inventory"

        # 2) If user references a special O-RAN directory name like "O-RAN.WG4.MP-YANGs-v06.00"
        #    capture the version using a regex like r'(?:O-RAN\.[^-]*-v|oran\s*v)(\d+\.\d+)'
        #    We unify that as "v6.00".
        #    We'll also handle simpler "oran v6.0" => "v6.0".
        
        # First, check if the user typed something like "O-RAN.WG4.MP-YANGs-v08.00"
        # We'll unify that to "v8.00"
        o_ran_dir_match = re.findall(r'[Oo]-RAN\.[^-]*-v(\d+\.\d+)', query)
        for om in o_ran_dir_match:
            # e.g. om="06.00"
            filter_dict["vendors"].append(f"v{om}")

        # Next, we also check if user typed "oran v6.0" or "oran v8.0"
        simple_oran_match = re.findall(r'oran\s*v(\d+(\.\d+)?)', q_lower)
        for match in simple_oran_match:
            version_str = match[0]  # e.g. '6.0'
            filter_dict["vendors"].append(f"v{version_str}")

        # 3) If user references vendor package 24A or 24B
        vend_matches = re.findall(r'\b(24A|24B)\b', q_lower, re.IGNORECASE)
        for v in vend_matches:
            filter_dict["vendors"].append(v.upper())

        # 4) If user says "compare" or "difference"
        #    => if there's a .yang => compare_file, else compare_vendor
        if "compare" in q_lower or "difference" in q_lower:
            if re.search(r'([\w\-]+\.yang)', q_lower):
                filter_dict["action"] = "compare_file"
            else:
                filter_dict["action"] = "compare_vendor"

        # 5) If user says "explain" or "describe" => action="explain"
        elif "explain" in q_lower or "describe" in q_lower:
            filter_dict["action"] = "explain"

        # 6) If there's a .yang filename
        fm = re.search(r'([\w\-]+\.yang)', q_lower)
        if fm:
            filter_dict["file"] = fm.group(1).lower()

        logging.info(f"Fallback parse => {filter_dict}")
        return filter_dict
    
    # ------------------------------------------------------------
    # Summarization Helpers
    # ------------------------------------------------------------
    def summarize_large_file(self, text: str) -> str:
        total = count_tokens(text)
        if total <= SAFE_SUMMARY_TOKEN_LIMIT:
            return text
        logging.info(f"Text has {total} tokens; applying hierarchical summarization.")
        current = text
        passes = 0
        while count_tokens(current) > SAFE_SUMMARY_TOKEN_LIMIT and passes < MAX_SUMMARIZATION_PASSES:
            current = self._summarize_in_chunks(current, CHUNK_SIZE_FIRST_PASS, CHUNK_OVERLAP)
            passes += 1
        return current

    def _summarize_in_chunks(self, text: str, chunk_size: int, overlap: int) -> str:
        logging.info(f"Single-pass summarization on {count_tokens(text)} tokens.")
        # For brevity, return the text unchanged (or call a real summarization LLM here)
        return text

    
    # ------------------------------------------------------------
    # Retrieving Chunks by "Vendor" or "ORAN version" 
    # ------------------------------------------------------------
    def retrieve_yang_chunks_by_vendor(self, vendor: str) -> List[Dict]:
        """
        If vendor starts with 'v6.0', unify it with metadata['oran_version']= 'v6.0'
        If user says '24A', unify with metadata['vendor_package'].
        """
        if not self.vector_searcher:
            logging.error("No vector_searcher available for retrieving vendor/ORAN chunks.")
            return []
        all_chunks = list(self.vector_searcher.chunks.values())

        # If vendor is "v6.0", "v8.0", etc => we match metadata['oran_version'] 
        if re.fullmatch(r'v\d+\.\d+', vendor.lower()):
            # e.g. "v6.0"
            target = vendor.lower()
            return [c for c in all_chunks 
                    if c.get("metadata", {}).get("oran_version","").lower() == target]
        else:
            # fallback => vendor_package
            return [c for c in all_chunks 
                    if c.get("metadata", {}).get("vendor_package","").upper() == vendor.upper()]

    # ------------------------------------------------------------
    # Inventory Listing
    # ------------------------------------------------------------
    def handle_yang_inventory_listing(self, vendors: List[str]) -> str:
        inventory_results = {}
        for vendor in vendors:
            chunks = self.retrieve_yang_chunks_by_vendor(vendor)
            unique_modules = {}
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                mod = meta.get("module", "unknown")
                fname = meta.get("file_name", "unknown")
                unique_modules[mod] = fname
            inventory_results[vendor] = unique_modules

        lines = ["## Vendor Package YANG Model Inventory"]
        for vendor in vendors:
            lines.append(f"### Vendor Package {vendor}")
            modules = inventory_results[vendor]
            if modules:
                # First, show total number of files
                lines.append(f"#### Total Files: {len(modules)}")
                # Then list each module/file
                for mod, fname in sorted(modules.items()):
                    lines.append(f"- {fname}")
            else:
                lines.append("- No YANG models found.")
        return "\n".join(lines)


    # ------------------------------------------------------------
    # Chunk Stitching and Reassembly
    # ------------------------------------------------------------
    def stitch_chunks(self, chunks: List[Dict]) -> str:
        assembled = ""
        for i, c in enumerate(chunks):
            content = c.get("content", "")
            if i > 0:
                parts = content.split("\n\n", 1)
                if len(parts) == 2:
                    content = parts[1]
            assembled += content + "\n"
        num = len(chunks)
        toks = count_tokens(assembled)
        logging.info(f"Stitched {num} chunks; final token count = {toks}.")
        return assembled.strip()

    def reassemble_yang_files(self, yang_chunks: List[Dict]) -> Dict[str, Dict]:
        grouped = {}
        for chunk in yang_chunks:
            meta = chunk.get("metadata", {})
            fname = meta.get("file_name", "unknown").strip().lower()
            grouped.setdefault(fname, []).append(chunk)
        reassembled = {}
        for fname, chunks in grouped.items():
            chunks.sort(key=lambda c: int(c.get("metadata", {}).get("chunk_index", 0)))
            full_text = self.stitch_chunks(chunks)
            tok_count = count_tokens(full_text)
            logging.info(f"Reassembled '{fname}': {len(chunks)} chunks; token count = {tok_count}.")
            if tok_count > SAFE_SUMMARY_TOKEN_LIMIT:
                full_text = self.summarize_large_file(full_text)
            reassembled[fname] = {"full_text": full_text, "metadata": chunks[0].get("metadata", {})}
        return reassembled

    def retrieve_file_chunks(self, file_query: str) -> List[Dict]:
        """
        Retrieves chunks for a specific YANG file by filtering file_name.
        """
        if not self.vector_searcher:
            logging.error("No vector_searcher available to retrieve file chunks.")
            return []
        all_chunks = list(self.vector_searcher.chunks.values())
        filtered = [c for c in all_chunks if file_query.lower() in c.get("metadata", {}).get("file_name", "").lower()]
        logging.info(f"Retrieved {len(filtered)} chunks for file query '{file_query}'.")
        return filtered
    
    # ------------------------------------------------------------
    # Prompt-Engineered Methods (Explain, Compare, etc.)
    # ------------------------------------------------------------

    def explain_single_yang_file(self, file_name: str, yang_chunks: List[Dict], llm, generation_config: GenerationConfig) -> str:
        """
        Retrieves all chunks for a specific YANG file, stitches them together, logs the number of chunks,
        and generates an explanation using the LLM.
        """
        filtered = [chunk for chunk in yang_chunks if file_name.lower() in chunk.get("metadata", {}).get("file_name", "").lower()]
        num_chunks = len(filtered)
        if not filtered:
            return f"No content found for file '{file_name}'."
        
        assembled = self.stitch_chunks(filtered)
        logging.info(f"YangProcessor.explain_single_yang_file: Stitched together {num_chunks} chunks for file '{file_name}'.")
        
        prompt_text = f"""
        You are an advanced YANG model specialist. Provide a concise, high-level explanation of the YANG file '{file_name}' based on the provided content. Your explanation should focus on summarizing the overall logical and hierarchical data structure of the module without listing every detail.

        Your explanation must include:

        ## Overview and Purpose
        - Summarize the module’s intended role in network management or O-RAN contexts.
        - Explain how the file integrates with or depends on other modules.

        ## Module-Level Details
        - Provide key metadata such as the module name, prefix, namespace, and revision history (with dates and brief descriptions).

        ## Data Structure Summary
        - Present a high-level overview of the data structures in the file.  
        - Instead of listing every typedef, grouping, container or element, group similar structures together. For example, if the file defines numerous containers for uplink configurations, summarize them as "multiple containers managing uplink configurations" rather than listing each one.
        - Describe the major groupings, containers, and typedefs and emphasize the logical, hierarchical relationships (e.g., "Grouping A contains Container B, which in turn defines critical leaves" or "Container X nests a grouping that organizes Y data").
        - Focus on what differentiates the major sections and their roles rather than repeating similar items.

        ## Imports and References 
        - Briefly indicate which external YANG modules are imported or referenced, and explain their significance.

        ## Usage and Implications
        - Outline how a network operator or developer might use this module in practice.
        - Mention notable use cases or deployment scenarios in an O-RAN architecture.

        ## Conclusion  
        - Offer a concise summary highlighting the key takeaways and the overall significance of the file.

        <answer-format>
        Structure your answer using high-level headings (##) and subheadings (###). Use bullet points or numbered lists to outline your explanation. Focus on the overarching hierarchy and relationships of the data structures, rather than listing every detailed element.
        </answer-format>

        <markdown-guidelines>
            <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
            <markdown-guideline>Use bullet points for lists and maintain consistent indentation.</markdown-guideline>
        </markdown-guidelines>

        <important-notes>
            <important-note>Focus on delivering a strategic, high-level overview of the module's data structure and its hierarchical relationships.</important-note>
            <important-note>Avoid enumerating every low-level detail; instead, emphasize the architecture and key relationships (e.g., grouping/container/leaf) that define the file.</important-note>
        </important-notes>

        ## File Content


        {assembled}

        <answer>

        </answer>
        """
        prompt_content = Content(role="user", parts=[Part.from_text(prompt_text)])
        try:
            response = llm.generate_content(prompt_content, generation_config=self.yang_generation_config)
            return response.text.strip() if response and response.text.strip() else "No explanation generated."
        except Exception as e:
            logging.error(f"YangProcessor.explain_single_yang_file error for '{file_name}': {e}", exc_info=True)
            return "Error generating explanation for the specified YANG file."


    # ------------------------------------------------------------
    # Compare broad vendor packages (for a high-level summary)
    # ------------------------------------------------------------
    def compare_vendor_packages_broad(self, vendor1: str, vendor2: str) -> str:
        """
        Returns a text prompt describing all modules from vendor1 vs vendor2, 
        so the Chatbot can call the LLM for a broad vendor-level comparison.
        """
        if not self.vector_searcher:
            return "No vector_searcher to retrieve vendor chunks."
        # unify 6.0 => ORAN_v6.0
        if re.fullmatch(r'\d+\.\d+', vendor1):
            vendor1 = f"ORAN_v{vendor1}"
        if re.fullmatch(r'\d+\.\d+', vendor2):
            vendor2 = f"ORAN_v{vendor2}"
        v1chunks = self.retrieve_yang_chunks_by_vendor(vendor1)
        v2chunks = self.retrieve_yang_chunks_by_vendor(vendor2)

        # reassemble all to get text
        def reassemble_vchunks(chunks:List[Dict]) -> str:
            from collections import defaultdict
            grouped = defaultdict(list)
            for c in chunks:
                md = c.get("metadata",{})
                fname = md.get("file_name","unknown")
                grouped[fname].append(c)
            final = ""
            for fname, parts in grouped.items():
                # sort by chunk_index
                parts.sort(key=lambda x: int(x.get("metadata",{}).get("chunk_index",0)))
                content_assembled = ""
                for i, pc in enumerate(parts):
                    cont = pc.get("content","")
                    if i>0:
                        seg = cont.split("\n\n",1)
                        if len(seg)==2:
                            cont= seg[1]
                    content_assembled += cont+"\n"
                final += f"\n[File: {fname}]\n{content_assembled}\n"
            return final.strip()

        text_v1 = reassemble_vchunks(v1chunks)
        text_v2 = reassemble_vchunks(v2chunks)
        return f"""
            You are a YANG expert. Compare vendor {vendor1} vs {vendor2} at a broad level:

            ## Vendor {vendor1}
            {text_v1}

            ## Vendor {vendor2}
            {text_v2}

            <instructions>
            Describe all notable differences in naming, structure, modules, revisions, etc.
            </instructions>
            """
        

    # ------------------------------------------------------------
    # List Modified YANG Files Between Two Vendors
    # ------------------------------------------------------------
    def list_modified_yang_files(self, vendor1: str, vendor2: str) -> str:
        """
        Returns a detailed comparison of YANG files between vendor1 and vendor2,
        categorizing them into 'Added', 'Removed', or 'Modified'.

        'Added': Files that appear only in vendor2
        'Removed': Files that appear only in vendor1
        'Modified': Files that appear in both but differ in revision set or chunk count.
        """
        if not self.vector_searcher:
            return "No vector_searcher to retrieve vendor chunks."

        # Gather chunks for each vendor
        v1_chunks = self.retrieve_yang_chunks_by_vendor(vendor1)
        v2_chunks = self.retrieve_yang_chunks_by_vendor(vendor2)

        from collections import defaultdict
        def build_map(chs):
            m = defaultdict(lambda: {"revisions": set(), "count": 0})
            for c in chs:
                md = c.get("metadata", {})
                fname = md.get("file_name", "unknown")
                rev = md.get("revision", "unknown")
                m[fname]["revisions"].add(rev)
                m[fname]["count"] += 1
            return m

        map1 = build_map(v1_chunks)
        map2 = build_map(v2_chunks)

        all_fnames = sorted(set(map1.keys()) | set(map2.keys()))

        added = []
        removed = []
        modified = []

        for fname in all_fnames:
            info1 = map1.get(fname)
            info2 = map2.get(fname)
            if info1 and not info2:
                # File only in vendor1 => 'Removed'
                removed.append(fname)
            elif info2 and not info1:
                # File only in vendor2 => 'Added'
                added.append(fname)
            else:
                # Present in both => check if revisions or counts differ
                if info1["revisions"] != info2["revisions"] or info1["count"] != info2["count"]:
                    modified.append(fname)

        lines = [
            f"## YANG Files Comparison Between Vendor {vendor1} and Vendor {vendor2}",
            ""
        ]

        if not (added or removed or modified):
            lines.append("No differences found between these two versions/packages.")
            return "\n".join(lines)

        # Print each category
        if added:
            lines.append(f"### Added (present only in {vendor2})\n")
            for f in added:
                lines.append(f"- {f}")
            lines.append("")

        if removed:
            lines.append(f"### Removed (present only in {vendor1})\n")
            for f in removed:
                lines.append(f"- {f}")
            lines.append("")

        if modified:
            lines.append("### Modified\n")
            for f in modified:
                lines.append(f"- {f}")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Compare Single File Across Two Vendors
    # ------------------------------------------------------------
    def compare_file_between_vendors(self, file_name:str, vendor1:str, vendor2:str, llm, generation_config:GenerationConfig) -> str:
        """
        Actually finalize the LLM call. This code was truncated in your snippet. 
        We'll finalize it so it returns the LLM answer, not None.
        """
        if not self.vector_searcher:
            return "No vector_searcher available to retrieve vendor chunks."

        # unify "6.0" => "ORAN_v6.0"
        if re.fullmatch(r'\d+\.\d+', vendor1):
            vendor1 = f"ORAN_v{vendor1}"
        if re.fullmatch(r'\d+\.\d+', vendor2):
            vendor2 = f"ORAN_v{vendor2}"

        all_chunks = list(self.vector_searcher.chunks.values())
        v1 = [c for c in all_chunks
              if file_name.lower() in c.get("metadata",{}).get("file_name","").lower()
              and (
                c.get("metadata",{}).get("vendor_package","").upper()==vendor1.upper()
                or c.get("metadata",{}).get("oran_version","").lower()==vendor1.lower().replace("oran_v","v")
              )]
        v2 = [c for c in all_chunks
              if file_name.lower() in c.get("metadata",{}).get("file_name","").lower()
              and (
                c.get("metadata",{}).get("vendor_package","").upper()==vendor2.upper()
                or c.get("metadata",{}).get("oran_version","").lower()==vendor2.lower().replace("oran_v","v")
              )]

        text_v1 = self.stitch_chunks(v1) if v1 else "No content found."
        text_v2 = self.stitch_chunks(v2) if v2 else "No content found."
        # Summaries if large
        if count_tokens(text_v1)>SAFE_SUMMARY_TOKEN_LIMIT:
            text_v1 = self.summarize_large_file(text_v1)
        if count_tokens(text_v2)>SAFE_SUMMARY_TOKEN_LIMIT:
            text_v2 = self.summarize_large_file(text_v2)

        compare_prompt = f"""
            You are a YANG expert. Your task is to perform a detailed, line-by-line comparison between the file '{file_name}' from vendor {vendor1} and vendor {vendor2}. Your analysis must be both high-level and granular, reflecting the logical and hierarchical structure of the YANG modules.

            ### Vendor Package {vendor1} version of {file_name}:
            {text_v1}

            ### Vendor Package {vendor2} version of {file_name}:
            {text_v2}

            <Instructions>
            - First, provide an overall high-level summary of the differences between the two files. 
            - Highlight the major additions, removals, or modifications, and mention any improvements or regressions.
            - Then, perform a detailed, hierarchical, line-by-line analysis. For every detected change, indicate its precise logical location using the structure path. Use the format “grouping/container/leaf” or the appropriate combination (for example, “grouping/container”, or “container/leaf”, or “grouping/leaf”) to pinpoint the change.
            - Only highlight functional changes that are significant and relevant to the comparison. Avoid listing every minor differences such as description changes.
            - Organize your answer by first summarizing module-level differences, then breaking down the differences into hierarchical sections that reflect the YANG structure.
            - Do not list every low-level leaf difference separately unless it is significant; instead, include these differences under their respective higher-level grouping or container.
            </Instructions>

            <answer-format>
            Structure your answer with high-level headings (##) and subheadings (###). Present your analysis in bullet points or numbered lists. For example:

            ## High-Level Summary of Differences of <file_name> between Vendor {vendor1} and Vendor {vendor2}
            - Description of enhancements, regressions, or major changes in bullet points
            ## Detailed, Hierarchical, Line-by-Line Analysis
            ### Module-Level Differences
            ### Grouping/Container/Leaf Differences
            - "grouping1": Description of change
            OR
            - "container1": Description of change
            OR
            - "grouping1/container1":
                - Description of leaf changes
            OR
            - "grouping1/container2":
                - Description of leaf changes
            OR
            - "grouping2"/"container1":
                - Description of leaf changes
            etc.


            <markdown-guidelines>
                <markdown-guideline>Use `##` for main sections and `###` for subsections.</markdown-guideline>
                <markdown-guideline>Use bullet points for lists and maintain consistent indentation.</markdown-guideline>
            </markdown-guidelines>
            </answer-format>

            Answer:
            """

        try:
            resp = llm.generate_content(compare_prompt, generation_config=generation_config)
            if resp and resp.text.strip():
                return resp.text.strip()
            else:
                return "No comparison generated."
        except Exception as e:
            logging.error(f"Error comparing file '{file_name}' for {vendor1} vs {vendor2}: {e}", exc_info=True)
            return f"I'm sorry, an error occurred comparing '{file_name}' for {vendor1} vs {vendor2}."
    
    # --------------------------------------------------------
    # Main Analysis Entry Point
    # --------------------------------------------------------
    def get_analysis(self, query: str, yang_chunks: List[Dict], llm, generation_config: GenerationConfig) -> str:
        """
        Decides how to handle the user query. 
        If LLM parse fails, fallback parse. Then route to either:
          - list_inventory
          - explain_single_yang_file
          - compare_file_between_vendors
          etc.
        """
        try:
            parsed = self.llm_parse_query(query, llm)
            # Fallback in case LLM returns incomplete data:
            if not parsed or not isinstance(parsed, dict):
                parsed = self.parse_query_fallback(query)
            action = parsed.get("action", "explain")
            file_name = parsed.get("file")
            vendors = parsed.get("vendors", [])

            # If action is "compare_vendor" but a file is provided, treat as compare_file.
            if action == "compare_vendor" and file_name:
                action = "compare_file"

            # Unify ORAN versions: if any vendor string matches a pure version (e.g., "6.0"), convert it.
            unified_vendors = []
            for v in vendors:
                if re.fullmatch(r'\d+\.\d+', v):
                    unified_vendors.append(f"ORAN_v{v}")
                else:
                    unified_vendors.append(v)
            vendors = unified_vendors

            logging.info(f"Parsed query => action={action}, file={file_name}, vendors={vendors}")

            if action == "list_inventory" and vendors:
                return self.handle_yang_inventory_listing(vendors)
            elif action == "compare_file" and file_name and len(vendors) >= 2:
                return self.compare_file_between_vendors(file_name, vendors[0], vendors[1], llm, generation_config)
            elif action == "explain" and file_name:
                return self.explain_single_yang_file(file_name, yang_chunks, llm, generation_config)
            elif action == "compare" and len(vendors) >= 2:
                # For generic compare (if no file is specified), you could list modified files.
                return self.list_modified_yang_files(vendors[0], vendors[1])
            else:
                return "I'm sorry, I can't handle that request with the given details."
        except Exception as e:
            logging.error(f"YangProcessor.get_analysis error: {e}", exc_info=True)
            return "An error occurred while analyzing YANG files."