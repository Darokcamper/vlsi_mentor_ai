from core.llm import llm
from core.rag_builder import retrieve

def ask_expert(
    question,
    role,
    expertise,
    history=None
):
    # If the question is a bloated prompt constructed by the graph, extract the original question for RAG retrieval
    import re
    retrieval_query = question
    if "Question:" in question:
        match = re.search(r"Question:\s*(.*?)(?:\n\nHere is|\n\nRevision Round|\n\n|$)", question, re.DOTALL)
        if match:
            retrieval_query = match.group(1).strip()

    # Context-Aware Query Rewriting: If the current question is short/vague and we have history,
    # blend it with the previous user question to preserve filenames and topic keywords for retrieval.
    if history:
        last_user_msg = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if last_user_msg:
            words = retrieval_query.strip().split()
            is_follow_up = len(words) < 6 or any(w in retrieval_query.lower() for w in ["what about", "why", "explain", "how", "it", "that", "they", "those", "this", "first", "second", "third", "prev", "last"])
            if is_follow_up:
                clean_last = last_user_msg.split("\n\n(Strict Grounding")[0].strip()
                retrieval_query = f"{retrieval_query} {clean_last}"

    # Detect if a specific source file is mentioned (e.g. Scan_DRC_part_2, Miscellaneous, etc.)
    source_filter = None
    try:
        import csv
        from pathlib import Path
        with open("knowledge/sources.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            file_stems = sorted([Path(row["File"]).stem for row in reader if row.get("File")], key=len, reverse=True)
        
        norm_question = retrieval_query.lower()
        for stem in file_stems:
            if stem.lower() in norm_question:
                source_filter = stem
                break
    except Exception as e:
        print(f"Error loading source file names: {e}")

    # Expand queries about DRCs, rules, or violations to automatically retrieve the specific rule notes from Miscellaneous.txt
    # Only expand if the user has NOT specified a target source document, to avoid poisoning the vector ranking.
    norm_query = retrieval_query.lower()
    if not source_filter and any(term in norm_query for term in ["drc", "design rule check", "rule", "violation"]):
        retrieval_query = f"{retrieval_query} K19 K22 F7 F10 Miscellaneous"

    # Try retrieving relevant knowledge chunks
    context = ""
    try:
        k = 8 if source_filter else 3
        results = retrieve(retrieval_query, top_k=k, source_filter=source_filter)
        if results:
            context_parts = []
            for r in results:
                context_parts.append(
                    f"Source Document: {r['source']} (Page {r['page']})\nContent: {r['text']}"
                )
            if context_parts:
                context = "\n\n---\n\n".join(context_parts)
    except Exception as e:
        print(f"RAG retrieval skipped or failed: {e}")

    # Build the system prompt
    system_prompt = f"""You are a {role}.

Expertise:
{expertise}

Rules:
- Strict Grounding: You must base your answer ONLY on the retrieved context from verified VLSI/DFT notes. Do NOT use your general pre-trained knowledge to invent or describe specific DRC rules, scripts, commands, or design steps if they are not present in the retrieved context.
- Vague Query Handling: If the user question is short or vague (e.g., "violations", "rules", "EDT"), or if the retrieved context does not contain the exact answer, you MUST state: "I do not have the exact answer in my verified source documents." and then list the source files retrieved in the context with their paths as clickable markdown links (e.g., `[filename.txt](file:///C:/vlsi-mentor-ai/knowledge/txt/filename.txt)`) and a 1-sentence summary of what topics that file covers on that page. Suggest how the user can refine their question to get a specific answer. Do NOT invent any rule lists, parameters, or details not present in the context.
- Stay within your expertise.
- If the question is outside your domain, clearly say so.
- Provide deep, concrete, production-ready engineering answers (suitable for tape-out reviews and senior DFT interviews) rather than high-level textbook summaries, strictly sourced from the retrieved context.
- Explicitly specify design parameters, rules, and hardware blocks where relevant (e.g., partition scan chains strictly by clock/power domains, define target scan chain depths like 500–2000 flops, use lockup latches for CDC shift timing, specify OCC broadside launch-on-capture pulses, require isolation cells and level shifters for power domains, detail decompressor/compactor/X-masking channels for EDT instead of generic compression, use IEEE 1500/1687 wrappers, plan JTAG daisy-chain/hierarchical access) strictly as verified in the retrieved notes.
- Prioritize current retrieved context over previous history: If the current retrieved context contains definitions, rules, or details that contradict or differ from previous model answers in the conversation history, you MUST strictly use the definitions/rules from the current retrieved context.
- Map numbers in follow-up questions to the actual slide/note numbering: If the user asks about section/rule numbers (e.g., "what about 3 and 4", "explain 5"), you MUST check the retrieved context for slide/section headers starting with those numbers (e.g., "# 3. FEEDBACK LOOP DRC", "# 4. X - SOURCE DRC"). Prioritize the main rule/slide headers starting with those numbers (e.g., "# 3. ..." or "# 4. ...") over any minor sub-component lists (like "3. NOR Gate" or "4. AND Gate" inside a gate schematic description) or list numbering from previous responses.
- Do not invent concepts.
- Be concise but complete and technically precise.
- Use previous conversation context when references like "it", "this", "that", "they", "those" are used.
"""

    if context:
        system_prompt += f"""

Here is relevant context retrieved from verified VLSI/DFT notes:
{context}

Guidelines for using the retrieved context:
1. Base your answer strictly on this context. Prioritize this context when answering, but ONLY if it is legible and technically correct.
2. If the context contains OCR corruption, typos, or garbled characters (e.g. "SOO ¥ 500", "pattewn", "Zou", "BOb", "VSING Formvta"), do NOT copy or reproduce the gibberish. You MUST clean it up and translate it into clear, proper technical English (e.g. translate "VSING Formvta" to "Using the formula").
3. If the math formulas in the context are broken or garbled by OCR (e.g. "T= / = SOO ¥ 500"), do NOT print the broken formula. Instead, write the standard correct DFT formula: "Test Time = Patterns * Shift Cycles * Clock Period" (where Shift Cycles / Pattern is equal to the length of the longest scan chain, which is the total number of flip-flops divided by the number of scan chains). If the question is about scan compression (EDT), explain that the decompressor expands a few external channels into many internal scan chains, making each internal chain significantly shorter, which reduces the shift cycles per pattern and thus reduces the test time by the compression factor.
4. Correct standard DFT acronyms if the context defines them incorrectly (e.g., EDT is Embedded Deterministic Test / Scan Compression, OCC is On-Chip Clock Controller, MBIST is Memory Built-In Self-Test).
5. Always cite the Source Document and Page when referencing ideas from the notes, but do not reproduce the OCR noise.
"""

    messages = [
        ("system", system_prompt)
    ]

    if history:
        # Limit history to last 3 messages to prevent prompt bloat and 413/TPM rate limits
        for msg in history[-3:]:
            messages.append(
                (
                    msg["role"],
                    msg["content"]
                )
            )

    # Append a strict grounding reminder directly to the user's question to guarantee instruction-following
    user_msg = question
    if not context:
        user_msg += "\n\n(Strict Grounding Reminder: The retrieved context is empty. You are forbidden from using general pre-trained knowledge to answer this. You MUST reply with exactly: 'I do not have this information in my verified source documents.')"
    else:
        user_msg += "\n\n(Strict Grounding Reminder: Base your answer strictly on the retrieved context above. If the context does not contain the specific rules, definitions, or commands requested, you are forbidden from using general pre-trained knowledge to invent them. In that case, you MUST state 'I do not have the exact answer in my verified source documents.' and list the source files as clickable markdown links `[filename.txt](file:///C:/vlsi-mentor-ai/knowledge/txt/filename.txt)` with a 1-sentence description of what topics they cover, suggesting query refinements.)"

    messages.append(
        (
            "user",
            user_msg
        )
    )

    response = llm.invoke(messages)
    response_content = response.content.strip()

    # Post-processing safeguard: if the LLM admitted it lacked verified context, discard any hallucinated lists and return only the fallback message
    normalized = response_content.lower()
    fallback_phrase = "do not have this information in my verified source documents"
    if fallback_phrase in normalized or "not found in the retrieved context" in normalized or "don't have this information" in normalized:
        # If the LLM successfully created clickable guide links for the user, allow the helpful response to pass through
        if "file://" not in response_content:
            return "I do not have this information in my verified source documents."

    return response_content