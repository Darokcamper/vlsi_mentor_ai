from core.llm import llm
from core.rag_builder import retrieve

def ask_expert(
    question,
    role,
    expertise,
    history=None
):
    # Try retrieving relevant knowledge chunks
    context = ""
    try:
        results = retrieve(question, top_k=4)
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
- Stay within your expertise.
- If the question is outside your domain, clearly say so.
- Give technically accurate VLSI answers.
- Do not invent concepts.
- Be concise but complete.
- Use previous conversation context when references like "it", "this", "that", "they", "those" are used.
"""

    if context:
        system_prompt += f"""

Here is relevant context retrieved from verified VLSI/DFT notes:
{context}

Guidelines for using the retrieved context:
1. Prioritize this context when answering, but ONLY if it is legible and technically correct.
2. If the context contains OCR corruption, typos, or garbled characters (e.g. "SOO ¥ 500", "pattewn", "Zou", "BOb", "VSING Formvta"), do NOT copy or reproduce the gibberish. You MUST clean it up and translate it into clear, proper technical English (e.g. translate "VSING Formvta" to "Using the formula").
3. If the math formulas in the context are broken or garbled by OCR (e.g. "T= / = SOO ¥ 500"), do NOT print the broken formula. Instead, write the standard correct DFT formula: "Test Time = Patterns * Shift Cycles * Clock Period" (where Shift Cycles / Pattern is equal to the length of the longest scan chain, which is the total number of flip-flops divided by the number of scan chains). If the question is about scan compression (EDT), explain that the decompressor expands a few external channels into many internal scan chains, making each internal chain significantly shorter, which reduces the shift cycles per pattern and thus reduces the test time by the compression factor.
4. Correct standard DFT acronyms if the context defines them incorrectly (e.g., EDT is Embedded Deterministic Test / Scan Compression, OCC is On-Chip Clock Controller, MBIST is Memory Built-In Self-Test).
5. Always cite the Source Document and Page when referencing ideas from the notes, but do not reproduce the OCR noise.
"""

    messages = [
        ("system", system_prompt)
    ]

    if history:
        for msg in history:
            messages.append(
                (
                    msg["role"],
                    msg["content"]
                )
            )

    messages.append(
        (
            "user",
            question
        )
    )

    response = llm.invoke(messages)
    return response.content