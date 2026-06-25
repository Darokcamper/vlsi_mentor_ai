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
        results = retrieve(question, top_k=2)
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

You MUST prioritize this context to answer the question. If the context contains the answer, use it. If the context is not sufficient, you can use your general knowledge, but clearly state what is from the notes and what is general knowledge. Always cite the Source Document and Page (e.g. [3. Level 1 session 3, Page 5]) when referencing information from the notes. Do not make up citations.
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