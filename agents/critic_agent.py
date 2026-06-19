from core.llm import llm

def critique(question, discussion):

    prompt = f"""
Question:
{question}

Discussion:
{discussion}

You are a Principal DFT Architect.

Tasks:

1. Identify incorrect statements.
2. Identify missing concepts.
3. Identify contradictions.
4. Suggest improvements.

Return:

Incorrect Points
Missing Concepts
Improvement Suggestions
"""

    response = llm.invoke(prompt)

    return response.content