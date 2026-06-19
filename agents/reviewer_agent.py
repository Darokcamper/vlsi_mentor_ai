from core.llm import llm

def review(question, discussion):

    prompt = f"""
You are the lead DFT architect.

Question:
{question}

Discussion:
{discussion}

Create a final answer.

Requirements:

- Technically correct
- Remove duplicate information
- Include important concepts
- Suitable for interview preparation

Return only final answer.
"""

    response = llm.invoke(prompt)

    return response.content