from core.llm import llm

def route_question(question):

    prompt = f"""
Classify this VLSI question into exactly ONE category.

Categories:

SCAN
ATPG
EDT
MBIST
JTAG
IJTAG
WRAPPER
BOUNDARYSCAN
OCC
STA
GLS
TCL
LINUX
GENERAL

Question:
{question}

Return ONLY the category.
"""

    response = llm.invoke(prompt)

    return response.content.strip().upper()