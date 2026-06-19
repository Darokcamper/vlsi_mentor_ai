from core.llm import llm

def ask_expert(
    question,
    role,
    expertise,
    history=None
):

    messages = []

    messages.append(
        (
            "system",
            f"""
You are a {role}.

Expertise:
{expertise}

Rules:

- Stay within your expertise.
- If the question is outside your domain,
  clearly say so.
- Give technically accurate VLSI answers.
- Do not invent concepts.
- Be concise but complete.
"""
        )
    )

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