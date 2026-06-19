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

Use previous conversation context whenever
the user refers to:

it
this
that
they
those

Answer clearly and accurately.
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