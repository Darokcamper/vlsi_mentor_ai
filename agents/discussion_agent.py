from agents.scan_agent import ask_scan
from agents.atpg_agent import ask_atpg
from agents.sta_agent import ask_sta

from agents.critic_agent import critique
from agents.reviewer_agent import review


def run_discussion(question, history=None):

    discussion = ""

    # =====================================
    # ROUND 1
    # =====================================

    scan_answer = ask_scan(
        question,
        history
    )

    discussion += f"""
==============================
SCAN EXPERT
==============================

{scan_answer}

"""

    sta_answer = ask_sta(
        f"""
Question:
{question}

Scan Expert Response:

{scan_answer}

Review the answer strictly from
a timing perspective.

Add anything important that is
missing.
""",
        history
    )

    discussion += f"""
==============================
STA EXPERT
==============================

{sta_answer}

"""

    atpg_answer = ask_atpg(
        f"""
Question:
{question}

Scan Expert Response:

{scan_answer}

STA Expert Response:

{sta_answer}

Review from ATPG perspective.

Add anything important that is
missing.
""",
        history
    )

    discussion += f"""
==============================
ATPG EXPERT
==============================

{atpg_answer}

"""

    # =====================================
    # CRITIC ROUND
    # =====================================

    critic_feedback = critique(
        question,
        discussion
    )

    discussion += f"""
==============================
CRITIC
==============================

{critic_feedback}

"""

    # =====================================
    # ROUND 2
    # Experts revise answers
    # =====================================

    scan_revision = ask_scan(
        f"""
Question:
{question}

Discussion:

{discussion}

Critic Feedback:

{critic_feedback}

Revise your answer.

Focus only on scan related
corrections.
""",
        history
    )

    discussion += f"""
==============================
SCAN REVISION
==============================

{scan_revision}

"""

    sta_revision = ask_sta(
        f"""
Question:
{question}

Discussion:

{discussion}

Critic Feedback:

{critic_feedback}

Revise your answer.

Focus only on timing related
corrections.
""",
        history
    )

    discussion += f"""
==============================
STA REVISION
==============================

{sta_revision}

"""

    atpg_revision = ask_atpg(
        f"""
Question:
{question}

Discussion:

{discussion}

Critic Feedback:

{critic_feedback}

Revise your answer.

Focus only on ATPG related
corrections.
""",
        history
    )

    discussion += f"""
==============================
ATPG REVISION
==============================

{atpg_revision}

"""

    # =====================================
    # FINAL REVIEW
    # =====================================

    final_answer = review(
        question,
        discussion
    )

    return {
        "scan": scan_answer,
        "sta": sta_answer,
        "atpg": atpg_answer,
        "critic": critic_feedback,
        "discussion": discussion,
        "final": final_answer
    }