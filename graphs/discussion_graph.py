from langgraph.graph import StateGraph
from langgraph.graph import END

from typing import TypedDict

from agents.scan_agent import ask_scan
from agents.atpg_agent import ask_atpg
from agents.sta_agent import ask_sta

from agents.critic_agent import critique
from agents.reviewer_agent import review

class DiscussionState(TypedDict):

    question: str

    participants: list

    scan_answer: str
    sta_answer: str
    atpg_answer: str

    critic_feedback: str

    final_answer: str
    
def scan_node(state):

    answer = ask_scan(
        state["question"]
    )

    return {
        "scan_answer": answer
    }
    
def sta_node(state):

    answer = ask_sta(
        f"""
Question:
{state['question']}

Scan Answer:
{state['scan_answer']}

Review from timing perspective.
"""
    )

    return {
        "sta_answer": answer
    }
    
def atpg_node(state):

    answer = ask_atpg(
        f"""
Question:
{state['question']}

Scan:
{state['scan_answer']}

STA:
{state['sta_answer']}

Review from ATPG perspective.
"""
    )

    return {
        "atpg_answer": answer
    }
    
def critic_node(state):

    discussion = f"""

SCAN:
{state['scan_answer']}

STA:
{state['sta_answer']}

ATPG:
{state['atpg_answer']}
"""

    feedback = critique(
        state["question"],
        discussion
    )

    return {
        "critic_feedback": feedback
    }
    
def reviewer_node(state):

    discussion = f"""

SCAN:
{state['scan_answer']}

STA:
{state['sta_answer']}

ATPG:
{state['atpg_answer']}

CRITIC:
{state['critic_feedback']}
"""

    final = review(
        state["question"],
        discussion
    )

    return {
        "final_answer": final
    }
    
builder = StateGraph(
    DiscussionState
)

builder.add_node(
    "scan",
    scan_node
)

builder.add_node(
    "sta",
    sta_node
)

builder.add_node(
    "atpg",
    atpg_node
)

builder.add_node(
    "critic",
    critic_node
)

builder.add_node(
    "reviewer",
    reviewer_node
)

builder.set_entry_point(
    "scan"
)

builder.add_edge(
    "scan",
    "sta"
)

builder.add_edge(
    "sta",
    "atpg"
)

builder.add_edge(
    "atpg",
    "critic"
)

builder.add_edge(
    "critic",
    "reviewer"
)

builder.add_edge(
    "reviewer",
    END
)

graph = builder.compile()

def run_graph(question):

    result = graph.invoke(
        {
            "question": question
        }
    )

    return result

