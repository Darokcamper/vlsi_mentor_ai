import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from core.llm import llm
from agents.scan_agent import ask_scan
from agents.atpg_agent import ask_atpg
from agents.sta_agent import ask_sta
from agents.edt_agent import ask_edt
from agents.mbist_agent import ask_mbist
from agents.jtag_agent import ask_jtag
from agents.ijtag_agent import ask_ijtag
from agents.wrapper_agent import ask_wrapper
from agents.occ_agent import ask_occ
from agents.general_agent import ask_general
from agents.critic_agent import critique
from agents.reviewer_agent import review

# Agent mapping
AGENT_MAP = {
    "SCAN": ask_scan,
    "ATPG": ask_atpg,
    "STA": ask_sta,
    "EDT": ask_edt,
    "MBIST": ask_mbist,
    "JTAG": ask_jtag,
    "IJTAG": ask_ijtag,
    "WRAPPER": ask_wrapper,
    "OCC": ask_occ,
}

class DiscussionState(TypedDict):
    question: str
    participants: List[str]
    current_participant_idx: int
    discussion_history: List[Dict[str, str]]
    critic_feedback: str
    critic_score: int
    revision_round: int
    final_answer: str

def router_node(state: DiscussionState):
    """Determines which expert agents should participate based on the question."""
    question = state["question"]
    print(f"\n[Graph] Starting router_node for question: '{question[:60]}...'")
    
    prompt = f"""Given the following VLSI / DFT question, select the expert agents that are highly relevant to answer and discuss this question.
You MUST choose between 1 to 3 agents from this list:
- SCAN (Scan insertion, scan chains, lockup latches, stitching)
- ATPG (Pattern generation, fault models, stuck-at, transition)
- EDT (Scan compression, decompressor, compactor, MISR)
- MBIST (Memory BIST, SRAM, repair, redundancy, March algorithms)
- JTAG (Boundary scan, TAP controller, IEEE 1149.1)
- IJTAG (IEEE 1687, instrument connectivity, SIB)
- WRAPPER (IEEE 1500, wrapper chains, core isolation)
- OCC (On-chip clock controller, at-speed testing, capture/shift clocks)
- STA (Static timing analysis, setup/hold, constraints, clock skew)

Question: {question}

Return ONLY a comma-separated list of the selected agent names in uppercase, e.g. SCAN,STA or ATPG,EDT,SCAN. Do not add any other text."""

    try:
        response = llm.invoke(prompt)
        participants = [p.strip().upper() for p in response.content.split(",") if p.strip()]
        # Filter to valid ones
        participants = [p for p in participants if p in AGENT_MAP]
    except Exception as e:
        print(f"[Graph] Router LLM call failed: {e}")
        participants = []
        
    if not participants:
        # Fallback to SCAN and STA
        participants = ["SCAN", "STA"]
        
    print(f"[Graph] Selected participants: {participants}")
    return {
        "participants": participants,
        "current_participant_idx": 0,
        "discussion_history": [],
        "revision_round": 0,
        "critic_feedback": "",
        "critic_score": 0,
        "final_answer": ""
    }

def expert_node(state: DiscussionState):
    """Executes the turn of the current expert agent, supporting initial draft and revision."""
    idx = state["current_participant_idx"]
    participants = state["participants"]
    
    if idx >= len(participants):
        return {}  # No-op safety
        
    agent_name = participants[idx]
    print(f"\n[Graph] Running expert_node for agent: '{agent_name}' (Turn {idx+1}/{len(participants)}, Revision Round {state['revision_round']})")
    agent_func = AGENT_MAP.get(agent_name, ask_general)
    
    # Build discussion context
    history_lines = []
    for item in state["discussion_history"]:
        history_lines.append(f"{item['agent']}: {item['content']}")
    history_text = "\n\n".join(history_lines)
    
    if state["revision_round"] == 0:
        if history_text:
            prompt = f"""Question: {state['question']}

Here is what has been discussed so far:
{history_text}

Please add your expert perspective on this question as the {agent_name} expert, responding to previous points if necessary."""
        else:
            prompt = state["question"]
    else:
        # Revision round
        prompt = f"""Question: {state['question']}

Here is the previous discussion:
{history_text}

Here is the feedback and corrections from the Lead DFT Architect (Critic):
{state['critic_feedback']}

Please revise your previous response as the {agent_name} expert to address the critic's points. Focus only on corrections related to your domain and keep it technically precise."""

    try:
        response_content = agent_func(prompt)
    except Exception as e:
        print(f"[Graph] Expert '{agent_name}' function failed: {e}")
        response_content = f"Error generating response: {e}"
        
    new_history = list(state["discussion_history"])
    # If in revision round, we replace the previous entry by this expert or append it
    replaced = False
    if state["revision_round"] > 0:
        for i, item in enumerate(new_history):
            if item["agent"] == agent_name:
                new_history[i] = {"agent": agent_name, "content": response_content}
                replaced = True
                break
                
    if not replaced:
        new_history.append({"agent": agent_name, "content": response_content})
        
    print(f"[Graph] Completed expert_node for agent: '{agent_name}'")
    return {
        "discussion_history": new_history,
        "current_participant_idx": idx + 1
    }

def critic_node(state: DiscussionState):
    """Principal DFT Architect critiques the discussion and assigns a score."""
    question = state["question"]
    print(f"\n[Graph] Running critic_node...")
    
    history_lines = []
    for item in state["discussion_history"]:
        history_lines.append(f"{item['agent']}: {item['content']}")
    discussion_text = "\n\n".join(history_lines)
    
    # Run critique
    try:
        feedback = critique(question, discussion_text)
    except Exception as e:
        print(f"[Graph] Critic critique failed: {e}")
        feedback = f"Critique error: {e}"
        
    # Rate the discussion score
    score_prompt = f"""Analyze the following technical discussion and rate its accuracy and completeness for the question: "{question}"

Discussion:
{discussion_text}

Critic Feedback:
{feedback}

Rate the discussion from 1 to 10 (10 being perfect, 1 being completely incorrect/hallucinated). Return ONLY the integer score, e.g. 7. Do not include any other text."""

    score = 7
    try:
        score_response = llm.invoke(score_prompt)
        match = re.search(r'\d+', score_response.content)
        if match:
            score = int(match.group())
    except Exception as e:
        print(f"[Graph] Critic score LLM call failed: {e}")
        pass
        
    print(f"[Graph] Critic completed. Score: {score}/10")
    return {
        "critic_feedback": feedback,
        "critic_score": score
    }

def reviewer_node(state: DiscussionState):
    """Lead DFT architect compiles the final answer."""
    question = state["question"]
    print(f"\n[Graph] Running reviewer_node...")
    
    history_lines = []
    for item in state["discussion_history"]:
        history_lines.append(f"{item['agent']}: {item['content']}")
    discussion_text = "\n\n".join(history_lines)
    
    try:
        final_ans = review(question, discussion_text)
    except Exception as e:
        print(f"[Graph] Reviewer review failed: {e}")
        final_ans = f"Review error: {e}"
        
    print(f"[Graph] Reviewer completed compilation.")
    return {
        "final_answer": final_ans
    }

# Conditional Edges
def should_continue_discussion(state: DiscussionState):
    if state["current_participant_idx"] < len(state["participants"]):
        return "expert"
    return "critic"

def should_revise_discussion(state: DiscussionState):
    if state["critic_score"] < 8 and state["revision_round"] < 1:
        return "reset_revision"
    return "reviewer"

# Set up graph builder
builder = StateGraph(DiscussionState)

builder.add_node("router", router_node)
builder.add_node("expert", expert_node)
builder.add_node("critic", critic_node)
builder.add_node("reviewer", reviewer_node)

# State transition after critic: loop back to expert (for revision) or go to reviewer
def reset_revision_node(state: DiscussionState):
    next_round = state["revision_round"] + 1
    print(f"\n[Graph] Critic score < 8. Resetting participant index for revision round {next_round}...")
    return {
        "current_participant_idx": 0,
        "revision_round": next_round
    }

builder.add_node("reset_revision", reset_revision_node)
builder.add_edge("reset_revision", "expert")

builder.set_entry_point("router")

# Router goes to expert
builder.add_edge("router", "expert")

# Expert conditional routing (loops back to expert or goes to critic)
builder.add_conditional_edges(
    "expert",
    should_continue_discussion,
    {
        "expert": "expert",
        "critic": "critic"
    }
)

builder.add_conditional_edges(
    "critic",
    should_revise_discussion,
    {
        "reset_revision": "reset_revision",
        "reviewer": "reviewer"
    }
)

builder.add_edge("reviewer", END)

graph = builder.compile()

def run_graph_stream(question):
    """Invokes the graph and yields progress updates for real-time streaming in Streamlit."""
    inputs = {"question": question}
    
    yield "Initializing agent graph and routing question...", None
    
    state = {}
    for event in graph.stream(inputs, stream_mode="updates"):
        for node_name, updates in event.items():
            if node_name == "router":
                participants = updates.get("participants", [])
                yield f"Selected active participants: {', '.join(participants)}", None
            elif node_name == "expert":
                history = updates.get("discussion_history", [])
                if history:
                    last_item = history[-1]
                    agent = last_item["agent"]
                    yield f"Completed expert response for agent: **{agent}**", None
            elif node_name == "reset_revision":
                rev_round = updates.get("revision_round", 1)
                yield f"Critic score < 8. Initiating revision round {rev_round}...", None
            elif node_name == "critic":
                score = updates.get("critic_score", 0)
                yield f"Critic finished evaluation. Score: **{score}/10**", None
            elif node_name == "reviewer":
                yield "Lead Architect compiled the final answer.", None
                
            # Accumulate state updates
            for k, v in updates.items():
                state[k] = v
                
    yield "Graph workflow execution completed!", state

def run_graph(question):
    """Fallback invoke wrapper that consumes the stream to return the final state."""
    final_result = {}
    for msg, state in run_graph_stream(question):
        if state is not None:
            final_result = state
    return final_result
