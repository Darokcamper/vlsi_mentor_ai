import streamlit as st
import re

from agents.router_agent import route_question
from agents.scan_agent import ask_scan
from agents.atpg_agent import ask_atpg
from agents.edt_agent import ask_edt
from agents.mbist_agent import ask_mbist
from agents.jtag_agent import ask_jtag
from agents.ijtag_agent import ask_ijtag
from agents.wrapper_agent import ask_wrapper
from agents.occ_agent import ask_occ
from agents.sta_agent import ask_sta
from agents.boundaryscan_agent import ask_boundaryscan
from agents.gls_agent import ask_gls
from agents.linux_agent import ask_linux
from agents.tcl_agent import ask_tcl
from agents.general_agent import ask_general

from agents.interviewer_agent import ask_interview_question
from agents.evaluator_agent import evaluate_answer
from agents.planner_agent import generate_study_plan
from graphs.discussion_graph import run_graph

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="VLSI Mentor AI", page_icon="🧠", layout="wide")

# =====================================
# SESSION STATE
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# =====================================
# AGENT MAP
# =====================================
agent_map = {
    "SCAN": ask_scan,
    "ATPG": ask_atpg,
    "EDT": ask_edt,
    "MBIST": ask_mbist,
    "JTAG": ask_jtag,
    "IJTAG": ask_ijtag,
    "WRAPPER": ask_wrapper,
    "OCC": ask_occ,
    "STA": ask_sta,
    "BOUNDARYSCAN": ask_boundaryscan,
    "GLS": ask_gls,
    "LINUX": ask_linux,
    "TCL": ask_tcl,
}

# Helper to extract score from evaluation text
def extract_score(eval_text):
    match = re.search(r'(?:Score|score)\s*(?::|=)?\s*(\d+)', eval_text)
    if match:
        return int(match.group(1))
    return None

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:
    st.title("🧠 VLSI Mentor AI")
    st.caption("Production Readiness Edition")
    
    mode = st.radio(
        "Mode",
        [
            "Ask Question",
            "Interview Practice",
            "Multi-Agent Discussion",
            "DFT Study Planner"
        ]
    )

    st.markdown("---")

    if mode == "Ask Question":
        st.markdown("### Supported Topics")
        for topic in sorted(list(agent_map.keys())):
            st.markdown(f"- {topic}")
            
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    elif mode == "Interview Practice":
        if st.button("🗑 Clear Interview", use_container_width=True):
            st.session_state.pop("interview_question", None)
            st.session_state.pop("evaluation", None)
            st.session_state["asked_questions"] = []
            st.session_state["evaluations"] = []
            st.rerun()
            
    elif mode == "DFT Study Planner":
        if st.button("🗑 Reset History", use_container_width=True):
            st.session_state["asked_questions"] = []
            st.session_state["evaluations"] = []
            st.session_state.pop("cached_study_plan", None)
            st.success("Study history cleared!")
            st.rerun()

# =====================================
# ASK QUESTION MODE
# =====================================
if mode == "Ask Question":
    st.title("🧠 VLSI Mentor AI")
    st.caption("Ask DFT, ATPG, Scan, MBIST, JTAG, IJTAG, Wrapper, OCC, Timing, Linux, and TCL questions.")

    # Display Old Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    question = st.chat_input("Ask a VLSI / DFT Question...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    followup_words = ["it", "this", "that", "they", "those"]
                    q = question.lower()

                    if len(question.split()) <= 5 and any(word in q for word in followup_words):
                        topic = st.session_state.get("last_topic", "GENERAL")
                    else:
                        topic = route_question(question)

                    st.session_state["last_topic"] = topic

                    topic_colors = {
                        "SCAN": "🟢",
                        "ATPG": "🔵",
                        "EDT": "🟣",
                        "MBIST": "🟠",
                        "JTAG": "🟡",
                        "IJTAG": "🟤",
                        "WRAPPER": "⚪",
                        "OCC": "🔴",
                        "STA": "🟩",
                        "BOUNDARYSCAN": "🧬",
                        "GLS": "🧪",
                        "LINUX": "🐧",
                        "TCL": "⚙️"
                    }

                    st.success(f"{topic_colors.get(topic, '⚫')} Router Selected: {topic}")
                    agent = agent_map.get(topic, ask_general)
                    st.info(f"Answered by: {topic} Expert")

                    history = st.session_state.messages[:-1]
                    answer = agent(question, history)

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# =====================================
# INTERVIEW PRACTICE MODE
# =====================================
elif mode == "Interview Practice":
    st.title("🎤 VLSI Interview Practice")
    st.caption("Practice mock interviews. Your responses will be graded and logged to build your customized study guide.")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox(
            "Select Interview Topic",
            ["SCAN", "ATPG", "EDT", "MBIST", "JTAG", "STA", "GLS", "BOUNDARYSCAN"]
        )
    with col2:
        difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])

    if st.button("Generate Question", use_container_width=True):
        with st.spinner("Generating interview question..."):
            question = ask_interview_question(topic, difficulty, st.session_state["asked_questions"])
            st.session_state["interview_question"] = question
            st.session_state["asked_questions"].append(question)
            st.session_state.pop("evaluation", None) # clear old evaluation

    if "interview_question" in st.session_state:
        st.markdown("### Interview Question")
        st.info(st.session_state["interview_question"])

        candidate_answer = st.text_area("Type your answer below:", height=200)

        if st.button("Evaluate Answer", use_container_width=True):
            with st.spinner("Evaluating your response..."):
                result = evaluate_answer(st.session_state["interview_question"], candidate_answer)
                st.session_state["evaluation"] = result
                st.session_state.evaluations.append(result)

    if "evaluation" in st.session_state:
        st.markdown("### Evaluation Report")
        st.markdown(st.session_state["evaluation"])

# =====================================
# MULTI-AGENT DISCUSSION MODE
# =====================================
elif mode == "Multi-Agent Discussion":
    st.title("🤖 Multi-Agent Discussion (LangGraph)")
    st.caption("Watch specialized DFT experts debate a topic dynamically, moderated by a Critic and compiled by a Lead Architect.")

    question = st.text_input("Enter a complex DFT/VLSI Question:")

    if st.button("Launch Discussion", use_container_width=True):
        if question:
            with st.status("Executing LangGraph multi-agent workflow...") as status:
                st.write("Initializing agent graph and routing question...")
                result = run_graph(question)
                status.update(label="Graph workflow execution completed!", state="complete")
                
            # Render stats
            st.markdown(f"**Active Participants:** " + ", ".join([f"`{p}`" for p in result['participants']]))
            st.markdown(f"**Self-Correction Rounds Run:** `{result['revision_round']}`")
            st.markdown(f"**Critic Quality Score:** `{result['critic_score']}/10`")
            
            with st.expander("👁 View Full Multi-Agent Transcript"):
                for idx, item in enumerate(result['discussion_history']):
                    st.markdown(f"### 👤 {item['agent']} Response")
                    st.write(item['content'])
                    st.markdown("---")
                    
                st.markdown("### 🔍 Principal Architect (Critic) Feedback")
                st.write(result['critic_feedback'])
                
            st.success("### 🏆 Lead DFT Architect Compiled Answer")
            st.markdown(result['final_answer'])
        else:
            st.error("Please enter a question.")

# =====================================
# DFT STUDY PLANNER MODE
# =====================================
elif mode == "DFT Study Planner":
    st.title("📚 Personalized DFT Study Planner")
    st.caption("Track your learning progress, view analytics, and follow your customized 7-day study guide.")

    if st.session_state.evaluations:
        scores = []
        for e in st.session_state.evaluations:
            score = extract_score(e)
            if score is not None:
                scores.append(score)

        if scores:
            col1, col2, col3 = st.columns(3)
            avg_score = sum(scores) / len(scores)
            
            with col1:
                st.metric("Average Score", f"{avg_score:.2f} / 10")
            with col2:
                st.metric("Questions Attempted", len(scores))
            with col3:
                proficiency = "Novice"
                if avg_score >= 8:
                    proficiency = "Proficient"
                elif avg_score >= 5:
                    proficiency = "Competent"
                st.metric("Proficiency Level", proficiency)

            # Score history chart
            st.markdown("### 📈 Score History")
            st.line_chart(scores)
        else:
            st.info("Start taking practice interviews to visualize your score history here!")

        if st.button("🔄 Regenerate 7-Day Study Plan", use_container_width=True):
            with st.spinner("Generating study plan..."):
                study_plan = generate_study_plan(st.session_state.asked_questions, st.session_state.evaluations)
                st.session_state["cached_study_plan"] = study_plan
                st.rerun()

        st.markdown("---")
        st.markdown("### 📋 Your Personalized 7-Day Study Guide")
        
        if "cached_study_plan" in st.session_state:
            st.markdown(st.session_state["cached_study_plan"])
        else:
            with st.spinner("Generating study plan..."):
                study_plan = generate_study_plan(st.session_state.asked_questions, st.session_state.evaluations)
                st.session_state["cached_study_plan"] = study_plan
                st.markdown(study_plan)
    else:
        st.info("No practice interview history found. Please practice some questions in **Interview Practice** mode first, and then come back to see your customized study plan!")
