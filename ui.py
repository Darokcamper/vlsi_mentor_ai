import streamlit as st

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

from agents.general_agent import ask_general

from agents.interviewer_agent import ask_interview_question
from agents.evaluator_agent import evaluate_answer

from agents.interviewer_agent import ask_interview_question
from agents.evaluator_agent import evaluate_answer

from agents.coordinator_agent import run_multi_agent
from agents.reviewer_agent import review

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
}

# =====================================

# SIDEBAR

# =====================================

with st.sidebar:

    st.title("🧠 VLSI Mentor AI")

    mode = st.radio(
        "Mode",
        [
            "Ask Question",
            "Interview Practice",
            "Multi-Agent Discussion"
        ]
    )

    st.markdown("---")

    if mode == "Ask Question":

        st.markdown("""
### Supported Topics

- SCAN
- ATPG
- EDT
- MBIST
- JTAG
- IJTAG
- WRAPPER
- OCC
- STA
""")

        if st.button("🗑 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    else:

        if st.button("🗑 Clear Interview"):

            st.session_state.pop(
                "interview_question",
                None
            )

            st.session_state.pop(
                "evaluation",
                None
            )
            st.session_state["asked_questions"] = []  
            st.rerun()



# =====================================

# ASK QUESTION MODE

# =====================================

if mode == "Ask Question":

    st.title("🧠 VLSI Mentor AI")

    st.caption("Ask DFT, ATPG, Scan, MBIST, JTAG, IJTAG, Wrapper, OCC and STA questions.")

# -----------------------------
# Display Old Chat
# -----------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------

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

                if len(question.split()) <= 5 and any(
                    word in q for word in followup_words
                ):
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
                }

                st.success(f"{topic_colors.get(topic,'⚫')} Router Selected: {topic}")

                agent = agent_map.get(topic, ask_general)

                st.info(f"Answered by: {topic} Expert")

                history = st.session_state.messages[:-1]

                answer = agent(question, history)

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:

                error_msg = f"Error: {e}"

                st.error(error_msg)

                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


# =====================================
# INTERVIEW MODE
# =====================================

elif mode == "Interview Practice":

    st.title("🎤 VLSI Interview Practice")

    topic = st.selectbox(
        "Topic",
        [
            "SCAN",
            "ATPG",
            "EDT",
            "MBIST",
            "JTAG"
        ]
    )

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    if st.button(
        "Generate Question",
        use_container_width=True
    ):
        st.write(st.session_state["asked_questions"])
        
        question = ask_interview_question(
            topic,
            difficulty,
            st.session_state["asked_questions"]
        )

        st.session_state["interview_question"] = question

        st.session_state["asked_questions"].append(
            question
        )

    if "interview_question" in st.session_state:

        st.markdown(
            "## Interview Question"
        )

        st.info(
            st.session_state[
                "interview_question"
            ]
        )

        candidate_answer = st.text_area(
            "Your Answer",
            height=250
        )

        if st.button(
            "Evaluate Answer",
            use_container_width=True
        ):

            result = evaluate_answer(
                st.session_state[
                    "interview_question"
                ],
                candidate_answer
            )

            st.session_state[
                "evaluation"
            ] = result

    if "evaluation" in st.session_state:

        st.markdown(
            "## Evaluation"
        )

        st.write(
            st.session_state[
                "evaluation"
            ]
        )
        

elif mode == "Multi-Agent Discussion":

    st.title("🤖 Multi-Agent Discussion")

    question = st.text_input(
        "Question"
    )

    if st.button("Run Discussion"):

        responses = run_multi_agent(
            question
        )

        for name, answer in responses:

            st.subheader(
                f"{name} Expert"
            )

            st.write(answer)

            discussion = "\n\n".join(
                [f"{name}:\n{answer}" for name, answer in responses]
            )

            final_answer = review(
                question,
                discussion
            )

        st.subheader(
            "Reviewer"
        )

        st.success(final_answer)
        
