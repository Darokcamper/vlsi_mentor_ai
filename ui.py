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
            "DFT Study Planner",
            "OCR Progress Tracker"
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

# =====================================
# OCR PROGRESS TRACKER MODE
# =====================================
elif mode == "OCR Progress Tracker":
    import os
    import re
    import time
    from pathlib import Path
    import fitz  # PyMuPDF

    st.title("📂 Knowledge Base OCR Status")
    st.caption("Monitor the progress of Gemini transcribing and OCR-ing your scanned PDF notes.")

    # Paths
    PROJECT_ROOT = Path("C:/vlsi-mentor-ai")
    KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
    PDF_DIR = KNOWLEDGE_DIR / "pdfs"
    TXT_DIR = KNOWLEDGE_DIR / "txt"

    # Find all PDFs
    pdf_files = sorted(list(PDF_DIR.glob("*.pdf")) + list(KNOWLEDGE_DIR.glob("*.pdf")))
    total_pdfs = len(pdf_files)

    if total_pdfs == 0:
        st.warning("No PDF files found in knowledge directories.")
    else:
        # Calculate total pages across all PDFs (cache this in session state so it doesn't run on every rerun)
        if "total_pdf_pages" not in st.session_state or "pdf_page_counts" not in st.session_state:
            with st.spinner("Analyzing PDF page counts..."):
                total_pages = 0
                pdf_page_counts = {}
                for pdf_path in pdf_files:
                    try:
                        doc = fitz.open(pdf_path)
                        pages = len(doc)
                        pdf_page_counts[pdf_path.name] = pages
                        total_pages += pages
                        doc.close()
                    except Exception as e:
                        pdf_page_counts[pdf_path.name] = 0
                st.session_state["total_pdf_pages"] = total_pages
                st.session_state["pdf_page_counts"] = pdf_page_counts
        
        total_pages = st.session_state["total_pdf_pages"]
        pdf_page_counts = st.session_state["pdf_page_counts"]

        # --- Read Ground Truth directly from Filesystem ---
        processed_files_disk = []
        failed_files_disk = []
        skipped_pages_disk = 0
        
        for pdf_path in pdf_files:
            txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    if "[OCR: Gemini]" in content:
                        if "OCR failed" in content:
                            failed_files_disk.append(pdf_path.name)
                        else:
                            processed_files_disk.append(pdf_path.name)
                            # Count completed pages inside this file by searching for PAGE markers
                            page_markers = re.findall(r"--- PAGE \d+ ---", content)
                            if page_markers:
                                skipped_pages_disk += len(page_markers)
                            else:
                                skipped_pages_disk += pdf_page_counts.get(pdf_path.name, 0)
                except Exception:
                    pass

        # Find latest log file in tasks directory
        tasks_dir = Path("C:/Users/hazar/.gemini/antigravity-cli/brain/483afadf-d34a-479e-89e3-0b1a32a03968/.system_generated/tasks")
        log_files = sorted(list(tasks_dir.glob("task-*.log")), key=os.path.getmtime, reverse=True) if tasks_dir.exists() else []

        # Default task tracking states
        current_file = None
        current_pages = 0
        current_total_pages = 0
        total_keys_found = 1
        active_key_index = 1
        latest_log_name = "None"
        log_content = ""

        if log_files:
            latest_log = log_files[0]
            latest_log_name = latest_log.name
            try:
                with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
                    log_content = f.read()
                
                # Scan lines for active job information
                for line in log_content.splitlines():
                    line = line.strip()
                    if line.startswith("Processing PDF:"):
                        current_file = line.replace("Processing PDF:", "").strip()
                    elif "Transcribing page" in line:
                        match = re.search(r"Transcribing page (\d+)/(\d+)", line)
                        if match:
                            current_pages = int(match.group(1))
                            current_total_pages = int(match.group(2))
                    elif "Initialized" in line and "Gemini client" in line:
                        match = re.search(r"Initialized (\d+) Gemini client", line)
                        if match:
                            total_keys_found = int(match.group(1))
                    elif "[Key Rotation]" in line:
                        match = re.search(r"Rotating to Key (\d+)", line)
                        if match:
                            active_key_index = int(match.group(1))
            except Exception:
                pass

        # Overall calculations based on disk ground truth (which is always accurate!)
        completed_pages = skipped_pages_disk
        
        # Add active page progress if there is a running task
        if current_file and current_file not in processed_files_disk:
            completed_pages += max(0, current_pages - 1)
            
        percentage = (completed_pages / total_pages) * 100 if total_pages > 0 else 0
        
        # Estimate remaining time
        avg_time_per_page = 6.5
        remaining_pages = total_pages - completed_pages
        remaining_seconds = remaining_pages * avg_time_per_page
        remaining_hours = remaining_seconds / 3600
        
        # Layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total PDFs Completed", f"{len(processed_files_disk)} / {total_pdfs}")
        with col2:
            st.metric("Total Pages Completed", f"{completed_pages} / {total_pages} ({percentage:.1f}%)")
        with col3:
            st.metric("Estimated Time Remaining", f"{remaining_hours:.1f} hours" if remaining_pages > 0 else "Complete")

        # Progress bar
        st.progress(min(percentage / 100.0, 1.0))

        if current_file and log_files and os.path.exists(log_files[0]) and time.time() - os.path.getmtime(log_files[0]) < 120:
            st.info(f"🔄 **Currently Processing**: `{current_file}` (Page {current_pages}/{current_total_pages})")
            st.warning(f"🔑 **Active API Key**: Key {active_key_index} of {total_keys_found} (Rotation Enabled)")
        else:
            st.success("🎉 **OCR Process Idle or Completed!**")

        # Refresh and Auto-Refresh control
        col_ref, col_auto = st.columns([1, 4])
        with col_ref:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
        with col_auto:
            st.caption(f"Last updated: {time.strftime('%X')} (using log: {latest_log_name})")

        # Expanders for detailed lists
        with st.expander(f"📁 View Processed Files ({len(processed_files_disk)})"):
            for idx, name in enumerate(processed_files_disk, 1):
                st.write(f"{idx}. {name} (Pages: {pdf_page_counts.get(name, 'Unknown')})")

        with st.expander(f"⏳ View Incomplete or Remaining Files ({total_pdfs - len(processed_files_disk)})"):
            remaining_list = [p.name for p in pdf_files if p.name not in processed_files_disk]
            for idx, name in enumerate(remaining_list, 1):
                status_lbl = "⚠️ Incomplete (OCR failed pages)" if name in failed_files_disk else "⏳ Not started"
                st.write(f"{idx}. {name} ({status_lbl} - Pages: {pdf_page_counts.get(name, 'Unknown')})")

        if log_content:
            with st.expander("📝 View Live Log Output (Last 30 Lines)"):
                log_lines = log_content.splitlines()
                last_lines = log_lines[-30:]
                st.code("\n".join(last_lines), language="text")

