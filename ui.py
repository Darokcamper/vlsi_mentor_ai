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
            from graphs.discussion_graph import run_graph_stream
            
            with st.status("Executing LangGraph multi-agent workflow...") as status:
                progress_placeholder = st.empty()
                status_messages = []
                result = None
                
                for msg, state in run_graph_stream(question):
                    if state is not None:
                        result = state
                    else:
                        status_messages.append(msg)
                        progress_placeholder.markdown("\n".join([f"- {m}" for m in status_messages]))
                
                status.update(label="Graph workflow execution completed!", state="complete")
                
            # Render stats
            if result:
                st.markdown(f"**Active Participants:** " + ", ".join([f"`{p}`" for p in result.get('participants', [])]))
                st.markdown(f"**Self-Correction Rounds Run:** `{result.get('revision_round', 0)}`")
                st.markdown(f"**Critic Quality Score:** `{result.get('critic_score', 0)}/10`")
                
                with st.expander("👁 View Full Multi-Agent Transcript"):
                    for idx, item in enumerate(result.get('discussion_history', [])):
                        st.markdown(f"### 👤 {item['agent']} Response")
                        st.write(item['content'])
                        st.markdown("---")
                        
                    st.markdown("### 🔍 Principal Architect (Critic) Feedback")
                    st.write(result.get('critic_feedback', 'No feedback.'))
                    
                st.success("### 🏆 Lead DFT Architect Compiled Answer")
                st.markdown(result.get('final_answer', 'No answer compiled.'))
            else:
                st.error("Failed to execute LangGraph workflow.")
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
    TXT_DIR = KNOWLEDGE_DIR / "txt"

    # Find all PDFs
    pdf_files = sorted([p for p in KNOWLEDGE_DIR.rglob("*.pdf") if "ocr_pdfs" not in p.parts])
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
        
        # --- Read Ground Truth directly from Filesystem ---
        vlsiguru_total = 0
        vlsiguru_completed = []
        vlsiguru_failed = []
        vlsiguru_completed_pages = 0
        vlsiguru_total_pages = 0
        
        digital_total = 0
        digital_completed = []
        digital_completed_pages = 0
        digital_total_pages = 0

        # Load sources.csv catalog
        import csv
        catalog = {}
        csv_path = KNOWLEDGE_DIR / "sources.csv"
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rel_file = row["File"].replace("\\", "/").strip()
                    catalog[rel_file] = row

        for pdf_path in pdf_files:
            rel_path = pdf_path.relative_to(KNOWLEDGE_DIR).as_posix()
            row = catalog.get(rel_path)
            is_vlsiguru = row and row.get("Source") == "VLSIGuru"
            pages = st.session_state["pdf_page_counts"].get(pdf_path.name, 0)
            
            txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
            exists = txt_path.exists() and txt_path.stat().st_size > 50
            
            if is_vlsiguru:
                vlsiguru_total += 1
                vlsiguru_total_pages += pages
                if exists:
                    try:
                        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if "[OCR: Gemini]" in content:
                            if "ocr failed" in content.lower():
                                vlsiguru_failed.append(pdf_path.name)
                            else:
                                vlsiguru_completed.append(pdf_path.name)
                                page_markers = re.findall(r"--- PAGE \d+ ---", content)
                                vlsiguru_completed_pages += len(page_markers) if page_markers else pages
                    except Exception:
                        pass
            else:
                digital_total += 1
                digital_total_pages += pages
                if exists:
                    digital_completed.append(pdf_path.name)
                    digital_completed_pages += pages

        # Dynamically find the latest conversation directory in brain path
        brain_dir = Path("C:/Users/hazar/.gemini/antigravity-cli/brain")
        active_conv_dir = None
        if brain_dir.exists():
            conv_dirs = [d for d in brain_dir.iterdir() if d.is_dir() and d.name != "scratch"]
            if conv_dirs:
                conv_dirs.sort(key=os.path.getmtime, reverse=True)
                active_conv_dir = conv_dirs[0]
                
        tasks_dir = active_conv_dir / ".system_generated" / "tasks" if active_conv_dir else Path(".")
        log_files = list(tasks_dir.glob("task-*.log")) if tasks_dir.exists() else []
        
        # Classify active logs based on contents
        gemini_log_path = None
        digital_log_path = None
        
        for log_path in log_files:
            mtime = os.path.getmtime(log_path)
            # Only track logs active in the last 15 minutes
            if time.time() - mtime > 900:
                continue
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    header = f.read(2048)
                if "gemini_ocr" in header or "Gemini client" in header or "Processing scanned note:" in header:
                    gemini_log_path = log_path
                elif "rag_builder" in header or "Extracting text from PDFs..." in header or "Computing embeddings" in header:
                    digital_log_path = log_path
            except Exception:
                pass

        # Gemini OCR Log parsing
        gemini_active = False
        gemini_file = None
        gemini_pages = 0
        gemini_total_pages = 0
        gemini_keys_found = 1
        gemini_key_index = 1
        gemini_log_content = ""
        
        if gemini_log_path:
            try:
                with open(gemini_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    gemini_log_content = f.read()
                
                # Check if it was modified recently
                if time.time() - os.path.getmtime(gemini_log_path) < 120:
                    gemini_active = True
                
                for line in gemini_log_content.splitlines():
                    line = line.strip()
                    if line.startswith("Processing scanned note:"):
                        gemini_file = line.replace("Processing scanned note:", "").strip()
                    elif "Transcribing page" in line:
                        match = re.search(r"Transcribing page (\d+)/(\d+)", line)
                        if match:
                            gemini_pages = int(match.group(1))
                            gemini_total_pages = int(match.group(2))
                    elif "Initialized" in line and "Gemini client" in line:
                        match = re.search(r"Initialized (\d+) Gemini client", line)
                        if match:
                            gemini_keys_found = int(match.group(1))
                    elif "[Key Rotation]" in line:
                        match = re.search(r"Rotating to Key (\d+)", line)
                        if match:
                            gemini_key_index = int(match.group(1))
            except Exception:
                pass

        # Digital Book Indexing Log parsing
        digital_active = False
        digital_file = None
        digital_batch = 0
        digital_total_batches = 0
        digital_status = "Idle"
        digital_log_content = ""
        
        if digital_log_path:
            try:
                with open(digital_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    digital_log_content = f.read()
                
                if time.time() - os.path.getmtime(digital_log_path) < 120:
                    digital_active = True
                
                for line in digital_log_content.splitlines():
                    line = line.strip()
                    if line.startswith("Processing digital book:"):
                        digital_file = line.replace("Processing digital book:", "").replace("...", "").strip()
                        digital_status = "Extracting Text"
                    elif "Building index..." in line or "Found" in line and "text files" in line:
                        digital_status = "Chunking and Loading Index"
                    elif "Batches:" in line:
                        match = re.search(r"Batches:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)", line)
                        if match:
                            digital_batch = int(match.group(2))
                            digital_total_batches = int(match.group(3))
                            digital_status = f"Computing Embeddings (Batch {digital_batch}/{digital_total_batches})"
            except Exception:
                pass

        # Scanned notes percentage
        scanned_completed_pages = vlsiguru_completed_pages
        if gemini_active and gemini_file and gemini_file not in vlsiguru_completed:
            scanned_completed_pages += max(0, gemini_pages - 1)
        scanned_percentage = (scanned_completed_pages / vlsiguru_total_pages) * 100 if vlsiguru_total_pages > 0 else 0
        
        # Digital books percentage
        digital_completed_count = len(digital_completed)
        digital_percentage = (digital_completed_count / digital_total) * 100 if digital_total > 0 else 0

        # UI Layout
        st.subheader("⚡ Section 1: Scanned VLSIGuru Notes (Gemini API OCR)")
        st.caption("Processes handwritten scanned lecture slides page-by-page entirely in the cloud.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PDFs Completed", f"{len(vlsiguru_completed)} / {vlsiguru_total}")
        with col2:
            st.metric("Pages Transcribed", f"{scanned_completed_pages} / {vlsiguru_total_pages} ({scanned_percentage:.1f}%)")
        with col3:
            st.metric("OCR Errors (Failed Pages)", f"{len(vlsiguru_failed)}")
            
        st.progress(min(scanned_percentage / 100.0, 1.0))
        
        if gemini_active and gemini_file:
            st.info(f"🔄 **Currently Processing**: `{gemini_file}` (Page {gemini_pages}/{gemini_total_pages})")
            st.warning(f"🔑 **Active API Key**: Key {gemini_key_index} of {gemini_keys_found} (Rotation Enabled)")
        else:
            st.success("🎉 **Gemini OCR Process Idle or Completed!**")
            
        st.write("---")
        
        st.subheader("📖 Section 2: Digital Books (Simple Text & FAISS Indexing)")
        st.caption("Extracts clean text from digital reference manuals and generates vector embeddings.")
        
        cold1, cold2 = st.columns(2)
        with cold1:
            st.metric("Digital Textbooks Processed", f"{digital_completed_count} / {digital_total}")
        with cold2:
            st.metric("Indexing Status", "ACTIVE" if digital_active else "Idle")
            
        st.progress(min(digital_percentage / 100.0, 1.0))
        
        if digital_active:
            st.info(f"🔄 **Indexer Action**: `{digital_status}`" + (f" | Active File: `{digital_file}`" if digital_file else ""))
        else:
            st.success("🎉 **Digital Book Indexer Idle or Completed!**")

        st.write("---")

        # Refresh and Auto-Refresh control
        col_ref, col_auto = st.columns([1, 4])
        with col_ref:
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
        with col_auto:
            st.caption(f"Last updated: {time.strftime('%X')}")
            
        # Detailed Expanders
        with st.expander("📁 View Scanned VLSIGuru Notes Status"):
            st.write(f"**Completed ({len(vlsiguru_completed)}):**")
            for idx, name in enumerate(vlsiguru_completed, 1):
                st.write(f"✓ {name} (Pages: {st.session_state['pdf_page_counts'].get(name, 0)})")
            remaining_vlsiguru = [p.name for p in pdf_files if p.name not in vlsiguru_completed and p.name not in digital_completed]
            st.write(f"**Remaining ({len(remaining_vlsiguru)}):**")
            for idx, name in enumerate(remaining_vlsiguru, 1):
                lbl = "⚠️ Has Failed Pages (retrying)" if name in vlsiguru_failed else "⏳ Waiting"
                st.write(f"- {name} ({lbl} - Pages: {st.session_state['pdf_page_counts'].get(name, 0)})")
                
        with st.expander("📁 View Digital Books Status"):
            st.write(f"**Processed ({len(digital_completed)}):**")
            for idx, name in enumerate(digital_completed, 1):
                st.write(f"✓ {name} (Pages: {st.session_state['pdf_page_counts'].get(name, 0)})")
            # Get textbook names from catalog
            textbook_names = [Path(d["File"]).name for d in catalog.values() if d["Source"] != "VLSIGuru"]
            remaining_digital = [name for name in textbook_names if name not in digital_completed]
            if remaining_digital:
                st.write(f"**Remaining ({len(remaining_digital)}):**")
                for idx, name in enumerate(remaining_digital, 1):
                    st.write(f"- {name} (Pages: {st.session_state['pdf_page_counts'].get(name, 0)})")
            else:
                st.write("All digital textbooks successfully indexed!")
                
        # Live Logs Expanders
        if gemini_log_content:
            latest_gemini_name = gemini_log_path.name if gemini_log_path else "None"
            with st.expander(f"📝 View Gemini OCR Live Log Output (Last 30 Lines - {latest_gemini_name})"):
                st.code("\n".join(gemini_log_content.splitlines()[-30:]), language="text")
                
        if digital_log_content:
            latest_digital_name = digital_log_path.name if digital_log_path else "None"
            with st.expander(f"📝 View Digital Indexer Live Log Output (Last 30 Lines - {latest_digital_name})"):
                st.code("\n".join(digital_log_content.splitlines()[-30:]), language="text")
