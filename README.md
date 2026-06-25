# 🧠 VLSI Mentor AI - Production Edition

VLSI Mentor AI is a production-quality, multi-agent educational platform designed to help students and freshers prepare for VLSI and Design for Testability (DFT) interviews. It provides grounded answers using a page-level citation RAG pipeline, a dynamic multi-agent discussion workflow built on LangGraph, and interactive interview practice with a personalized 7-day study planner.

---

## 🚀 Key Features

1. **RAG-Grounded Answers with Citations**: Evaluates questions using custom DFT/VLSI course notes, automatically extracting and citing source documents and page numbers.
2. **Dynamic Multi-Agent Discussion (LangGraph)**: Routes complex questions to relevant domain experts (e.g. SCAN, STA, ATPG, MBIST, JTAG, OCC, wrappers) who collaborate in a self-correcting critique loop.
3. **Interactive Mock Interviews**: Generates realistic interview questions tailored by topic and difficulty, evaluates candidate answers, and scores them.
4. **Personalized Study Planner**: Analyzes interview performance over time to visualize score history and automatically generate day-by-day learning roadmaps.
5. **Robust OCR Automation**: Runs local asynchronous OCR extraction on scanned PDFs using `ocrmypdf` (multi-core Tesseract backend) with page-level text caching.

---

## 📁 Folder Structure

```text
vlsi-mentor-ai/
│
├── agents/                 # Specialized domain expert and coordinator agents
│   ├── base_agent.py       # Core base agent with RAG context inject & citation formatting
│   ├── expert_agent.py     # Unifies domain experts' ask_expert calls
│   ├── router_agent.py     # LLM-based router classifying questions into domains
│   ├── planner_agent.py    # Personalized study planner agent
│   ├── critic_agent.py     # Principled Architect critiquing the discussion
│   ├── reviewer_agent.py   # Lead Architect compiling the final answer
│   └── *_agent.py          # Domain-specific expert profiles (SCAN, STA, etc.)
│
├── core/                   # Core infrastructure modules
│   ├── llm.py              # LLM configuration (ChatGroq with Streamlit secret fallbacks)
│   ├── rag_builder.py      # PDF text extraction, ocrmypdf automation, and FAISS indexing
│   ├── memory.py           # Short-term chat history formatting
│   └── streaming.py        # Stream response utilities
│
├── graphs/                 # Agent execution graphs
│   └── discussion_graph.py # Redesigned LangGraph self-correcting discussion workflow
│
├── knowledge/              # Knowledge Base files
│   ├── pdfs/               # Input DFT reference PDFs
│   ├── ocr_pdfs/           # Cached OCR-processed PDFs
│   ├── txt/                # Page-by-page cached text files
│   └── vectorstore/        # FAISS index and chunk metadata pickle
│
├── tests/                  # Test suite
│   ├── test_agents.py      # Tests LLM connection, router, and planner agents
│   ├── test_rag.py         # Tests FAISS loading and retriever logic
│   └── test_graph.py       # Tests LangGraph discussion workflow execution
│
├── app.py                  # CLI interview practice runner
├── app1.py                 # CLI Q&A agent runner
├── ui.py                   # Streamlit production web UI
└── requirements.txt        # Package dependencies
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.10+
- Tesseract OCR (required for the OCR pipeline via `ocrmypdf`).
  - **Windows**: Install Tesseract using [UB-Mannheim installers](https://github.com/UB-Mannheim/tesseract/wiki) and ensure it or `ocrmypdf` is in your virtual environment/PATH.

### 2. Install Dependencies
Initialize a virtual environment and install the required python packages:
```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🗄️ Building the RAG Knowledge Base

To extract text from your reference PDFs and build the FAISS vector database, run the RAG builder script:
```bash
./venv/Scripts/python.exe core/rag_builder.py build
```
- **Digital PDFs** are parsed instantly.
- **Scanned PDFs** automatically trigger the multi-core `ocrmypdf` engine to apply a text layer.
- Extracted text is cached in `knowledge/txt/` to speed up future runs.
- The compiled database is saved in `knowledge/vectorstore/`.

To test retrieval alone:
```bash
./venv/Scripts/python.exe core/rag_builder.py test "Why are lockup latches needed?"
```

---

## 💻 Running the Application

### Launch the Streamlit Web Application
```bash
./venv/Scripts/streamlit run ui.py
```
Open your browser at `http://localhost:8501` to access:
- **Ask Question**: Interactive Q&A chat.
- **Interview Practice**: Graded mock interview scenarios.
- **Multi-Agent Discussion**: Visual LangGraph agent debate workflow.
- **DFT Study Planner**: Analytics dashboard and study plans.

### CLI Runners
- Console Q&A: `./venv/Scripts/python.exe app1.py`
- Console Interview: `./venv/Scripts/python.exe app.py`

---

## 🧪 Testing

We use the standard python `unittest` library. Run the complete test suite with:
```bash
./venv/Scripts/python.exe -m unittest discover -s tests
```

---

## 🛡️ Security & Production Guidelines
- **API Keys**: Always load secrets using `os.getenv` or Streamlit secrets (`st.secrets`). Never hardcode keys.
- **Git Ignore**: The `.gitignore` file blocks `.env`, local virtual environments, cached text, and vector stores from being pushed to git.
