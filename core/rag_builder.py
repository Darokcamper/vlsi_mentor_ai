import os
import re
import fitz
import pickle
import subprocess
import numpy as np
from pathlib import Path

import sys

# Prevent high CPU/RAM usage and deadlocks on Windows/Intel Core i5:
# - In Streamlit: limit to 1 thread to avoid deadlocks.
# - Offline Indexing: limit to 2 threads to prevent 100% CPU lockup and thermal/RAM throttling.
is_streamlit = any("streamlit" in arg or "streamlit" in sys.argv[0].lower() for arg in sys.argv) or os.environ.get("STREAMLIT_SERVER_PORT") is not None

os.environ["OMP_NUM_THREADS"] = "1" if is_streamlit else "2"
os.environ["MKL_NUM_THREADS"] = "1" if is_streamlit else "2"
os.environ["MKL_DYNAMIC"] = "FALSE"

try:
    import torch
    torch.set_num_threads(1 if is_streamlit else 2)
except Exception:
    pass

from sentence_transformers import SentenceTransformer
import faiss


# Paths
KNOWLEDGE_DIR = Path("C:/vlsi-mentor-ai/knowledge")
TXT_DIR = KNOWLEDGE_DIR / "txt"
VECTORSTORE_DIR = KNOWLEDGE_DIR / "vectorstore"

TXT_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

def ocr_document(pdf_path, ocr_pdf_path):
    """Runs ocrmypdf via subprocess on the pdf file."""
    ocrmypdf_exe = "C:/vlsi-mentor-ai/venv/Scripts/ocrmypdf.exe"
    if not os.path.exists(ocrmypdf_exe):
        ocrmypdf_exe = "ocrmypdf" # fallback to PATH
        
    print(f"Running ocrmypdf on {pdf_path.name}...")
    try:
        result = subprocess.run([
            ocrmypdf_exe,
            "--force-ocr",
            str(pdf_path),
            str(ocr_pdf_path)
        ], capture_output=True, text=True, check=True)
        print(f"OCR successful for {pdf_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ocrmypdf failed for {pdf_path.name}: {e.stderr}")
        return False

def extract_text_from_pdfs():
    """Extracts text page-by-page from all PDFs in the knowledge directory, saving them to TXT_DIR."""
    pdf_files = [p for p in KNOWLEDGE_DIR.rglob("*.pdf") if "ocr_pdfs" not in p.parts]
    print(f"Found {len(pdf_files)} PDFs to process.")
    
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
        txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
        
        # Check if already cached
        if txt_path.exists() and txt_path.stat().st_size > 50:
            print(f"Already cached text for: {pdf_path.name}")
            continue
            
        rel_path = pdf_path.relative_to(KNOWLEDGE_DIR).as_posix()
        row = catalog.get(rel_path)
        is_vlsiguru = row and row.get("Source") == "VLSIGuru"
        
        if is_vlsiguru:
            print(f"Skipping VLSIGuru scanned PDF: {pdf_path.name} (will be transcribed via Gemini OCR)")
            continue
            
        print(f"Processing digital book: {pdf_path.name}...")
        try:
            doc = fitz.open(pdf_path)
            all_page_texts = []
            for page_num in range(len(doc)):
                page_text = doc[page_num].get_text("text").strip()
                all_page_texts.append(f"--- PAGE {page_num + 1} ---\n{page_text}\n")
            doc.close()
            
            # Write to cache
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_page_texts))
            print(f"Successfully cached text for digital book: {pdf_path.name}")
        except Exception as e:
            print(f"Failed to process digital book {pdf_path.name}: {e}")

def chunk_text(text, source_name, chunk_size=1000, overlap=200):
    """Splits a single PDF's cached text into chunks, keeping track of page numbers from markers."""
    chunks = []
    pages = re.split(r"--- PAGE (\d+) ---", text)
    
    if pages and not pages[0].strip().isdigit() and pages[0].strip():
        p_text = pages[0]
        chunks.extend(_make_chunks(p_text, source_name, 1, chunk_size, overlap))
        
    for i in range(1, len(pages), 2):
        try:
            current_page = int(pages[i])
        except ValueError:
            current_page = 1
            
        page_text = pages[i+1] if i+1 < len(pages) else ""
        chunks.extend(_make_chunks(page_text, source_name, current_page, chunk_size, overlap))
        
    return chunks

def _make_chunks(text, source_name, page_num, chunk_size, overlap):
    chunks = []
    text = text.strip()
    if not text:
        return []
        
    words = text.split()
    if not words:
        return []
        
    current_chunk_words = []
    current_len = 0
    
    for word in words:
        current_chunk_words.append(word)
        current_len += len(word) + 1
        
        if current_len >= chunk_size:
            chunk_text = " ".join(current_chunk_words)
            chunks.append({
                "source": source_name,
                "page": page_num,
                "text": chunk_text
            })
            
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_chunk_words):
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
                if overlap_len >= overlap:
                    break
            current_chunk_words = overlap_words
            current_len = overlap_len
            
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunks.append({
            "source": source_name,
            "page": page_num,
            "text": chunk_text
        })
        
    return chunks

def build_index():
    """Reads all cached txt files, chunks them, embeds them, and builds a FAISS index."""
    txt_files = list(TXT_DIR.glob("*.txt"))
    if not txt_files:
        print("No cached text files found. Run extract_text_from_pdfs() first.")
        return
        
    print(f"Found {len(txt_files)} text files. Starting chunking...")
    all_chunks = []
    for txt_path in txt_files:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, txt_path.name)
        all_chunks.extend(chunks)
        
    print(f"Total chunks created: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks to index.")
        return
        
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Computing embeddings (batch_size=32)...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    
    faiss.normalize_L2(embeddings)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    index_path = str(VECTORSTORE_DIR / "faiss_index.index")
    faiss.write_index(index, index_path)
    
    metadata_path = str(VECTORSTORE_DIR / "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(all_chunks, f)
        
    print(f"FAISS index built and saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")

# --- RAG RUNTIME ---

_loaded_model = None
_loaded_index = None
_loaded_metadata = None

# Eagerly load the model in the main thread during module import
try:
    print("Eagerly loading SentenceTransformer model in main thread to avoid Windows multi-threading deadlocks...")
    _loaded_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"Warning: Eager loading of SentenceTransformer failed: {e}")
    _loaded_model = None

def load_rag():
    global _loaded_model, _loaded_index, _loaded_metadata
    
    index_path = VECTORSTORE_DIR / "faiss_index.index"
    metadata_path = VECTORSTORE_DIR / "metadata.pkl"
    
    if not index_path.exists() or not metadata_path.exists():
        return False
        
    if _loaded_model is None:
        try:
            _loaded_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            return False
        
    if _loaded_index is None:
        _loaded_index = faiss.read_index(str(index_path))
        
    if _loaded_metadata is None:
        with open(metadata_path, "rb") as f:
            _loaded_metadata = pickle.load(f)
            
    return True

def retrieve(query, top_k=5, source_filter=None):
    """Retrieves top_k chunks matching the query, returning a list of dictionaries with source, page, text, and score."""
    if not load_rag():
        print("RAG index files not found. Please build the index first.")
        return []
        
    query_vector = _loaded_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)
    
    # If source filter is specified, perform exact local cosine similarity ranking on chunks of that specific file
    if source_filter:
        sf_clean = source_filter.lower().replace(".txt", "").replace(".pdf", "")
        doc_indices = []
        doc_chunks = []
        
        for idx, chunk in enumerate(_loaded_metadata):
            chunk_src_clean = chunk["source"].lower().replace(".txt", "").replace(".pdf", "")
            if sf_clean in chunk_src_clean:
                doc_indices.append(idx)
                doc_chunks.append(chunk)
                
        if not doc_chunks:
            return []
            
        # Reconstruct vectors from FAISS index and calculate cosine similarities locally
        import numpy as np
        doc_vectors = []
        for idx in doc_indices:
            vec = _loaded_index.reconstruct(idx)
            doc_vectors.append(vec)
            
        doc_vectors = np.array(doc_vectors).astype("float32")
        faiss.normalize_L2(doc_vectors)
        
        # Dot product of normalized vectors gives cosine similarity
        scores = np.dot(doc_vectors, query_vector[0])
        
        # Boost scores based on exact keyword/number matching to handle follow-up queries like "what about 3 and 4"
        boosted_ranked = []
        import re
        for score, chunk in zip(scores, doc_chunks):
            boost = 0.0
            # Check for numbers in the query (e.g. "3", "4") matching rule headers (e.g. "3.", "4.")
            for num in re.findall(r"\b\d+\b", query):
                # Strong boost for slide headers starting with '# 3' or '# 4' or starting with '3.'
                if re.search(r"#\s*" + num + r"\b", chunk["text"]):
                    boost += 1.2
                elif re.search(r"^\s*" + num + r"\b[\.\-:]", chunk["text"]):
                    boost += 1.2
                # Medium boost for rule numbers inside the text
                elif re.search(r"\b" + num + r"\b[\.\-:]", chunk["text"]):
                    boost += 0.4
                elif re.search(r"page\s+" + num + r"\b", chunk["text"].lower()):
                    boost += 0.4
                    
            # Check for specific technical terms in the query matching chunk text
            query_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", query) if w.lower() not in ["what", "about", "from", "scan", "rule", "rules", "with", "this", "that", "they", "them"]]
            for qw in query_words:
                if qw in chunk["text"].lower():
                    boost += 0.1
                    
            boosted_ranked.append((score + boost, chunk))
            
        ranked = sorted(boosted_ranked, key=lambda x: x[0], reverse=True)
        
        # Retrieve the top_k boosted results
        top_results = ranked[:top_k]
        
        # Sort the top results by page number to keep chronological slide sequence for the LLM
        sorted_results = sorted(top_results, key=lambda x: x[1]["page"])
        
        results = []
        for score, chunk in sorted_results:
            results.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score)
            })
        return results

    # Global search when no source filter is specified
    scores, indices = _loaded_index.search(query_vector, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_loaded_metadata):
            continue
        chunk = _loaded_metadata[idx]
        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score)
        })
        
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print("Extracting text from PDFs...")
        extract_text_from_pdfs()
        print("Building index...")
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "index":
        print("Building index from current cached text files...")
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) > 2:
            query = sys.argv[2]
        else:
            query = "lockup latch hold violation"
        print(f"Testing retrieval for: '{query}'")
        res = retrieve(query)
        for i, r in enumerate(res):
            print(f"\n[{i+1}] Score: {r['score']:.4f} | Source: {r['source']} (Page {r['page']})")
            print("-" * 60)
            print(r['text'])
