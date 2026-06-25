import os
import re
import fitz
import pickle
import subprocess
import numpy as np
from pathlib import Path

# Prevent PyTorch/OpenMP multithreading deadlocks on Windows (especially under Streamlit threads)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"

try:
    import torch
    torch.set_num_threads(1)
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
    pdf_files = list((KNOWLEDGE_DIR / "pdfs").glob("*.pdf")) + list(KNOWLEDGE_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process.")
    
    for pdf_path in pdf_files:
        txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
        
        # Check if already cached
        if txt_path.exists() and txt_path.stat().st_size > 50:
            print(f"Already cached text for: {pdf_path.name}")
            continue
            
        print(f"Processing: {pdf_path.name}...")
        try:
            doc = fitz.open(pdf_path)
            
            # Check text length across first few pages
            total_text = ""
            for page in doc:
                total_text += page.get_text("text").strip()
                
            # If text is too short, we need to run OCR
            if len(total_text) < 100:
                print(f"  Scanned PDF detected. Running OCR...")
                ocr_pdf_path = KNOWLEDGE_DIR / "ocr_pdfs" / f"{pdf_path.stem}_ocr.pdf"
                ocr_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Perform OCR if ocr PDF does not exist or is empty
                if not ocr_pdf_path.exists() or ocr_pdf_path.stat().st_size == 0:
                    success = ocr_document(pdf_path, ocr_pdf_path)
                    if not success:
                        print(f"  Skipping {pdf_path.name} due to OCR failure.")
                        continue
                
                # Load text from the OCR PDF
                ocr_doc = fitz.open(ocr_pdf_path)
                all_page_texts = []
                for page_num in range(len(ocr_doc)):
                    page_text = ocr_doc[page_num].get_text("text").strip()
                    all_page_texts.append(f"--- PAGE {page_num + 1} ---\n{page_text}\n")
                ocr_doc.close()
            else:
                print(f"  Digital PDF detected.")
                all_page_texts = []
                for page_num in range(len(doc)):
                    page_text = doc[page_num].get_text("text").strip()
                    all_page_texts.append(f"--- PAGE {page_num + 1} ---\n{page_text}\n")
            
            doc.close()
            
            # Write to cache
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_page_texts))
                
            print(f"Successfully cached text for {pdf_path.name}")
            
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")

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
    
    print("Computing embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
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

def retrieve(query, top_k=5):
    """Retrieves top_k chunks matching the query, returning a list of dictionaries with source, page, text, and score."""
    if not load_rag():
        print("RAG index files not found. Please build the index first.")
        return []
        
    query_vector = _loaded_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)
    
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
