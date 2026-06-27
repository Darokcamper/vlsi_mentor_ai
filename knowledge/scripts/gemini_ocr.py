import os
import sys
import re
import time
from pathlib import Path
import fitz  # PyMuPDF
import io

# Set encoding to utf-8 for stdout to prevent cp1252/charmap errors on Windows
if sys.platform.startswith("win"):
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

# Set paths
PROJECT_ROOT = Path("C:/vlsi-mentor-ai")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
PDF_DIR = KNOWLEDGE_DIR / "pdfs"
TXT_DIR = KNOWLEDGE_DIR / "txt"
VECTORSTORE_DIR = KNOWLEDGE_DIR / "vectorstore"

# Ensure directories exist
TXT_DIR.mkdir(parents=True, exist_ok=True)

# Load environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def get_gemini_clients():
    keys = []
    
    # 1. Check GEMINI_API_KEY (could be comma-separated)
    env_val = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_val:
        if "," in env_val:
            keys.extend([k.strip() for k in env_val.split(",") if k.strip()])
        else:
            keys.append(env_val.strip())
            
    # 2. Check for GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    i = 1
    while True:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        keys.append(key.strip())
        i += 1
        
    # Remove duplicates while preserving order
    seen = set()
    keys = [x for x in keys if not (x in seen or seen.add(x))]
    
    if not keys:
        print("\n[Error] No GEMINI_API_KEY found in environment or .env file.")
        print("Please add 'GEMINI_API_KEY=your_api_key' to C:/vlsi-mentor-ai/.env and try again.")
        sys.exit(1)
        
    try:
        from google import genai
        clients = []
        for k in keys:
            clients.append(genai.Client(api_key=k))
        print(f"Initialized {len(clients)} Gemini client(s) for key rotation.")
        return clients
    except ImportError:
        print("\n[Error] google-genai package is not installed.")
        print("Please run: .\\venv\\Scripts\\pip.exe install google-genai")
        sys.exit(1)

def ocr_pdf_with_gemini(pdf_path, clients, client_state, txt_path):
    """Renders PDF pages to images in memory and uses Gemini 2.5 Flash to transcribe them.
    Loads existing page-level transcripts and saves progressive progress after every page to prevent starting from scratch on interruption."""
    from google.genai import types
    print(f"Opening: {pdf_path.name}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Failed to open PDF {pdf_path.name}: {e}")
        return False
        
    prompt = """You are an expert VLSI and DFT engineer.
Your task is to transcribe the slide, document, or handwritten page literally and comprehensively.
CRITICAL GUIDELINES:
1. DO NOT summarize, paraphrase, or omit any content. Transcribe every word, note, scribbled comment, formula, and table literally.
2. If there are calculations, write out all equations, numbers, and math symbols exactly as they appear.
3. For diagrams and schematics (e.g. gates, scan chains, EDT logic, MUXes), write a detailed textual explanation describing the components, inputs, outputs, connections, and flow of signals to help semantic search index them.
4. Output in clean, structured Markdown, maintaining the original page layout as closely as possible.
5. If the page is blank, output an empty string."""

    # 1. Parse existing txt file if it has progress
    transcribed_pages = {}
    if txt_path.exists():
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            # Split content by the PAGE marker
            parts = re.split(r"--- PAGE (\d+) ---\n", content)
            if len(parts) > 1:
                for i in range(1, len(parts), 2):
                    p_num = int(parts[i])
                    p_text = parts[i+1].strip()
                    # Only cache pages that did NOT fail
                    if "ocr failed" not in p_text.lower() and p_text:
                        transcribed_pages[p_num] = p_text
                print(f"  Loaded {len(transcribed_pages)} pages from existing cache.")
        except Exception as e:
            print(f"  Failed to load cached pages: {e}")

    has_new_transcriptions = False
    
    for page_num_0indexed in range(len(doc)):
        page_num = page_num_0indexed + 1
        
        # Check if already transcribed
        if page_num in transcribed_pages:
            continue
            
        print(f"  Transcribing page {page_num}/{len(doc)}...", end="", flush=True)
        page = doc[page_num_0indexed]
        
        # Render page to image at high resolution (150 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        
        # Call Gemini API
        retries = 0
        page_text = ""
        models = ['gemini-2.5-flash', 'gemini-2.5-flash-lite']
        if "model_index" not in client_state:
            client_state["model_index"] = 0

        while retries < 10:
            client_index = client_state["index"]
            client = clients[client_index]
            model_name = models[client_state["model_index"]]
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(
                            data=img_data,
                            mime_type="image/png"
                        ),
                        prompt
                    ]
                )
                page_text = response.text.strip() if response.text else ""
                transcribed_pages[page_num] = page_text
                print(" Done.")
                has_new_transcriptions = True
                break
            except Exception as e:
                err_str = str(e).lower()
                is_quota_error = any(x in err_str for x in ["quota", "exhausted", "429"])
                if is_quota_error:
                    # If we have tried all keys, fall back to the next model if available
                    if retries >= len(clients) and client_state["model_index"] < len(models) - 1:
                        old_model = models[client_state["model_index"]]
                        client_state["model_index"] += 1
                        new_model = models[client_state["model_index"]]
                        print(f"\n[Fallback] Persistent rate limit/quota hit for '{old_model}' across all keys. Falling back to '{new_model}'...")
                        retries = 0
                        continue

                    next_index = (client_index + 1) % len(clients)
                    client_state["index"] = next_index
                    sleep_time = 12.0 + (3.0 * retries)
                    print(f"\n[Key Rotation] Key {client_index + 1} rate limited. Rotating to Key {next_index + 1} and backing off for {sleep_time:.1f}s...")
                    retries += 1
                    time.sleep(sleep_time)
                    continue
                elif any(x in err_str for x in ["503", "unavailable", "network", "unreachable"]):
                    retries += 1
                    sleep_time = 10.0 * (1.5 ** retries)
                    print(f" Service Unavailable/Network Error. Retrying in {sleep_time:.1f}s... (Attempt {retries}/10)")
                    time.sleep(sleep_time)
                else:
                    print(f" Failed: {e}")
                    page_text = f"[OCR failed for this page: {e}]"
                    transcribed_pages[page_num] = page_text
                    break
        else:
            print(" Skipping page due to persistent failures.")
            page_text = "[OCR failed due to rate limits/unavailability]"
            transcribed_pages[page_num] = page_text
            
        # Write progressive updates to file immediately
        file_lines = ["[OCR: Gemini]"]
        for p in sorted(transcribed_pages.keys()):
            file_lines.append(f"--- PAGE {p} ---")
            file_lines.append(transcribed_pages[p])
            file_lines.append("")
            
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(file_lines))
            
        # 7.5 second delay to respect rate/TPM limits and avoid resource exhaustion
        time.sleep(7.5)
        
    doc.close()
    return has_new_transcriptions

def process_all_pdfs():
    clients = get_gemini_clients()
    client_state = {"index": 0}
    
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
                
    # Locate all PDFs
    pdf_files = [p for p in KNOWLEDGE_DIR.rglob("*.pdf") if "ocr_pdfs" not in p.parts]
    print(f"Found {len(pdf_files)} PDFs in knowledge directory.")
    
    success_count = 0
    
    for pdf_path in pdf_files:
        txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
        
        rel_path = pdf_path.relative_to(KNOWLEDGE_DIR).as_posix()
        row = catalog.get(rel_path)
        is_vlsiguru = row and row.get("Source") == "VLSIGuru"
        
        if not is_vlsiguru:
            source_info = row.get("Source") if row else "Unknown"
            print(f"Skipping digital book/textbook: {pdf_path.name} (Source: {source_info})")
            continue
            
        # Check page count to avoid OCR on files > 100 pages (prevents free tier exhaustion)
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
        except Exception as e:
            print(f"Failed to open {pdf_path.name} to check page count: {e}")
            continue
            
        if num_pages > 100:
            print(f"Skipping scanned note: {pdf_path.name} (Too many pages: {num_pages} > 100). Gemini API cannot handle large files on free tier.")
            continue
            
        # Check if already processed using Gemini without failures
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "[OCR: Gemini]" in content and "OCR failed" not in content:
                print(f"Already successfully processed with Gemini: {pdf_path.name}")
                continue
                
        print(f"\nProcessing scanned note: {pdf_path.name}")
        has_new = ocr_pdf_with_gemini(pdf_path, clients, client_state, txt_path)
        
        if has_new:
            success_count += 1
            
    print(f"\nProcessed {success_count} PDFs using Gemini.")
    
    if success_count > 0:
        print("\nRebuilding FAISS vector index...")
        from core.rag_builder import build_index
        build_index()
        print("FAISS index successfully rebuilt!")

if __name__ == "__main__":
    process_all_pdfs()
