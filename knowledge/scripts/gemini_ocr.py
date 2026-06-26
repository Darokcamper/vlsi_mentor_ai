import os
import sys
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
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
PDF_DIR = KNOWLEDGE_DIR / "pdfs"
TXT_DIR = KNOWLEDGE_DIR / "txt"
VECTORSTORE_DIR = KNOWLEDGE_DIR / "vectorstore"

# Ensure directories exist
TXT_DIR.mkdir(parents=True, exist_ok=True)

# Load environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n[Error] GEMINI_API_KEY not found in environment or .env file.")
        print("Please add 'GEMINI_API_KEY=your_api_key_here' to C:/vlsi-mentor-ai/.env and try again.")
        sys.exit(1)
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        print("\n[Error] google-genai package is not installed.")
        print("Please run: .\\venv\\Scripts\\pip.exe install google-genai")
        sys.exit(1)

def ocr_pdf_with_gemini(pdf_path, client):
    """Renders PDF pages to images in memory and uses Gemini 3.1 Flash Lite to transcribe them."""
    from google.genai import types
    print(f"Opening: {pdf_path.name}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Failed to open PDF {pdf_path.name}: {e}")
        return None
        
    prompt = """You are an expert VLSI and DFT engineer.
Transcribe the handwritten text, slides, diagrams, and notes from this page.
Guidelines:
1. Maintain all technical terms, formulas, and connections accurately.
2. Clean up any obvious spelling errors or scribbles, but do not omit technical content.
3. Explain diagrams and circuits in text (e.g. describing gates, inputs, outputs, and connections) to help semantic search index them.
4. Output clean, structured markdown text.
5. If the page is blank or has no content, just output an empty string."""

    transcribed_pages = []
    
    for page_num in range(len(doc)):
        print(f"  Transcribing page {page_num + 1}/{len(doc)}...", end="", flush=True)
        page = doc[page_num]
        
        # Render page to image at high resolution (150 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        
        # Call Gemini API
        retries = 0
        while retries < 6:
            try:
                response = client.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=[
                        types.Part.from_bytes(
                            data=img_data,
                            mime_type="image/png"
                        ),
                        prompt
                    ]
                )
                transcribed_text = response.text.strip()
                transcribed_pages.append(f"--- PAGE {page_num + 1} ---\n{transcribed_text}\n")
                print(" Done.")
                break
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["429", "quota", "rate", "limit", "503", "unavailable", "network", "unreachable"]):
                    retries += 1
                    sleep_time = 4.0 * (2.0 ** retries)
                    print(f" Rate limited/Unavailable. Retrying in {sleep_time:.1f}s... (Attempt {retries}/6)")
                    time.sleep(sleep_time)
                else:
                    print(f" Failed: {e}")
                    transcribed_pages.append(f"--- PAGE {page_num + 1} ---\n[OCR failed for this page: {e}]\n")
                    break
        else:
            print(" Skipping page due to persistent failures.")
            transcribed_pages.append(f"--- PAGE {page_num + 1} ---\n[OCR failed due to rate limits/unavailability]\n")
            
        # 4.5 second delay to respect 15 RPM rate limits
        time.sleep(4.5)
        
    doc.close()
    return "\n".join(transcribed_pages)

def process_all_pdfs():
    client = get_gemini_client()
    
    # Locate all PDFs
    pdf_files = list(PDF_DIR.glob("*.pdf")) + list(KNOWLEDGE_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in knowledge directory.")
    
    success_count = 0
    
    for pdf_path in pdf_files:
        txt_path = TXT_DIR / f"{pdf_path.stem}.txt"
        
        # Check if already processed using Gemini without failures
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "[OCR: Gemini]" in content and "OCR failed" not in content:
                print(f"Already successfully processed with Gemini: {pdf_path.name}")
                continue
                
        print(f"\nProcessing PDF: {pdf_path.name}")
        transcription = ocr_pdf_with_gemini(pdf_path, client)
        
        if transcription:
            # Write to cache with marker
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"[OCR: Gemini]\n{transcription}")
            print(f"Successfully saved Gemini OCR to {txt_path.name}")
            success_count += 1
            
    print(f"\nProcessed {success_count} PDFs using Gemini.")
    
    if success_count > 0:
        print("\nRebuilding FAISS vector index...")
        from core.rag_builder import build_index
        build_index()
        print("FAISS index successfully rebuilt!")

if __name__ == "__main__":
    process_all_pdfs()
