import os
# Disable oneDNN to avoid the Unimplemented ConvertPirAttribute2RuntimeAttribute error on Windows CPU
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import fitz
import tempfile
from pathlib import Path
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=False)

def ocr_pdf(pdf_path, txt_path):
    print(f"Processing: {pdf_path}")
    doc = fitz.open(pdf_path)
    all_page_texts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Extract digital text first
        text = page.get_text("text").strip()
        
        # Fallback to OCR if page has little text
        if len(text) < 50:
            print(f"  Page {page_num + 1}/{len(doc)}: Scanned page. Running OCR...")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                temp_img.write(img_bytes)
                temp_img_path = temp_img.name
                
            try:
                result = ocr.ocr(temp_img_path)
                text_lines = []
                if result and result[0]:
                    for line in result[0]:
                        text_lines.append(line[1][0])
                text = "\n".join(text_lines)
            except Exception as e:
                print(f"  Page {page_num + 1} OCR failed: {e}")
                text = ""
            finally:
                try:
                    os.remove(temp_img_path)
                except OSError:
                    pass
        else:
            print(f"  Page {page_num + 1}/{len(doc)}: Digital text extracted.")
            
        all_page_texts.append(f"--- PAGE {page_num + 1} ---\n{text}\n")
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_page_texts))
    print(f"Saved: {txt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ocr_worker.py <pdf_path> <txt_path>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    txt_path = sys.argv[2]
    ocr_pdf(pdf_path, txt_path)
