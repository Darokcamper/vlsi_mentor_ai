# pdf_to_images.py

import fitz
from pathlib import Path

KNOWLEDGE_DIR = Path("C:/vlsi-mentor-ai/knowledge")
IMG_DIR = KNOWLEDGE_DIR / "images"

IMG_DIR.mkdir(exist_ok=True)

for pdf_file in KNOWLEDGE_DIR.rglob("*.pdf"):
    if "ocr_pdfs" in pdf_file.parts:
        continue
    print("Processing:", pdf_file.name)

    pdf = fitz.open(pdf_file)

    out_dir = IMG_DIR / pdf_file.stem
    out_dir.mkdir(exist_ok=True)

    for page_num in range(len(pdf)):

        page = pdf[page_num]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(3, 3)
        )

        pix.save(
            out_dir / f"page_{page_num+1}.png"
        )

    pdf.close()