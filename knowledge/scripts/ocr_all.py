from pathlib import Path
import subprocess

PDF_FOLDER = Path("../pdfs")
OCR_FOLDER = Path("../ocr_pdfs")

OCR_FOLDER.mkdir(exist_ok=True)

for pdf in PDF_FOLDER.glob("*.pdf"):

    output_pdf = OCR_FOLDER / pdf.name

    print(f"\nProcessing: {pdf.name}")

    try:

        subprocess.run(
            [
                "ocrmypdf",
                "--force-ocr",
                str(pdf),
                str(output_pdf)
            ],
            check=True
        )

        print("SUCCESS")

    except Exception as e:

        print(f"FAILED: {e}")