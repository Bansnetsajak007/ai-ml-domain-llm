from pathlib import Path
import json
import fitz  # PyMuPDF - works on Windows without external tools
import pytesseract
from pdf2image import convert_from_path
from ebooklib import epub
from bs4 import BeautifulSoup
import docx
import re
from tqdm import tqdm

DATA_DIR = Path("../Data/Day1/Books")
OUT_DIR = Path("Processed_dataset")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------
# Extractors
# -----------------------

def pdf_to_text(pdf):
    """Extract text from PDF using PyMuPDF (works on Windows)"""
    try:
        doc = fitz.open(pdf)
        text = []
        for page in doc:
            text.append(page.get_text())
        doc.close()
        return "\n".join(text)
    except Exception as e:
        print(f"   PyMuPDF error: {e}")
        return ""

def ocr_pdf(pdf):
    pages = convert_from_path(pdf, dpi=200)  # higher DPI = better OCR
    text = []
    for p in pages:
        t = pytesseract.image_to_string(
            p,
            config="--oem 3 --psm 3"
        )
        text.append(t)
    return "\n".join(text)

def epub_to_text(path):
    book = epub.read_epub(path)
    out = []
    for item in book.get_items():
        if item.get_type() == 9:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            out.append(soup.get_text())
    return "\n".join(out)

def docx_to_text(path):
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

# -----------------------
# Cleaning
# -----------------------

def clean(text):
    text = text.replace("\x00", "")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)

    # remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    return text.strip()

# -----------------------
# Chunking
# -----------------------

def chunk(text, size=1200):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

# -----------------------
# Smart PDF handler
# -----------------------

def smart_pdf_extract(pdf):
    text = pdf_to_text(pdf)

    # If very little text → scanned → OCR it
    if len(text.strip()) < 1000:
        text = ocr_pdf(pdf)

    return text

# -----------------------
# Main runner
# -----------------------

def process(file, idx):
    ext = file.suffix.lower()

    if ext == ".pdf":
        text = smart_pdf_extract(file)

    elif ext == ".epub":
        text = epub_to_text(file)

    elif ext == ".docx":
        text = docx_to_text(file)

    else:
        return

    text = clean(text)

    if len(text) < 1000:
        return

    out = OUT_DIR / f"book_{idx}.jsonl"

    with out.open("w", encoding="utf-8") as f:
        for c in chunk(text):
            f.write(json.dumps({"text": c}) + "\n")

# -----------------------
# Run
# -----------------------

files = list(DATA_DIR.rglob("*"))

for i, f in enumerate(tqdm(files)):
    try:
        process(f, i)
    except Exception as e:
        print(f"Failed: {f} - {e}")
