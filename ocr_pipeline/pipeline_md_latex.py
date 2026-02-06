from pathlib import Path
import fitz  # PyMuPDF - works on Windows without external tools
import pytesseract
from pdf2image import convert_from_path
from ebooklib import epub
from bs4 import BeautifulSoup
import docx
import re
from tqdm import tqdm
 
# Install: pip install pymupdf4llm
try:
    import pymupdf4llm
    LATEX_SUPPORT = True
except ImportError:
    LATEX_SUPPORT = False
 
#dynamic path
DATA_DIR = Path("../Data/Day2")
OUT_DIR = Path("processed_dataset_latex_md/Day2")
OUT_DIR.mkdir(exist_ok=True)
 
# -----------------------
# Extractors
# -----------------------
 
def pdf_to_text(pdf):
    """Extract text from PDF using PyMuPDF with basic LaTeX preservation"""
    try:
        doc = fitz.open(pdf)
        text = []
       
        for page in doc:
            # Extract text with basic formatting
            page_text = page.get_text(sort=True)
            text.append(page_text)
       
        doc.close()
        return "\n".join(text)
    except Exception as e:
        print(f"   PyMuPDF error: {e}")
        return ""
 
def ocr_pdf(pdf):
    pages = convert_from_path(pdf, dpi=400)  # higher DPI = better OCR
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
# Cleaning (preserve paragraphs for markdown)
# -----------------------
 
def clean(text):
    # Remove null characters
    text = text.replace("\x00", "")
   
    # Normalize whitespace within lines (but keep newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
   
    # Remove excessive blank lines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
   
    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
   
    return text.strip()
 
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
# Extract book title from filename
# -----------------------
 
def get_book_title(filepath):
    """Extract clean title from filename"""
    name = filepath.stem  # filename without extension
    # Remove common patterns like "(Z-Library)", author names in parens
    name = re.sub(r'\([^)]*\)', '', name)
    name = name.strip(' -_')
    return name
 
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
 
    # Get book title for the header
    title = get_book_title(file)
   
    # Create markdown content
    md_content = f"# {title}\n\n"
    md_content += f"**Source:** `{file.name}`\n\n"
    md_content += "---\n\n"
    md_content += text
 
    # Save as markdown
    out = OUT_DIR / f"book_{idx}.md"
    with out.open("w", encoding="utf-8") as f:
        f.write(md_content)
   
    print(f"   ✅ Saved: {out.name} ({len(text):,} chars)")
 
# -----------------------
# Run
# -----------------------
 
if __name__ == "__main__":
    files = list(DATA_DIR.rglob("*"))
    print(f"📚 Found {len(files)} files to process...\n")
 
    success = 0
    for i, f in enumerate(tqdm(files)):
        try:
            process(f, i)
            success += 1
        except Exception as e:
            print(f"❌ Failed: {f.name} - {e}")
 
    print(f"\n✅ Done! Processed {success} books → {OUT_DIR}/")