import fitz  # PyMuPDF
from pathlib import Path
import re

INPUT_DIR = Path("data_raw/pdf")
OUTPUT_DIR = Path("data_txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def clean_text(text: str) -> str:
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


pdf_files = list(INPUT_DIR.glob("*.pdf"))

if not pdf_files:
    print(f"Tidak ada file PDF di folder {INPUT_DIR}")
    raise SystemExit

for pdf_path in pdf_files:
    print(f"Memproses PDF ke TXT: {pdf_path.name}")

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_text(text)

        all_text.append(f"[Halaman {page_num}]\n{text}")

    output_path = OUTPUT_DIR / f"{clean_filename(pdf_path.stem)}.txt"
    output_path.write_text("\n\n".join(all_text), encoding="utf-8")

    print(f"Berhasil dibuat: {output_path}")

print("Selesai convert semua PDF ke TXT.")
