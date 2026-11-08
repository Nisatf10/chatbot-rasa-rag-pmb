from pathlib import Path
import mammoth
import re

INPUT_DIR = Path("data_raw/docx")
OUTPUT_DIR = Path("data_txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def clean_text(text: str) -> str:
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


docx_files = list(INPUT_DIR.glob("*.docx"))

if not docx_files:
    print(f"Tidak ada file DOCX di folder {INPUT_DIR}")
    raise SystemExit

for docx_path in docx_files:
    print(f"Memproses DOCX ke TXT: {docx_path.name}")

    with open(docx_path, "rb") as docx_file:
        result = mammoth.extract_raw_text(docx_file)
        text = clean_text(result.value)

    output_path = OUTPUT_DIR / f"{clean_filename(docx_path.stem)}.txt"
    output_path.write_text(text, encoding="utf-8")

    print(f"Berhasil dibuat: {output_path}")

print("Selesai convert semua DOCX ke TXT.")
