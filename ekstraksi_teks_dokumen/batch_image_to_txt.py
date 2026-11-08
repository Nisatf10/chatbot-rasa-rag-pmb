# from pathlib import Path
# from pdf2image import convert_from_path
# from PIL import Image
# import pytesseract

# # Folder input dan output
# INPUT_DIR = Path("data_raw/img")
# OUTPUT_DIR = Path("data_txt")
# OUTPUT_DIR.mkdir(exist_ok=True)

# # Bahasa OCR: "ind" untuk Indonesia, "eng" untuk Inggris
# OCR_LANG = "ind+eng"

# def ocr_image(image_path):
#     image = Image.open(image_path)
#     text = pytesseract.image_to_string(image, lang=OCR_LANG)
#     return text

# def ocr_pdf(pdf_path):
#     pages = convert_from_path(pdf_path, dpi=300)
#     all_text = []

#     for i, page in enumerate(pages, start=1):
#         text = pytesseract.image_to_string(page, lang=OCR_LANG)
#         all_text.append(f"\n\n===== Halaman {i} =====\n\n{text}")

#     return "\n".join(all_text)

# def clean_text(text):
#     lines = text.splitlines()
#     cleaned_lines = []

#     for line in lines:
#         line = line.strip()
#         if line:
#             cleaned_lines.append(line)

#     return "\n".join(cleaned_lines)

# for file_path in INPUT_DIR.iterdir():
#     if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
#         text = ocr_image(file_path)

#     elif file_path.suffix.lower() == ".pdf":
#         text = ocr_pdf(file_path)

#     else:
#         continue

#     text = clean_text(text)

#     output_file = OUTPUT_DIR / f"{file_path.stem}.txt"
#     output_file.write_text(text, encoding="utf-8")

#     print(f"Selesai: {file_path.name} -> {output_file.name}")