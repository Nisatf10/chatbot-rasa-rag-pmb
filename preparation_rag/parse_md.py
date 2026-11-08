import re
import json
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_MD_DIR = PROJECT_ROOT / "dokumen_pmb" / "data_md_tagged"
OUTPUT_DIR = BASE_DIR / "chunks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONTENT_CHARS = 80


def clean_text(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_meta(section):
    intents_match = re.search(r"<!--\s*intents:\s*(.*?)\s*-->", section)
    topik_match = re.search(r"<!--\s*topik:\s*(.*?)\s*-->", section)
    alias_match = re.search(r"<!--\s*alias_topik:\s*(.*?)\s*-->", section)

    intents = []
    if intents_match:
        intents = [x.strip() for x in intents_match.group(1).split(",") if x.strip()]

    alias_topik = []
    if alias_match:
        alias_topik = [x.strip() for x in alias_match.group(1).split(",") if x.strip()]

    topik = topik_match.group(1).strip() if topik_match else None

    return intents, topik, alias_topik


def is_header_only(content):
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) <= 1 and lines[0].startswith("#"):
        return True

    non_heading = [line for line in lines if not line.startswith("#")]
    joined = " ".join(non_heading).strip()

    return len(joined) < MIN_CONTENT_CHARS


def make_chunk_id(source_file, title, content):
    raw = f"{source_file}|{title}|{content[:120]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def parse_file(path):
    text = path.read_text(encoding="utf-8")

    sections = re.split(r"(?=^#{1,6}\s+.+$)", text, flags=re.MULTILINE)
    chunks = []

    current_h1 = None
    current_h2 = None
    current_h3 = None

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading = section.splitlines()[0].strip()
        level = len(re.match(r"^(#+)", heading).group(1))
        heading_text = re.sub(r"^#{1,6}\s*", "", heading).strip()

        if level == 1:
            current_h1 = heading_text
            current_h2 = None
            current_h3 = None
        elif level == 2:
            current_h2 = heading_text
            current_h3 = None
        elif level == 3:
            current_h3 = heading_text

        intents, topik, alias_topik = extract_meta(section)
        clean_content = clean_text(section)

        if is_header_only(clean_content):
            continue

        parent_context = " > ".join(
            [x for x in [current_h1, current_h2, current_h3] if x]
        )

        title = topik or heading_text

        search_text = "\n".join(
            [
                f"Judul: {title}",
                f"Konteks: {parent_context}",
                f"Alias: {', '.join(alias_topik)}" if alias_topik else "",
                f"Intent: {', '.join(intents)}" if intents else "",
                "",
                clean_content,
            ]
        ).strip()

        answer = clean_content

        chunk = {
            "id": make_chunk_id(path.name, title, clean_content),
            "source_file": path.name,
            "heading": heading,
            "parent_context": parent_context,
            "intents": intents,
            "topik": title,
            "alias_topik": alias_topik,
            "search_text": search_text,
            "content": answer,
        }

        chunks.append(chunk)

    return chunks


all_chunks = []
seen = set()

if not DATA_MD_DIR.exists():
    raise FileNotFoundError(f"Folder tidak ditemukan: {DATA_MD_DIR}")

for md_file in DATA_MD_DIR.glob("*.md"):
    for chunk in parse_file(md_file):
        dedup_key = re.sub(r"\s+", " ", chunk["content"]).lower().strip()

        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        all_chunks.append(chunk)

output_path = OUTPUT_DIR / "chunks.json"
output_path.write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"Berhasil membuat {output_path}")
print(f"Jumlah chunk: {len(all_chunks)}")