import json
import shutil
from pathlib import Path
from collections import defaultdict

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_PATH = BASE_DIR / "chunks" / "chunks.json"
VECTORSTORE_DIR = BASE_DIR / "vectorstores"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_CONTENT_CHARS = 50

model_emb = SentenceTransformer(MODEL_NAME)


def is_valid_chunk(chunk):
    content = chunk.get("content", "").strip()
    search_text = chunk.get("search_text", "").strip()
    intents = chunk.get("intents", [])

    if not intents:
        return False

    if len(content) < MIN_CONTENT_CHARS:
        return False

    if not search_text:
        return False

    return True


def clear_vectorstore_dir():
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    for item in VECTORSTORE_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except PermissionError:
            print(f"Skip karena masih terkunci: {item}")


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"File chunks tidak ditemukan: {CHUNKS_PATH}")

    clear_vectorstore_dir()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    chunks_by_intent = defaultdict(list)

    for chunk in chunks:
        if not is_valid_chunk(chunk):
            continue

        for intent in chunk.get("intents", []):
            chunks_by_intent[intent].append(chunk)

    if not chunks_by_intent:
        print("Tidak ada chunk valid yang memiliki intent.")
        return

    manifest = {
        "model": MODEL_NAME,
        "normalize_embeddings": True,
        "index_type": "IndexFlatIP",
        "embedding_field": "search_text",
        "total_chunks": len(chunks),
        "valid_intents": sorted(chunks_by_intent.keys()),
    }

    for intent, intent_chunks in chunks_by_intent.items():
        print(f"Membuat FAISS untuk intent: {intent}")

        texts = [chunk["search_text"] for chunk in intent_chunks]

        vectors = model_emb.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dim = vectors.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        intent_dir = VECTORSTORE_DIR / intent
        intent_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(intent_dir / "index.faiss"))

        with open(intent_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(intent_chunks, f, ensure_ascii=False, indent=2)

        manifest[intent] = {
            "chunks": len(intent_chunks),
            "dimension": dim,
        }

        print(f"  {len(intent_chunks)} chunk disimpan ke {intent_dir}")

    with open(VECTORSTORE_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Selesai build vectorstore.")
    print(f"Manifest disimpan di {VECTORSTORE_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()