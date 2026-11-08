import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = Path(__file__).parent / "vectorstores"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model_emb = SentenceTransformer(MODEL_NAME)


def search_by_intent(query, intent, top_k=8, min_score=0.0):
    intent_dir = VECTORSTORE_DIR / intent

    index_path = intent_dir / "index.faiss"
    chunks_path = intent_dir / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        print(f"Vectorstore untuk intent '{intent}' tidak ditemukan.")
        print(f"Dicari di: {intent_dir}")
        return []

    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    query_vector = model_emb.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        score = float(score)

        if score < min_score:
            continue

        chunk = chunks[idx].copy()
        chunk["score"] = score
        results.append(chunk)

    return results


if __name__ == "__main__":
    test_cases = [
        ("berapa biaya ukt sistem informasi", "tanya_biaya"),
        ("bagaimana alur pendaftaran mandiri", "tanya_prosedur_pendaftaran"),
        ("apa saja syarat beasiswa BLU", "tanya_persyaratan"),
        ("cara bayar pendaftaran SPMB mandiri", "cara_bayar"),
    ]

    for query, intent in test_cases:
        print("=" * 80)
        print("QUERY:", query)
        print("INTENT:", intent)

        results = search_by_intent(
            query=query,
            intent=intent,
            top_k=8,
            min_score=0.0,
        )

        if not results:
            print("Tidak ada hasil ditemukan.")
            continue

        for result in results:
            print("SCORE:", result["score"])
            print("SOURCE:", result.get("source_file"))
            print("HEADING:", result.get("heading"))
            print("TOPIK:", result.get("topik"))
            print("PARENT:", result.get("parent_context"))
            print("CONTENT:")
            print(result.get("content"))
            print("-" * 50)