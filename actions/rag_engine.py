import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

# === PERBAIKAN IMPORT LOKAL ===
# Langsung panggil pakai titik (.) karena sudah satu folder di 'actions'
from .search_faiss import search_by_intent


# 1. Load variabel lingkungan
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Konfigurasi SDK
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di .env!")

genai.configure(api_key=GEMINI_API_KEY)


def generate_answer_with_gemini(query: str, intent: str) -> str:
    """
    Versi standalone:
    FAISS search dijalankan di sini.
    Cocok untuk testing tanpa Rasa.
    """
    results = search_by_intent(
        query=query,
        intent=intent,
        top_k=3,
        min_score=0.45
    )

    print("DEBUG GEMINI STANDALONE RESULTS:")
    print(results)

    if not results:
        return (
            "Mohon maaf 🙏 saya belum menemukan informasi yang spesifik mengenai hal tersebut. "
            "Bisa tolong jelaskan kembali pertanyaannya?"
        )

    context = build_context_from_results(results)

    print("CONTEXT SENT TO GEMINI:")
    print(context)

    if not context.strip():
        return (
            "Mohon maaf 🙏 saya belum menemukan informasi yang spesifik mengenai hal tersebut. "
            "Bisa tolong jelaskan kembali pertanyaannya?"
        )

    return _call_gemini(
        query=query,
        context=context,
        conversation_history=None
    )


def generate_answer_with_gemini_with_context(
    query: str,
    context: str,
    conversation_history=None
) -> str:
    """
    Versi terintegrasi dengan action.py:
    - FAISS search & reranking sudah dilakukan di action.py
    - Fungsi ini menerima context yang sudah siap
    - conversation_history opsional untuk multi-turn chat
    """
    print("CONTEXT RECEIVED FROM ACTION.PY:")
    print(context)

    print("CONVERSATION HISTORY RECEIVED FROM ACTION.PY:")
    print(conversation_history)

    if not context or not context.strip():
        return (
            "Mohon maaf 🙏 saya belum menemukan informasi yang spesifik mengenai hal tersebut. "
            "Bisa tolong jelaskan kembali pertanyaannya?"
        )

    return _call_gemini(
        query=query,
        context=context,
        conversation_history=conversation_history
    )


def build_context_from_results(results: list) -> str:
    """
    Ambil isi dokumen dari hasil FAISS/RAG secara fleksibel.
    Key hasil retrieval bisa berbeda-beda:
    content, text, answer, document, metadata.content, metadata.text.
    """
    contexts = []

    for i, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue

        topic = (
            result.get("topik")
            or result.get("topic")
            or result.get("title")
            or result.get("intent")
            or result.get("alias_topik")
            or "Informasi"
        )

        content = (
            result.get("content")
            or result.get("text")
            or result.get("answer")
            or result.get("document")
            or result.get("metadata", {}).get("content")
            or result.get("metadata", {}).get("text")
            or ""
        )

        if content and str(content).strip():
            contexts.append(
                f"[Konteks {i} - {topic}]\n{str(content).strip()}"
            )

    return "\n\n".join(contexts).strip()


def build_history_text(conversation_history=None) -> str:
    """
    Ubah conversation_history dari list menjadi teks.
    Aman jika None, string, atau list.
    """
    if not conversation_history:
        return ""

    if isinstance(conversation_history, str):
        return conversation_history.strip()

    if isinstance(conversation_history, list):
        history_parts = []

        for i, turn in enumerate(conversation_history, start=1):
            if turn and str(turn).strip():
                history_parts.append(
                    f"Percakapan sebelumnya [{i}]: {str(turn).strip()}"
                )

        return "\n".join(history_parts).strip()

    return str(conversation_history).strip()


def _call_gemini(
    query: str,
    context: str,
    conversation_history=None
) -> str:
    """
    Helper internal:
    Susun prompt dan panggil Gemini API.
    """
    history_text = build_history_text(conversation_history)

    history_section = ""
    if history_text:
        history_section = f"""
RIWAYAT PERCAKAPAN:
{history_text}
"""

    prompt = f"""
Kamu adalah asisten virtual resmi PMB UIN Syarif Hidayatullah Jakarta.

TUGAS:
- Jawab pertanyaan hanya berdasarkan KONTEKS di bawah.
- Gunakan RIWAYAT PERCAKAPAN hanya untuk memahami maksud pertanyaan lanjutan, bukan sebagai sumber fakta utama.
- Jika KONTEKS berisi informasi yang relevan dengan pertanyaan, jawab langsung dari KONTEKS tersebut.
- Jangan mengatakan informasi belum tersedia jika KONTEKS sudah memuat informasi yang berkaitan.
- Jika KONTEKS benar-benar kosong atau tidak relevan sama sekali, baru katakan bahwa informasi belum tersedia.
- Gunakan bahasa Indonesia yang formal namun ramah.
- Jawaban harus singkat, jelas, dan membantu.
- Langsung berikan jawaban, jangan awali dengan "Berdasarkan konteks...".
- KONTEKS hanya sebagai sumber fakta, bukan berarti semua isi konteks harus dijelaskan.
ATURAN RELEVANSI JAWABAN:
- Fokus hanya pada pertanyaan pengguna.
- Jawaban tidak boleh menambahkan detail yang tidak diminta pengguna.
- Jika pengguna hanya bertanya "apakah", "benar", "ada", "bisa", atau "menerima", jawab maksimal 1 kalimat.
- Untuk pertanyaan ya/tidak, jawab langsung dengan "Ya" atau "Tidak/Belum tersedia".
- Jangan menjelaskan alasan, syarat, ketentuan, prosedur, atau detail tambahan kecuali pengguna memintanya.
- Jangan mengambil semua informasi dari konteks; ambil hanya bagian yang menjawab pertanyaan.
- Jangan menyebut daftar beasiswa/program lain kecuali pengguna meminta daftar.
- Jika informasi tidak tersedia pada konteks, sebutkan topik yang ditanyakan lalu arahkan ke sumber resmi. Contoh format: "Informasi mengenai [sebutkan topik pertanyaan] belum tersedia. Untuk info lebih lanjut, silakan kunjungi https://uinjkt.ac.id atau hubungi WhatsApp UIN Jakarta (layanan 08.00–16.00 WIB) https://wa.me/+6285136908556"
INFORMASI TAMBAHAN
- NIM tidak sama dengan Nomor Peserta atau Nomor Pendaftaran


FORMAT:
- Untuk pertanyaan ya/tidak: jawab 1 kalimat saja.
- Untuk pertanyaan syarat/cara/prosedur: jawab ringkas dalam 2 sampai 4 poin.
- Untuk pertanyaan info umum: jawab 2 sampai 3 kalimat saja.

{history_section}
KONTEKS:
{context}

PERTANYAAN:
{query}

JAWABAN:
"""

    print("PROMPT SENT TO GEMINI:")
    print(prompt)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        if response and response.text:
            final_answer = response.text.strip()

            print("FINAL ANSWER FROM GEMINI:")
            print(final_answer)

            return final_answer

        return "Maaf, sistem tidak memberikan respon. Silakan coba lagi."

    except Exception as e:
        print(f"Error pada Gemini API: {e}")
        return "Mohon maaf, saat ini layanan informasi sedang mengalami gangguan teknis."