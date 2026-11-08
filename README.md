# Chatbot Layanan Informasi PMB — Rasa + RAG

Chatbot layanan informasi **Penerimaan Mahasiswa Baru (PMB) UIN Syarif Hidayatullah Jakarta**, dibangun dengan framework **Rasa** yang diintegrasikan dengan **Retrieval-Augmented Generation (RAG)**. Pengembangan mengikuti metode **CRISP-DM**.

Bot menjawab pertanyaan calon mahasiswa seputar jalur masuk, jadwal, biaya/UKT, beasiswa, persyaratan, cara pembayaran, program pascasarjana dan profesi, serta kontak PMB, melalui **Telegram**.

---

## Arsitektur

```
Pengguna (Telegram)
        │
        ▼
     Caddy (HTTPS reverse proxy)
        │
        ▼
     Rasa Server ──► NLU: klasifikasi intent
        │
        ▼
     Action Server (custom actions)
        │
        ├─► FAISS: cari potongan dokumen yang relevan, per intent
        └─► Google Gemini: susun jawaban dari dokumen yang ditemukan
```

| Komponen | Teknologi |
|---|---|
| NLU & dialog | Rasa 3.6 |
| Vector store | FAISS (satu indeks per intent) |
| LLM | Google Gemini |
| Kanal | Telegram |
| Deployment | Docker Compose + Caddy |

---

## Struktur Folder

```
.
├── actions/                  # Action server Rasa
│   ├── actions.py            # Custom actions
│   ├── rag_engine.py         # Pemanggilan Gemini (RAG)
│   ├── search_faiss.py       # Pencarian FAISS per intent
│   ├── vectorstores/         # Indeks FAISS + chunks per intent
│   ├── Dockerfile
│   └── requirements.txt
├── data/                     # Data latih Rasa (nlu, rules, stories)
├── dokumen_raw/              # Dokumen sumber asli (PDF, DOCX, gambar)
├── ekstraksi_teks_dokumen/   # Skrip ekstraksi teks dari PDF/DOCX/gambar
├── dokumen_pmb/              # Hasil ekstraksi: txt → txt bersih → markdown ber-tag
├── preparation_rag/          # Parsing markdown, chunking, dan pembuatan indeks FAISS
├── models/                   # Model Rasa hasil training
├── augment_nlu.py            # Augmentasi data NLU
├── telegram_patched.py       # Patch kanal Telegram Rasa (event loop aiogram)
├── config.yml                # Pipeline & policy Rasa
├── domain.yml
├── endpoints.yml
├── docker-compose.yml
├── Caddyfile
├── credentials.example.yml   # Contoh konfigurasi kanal (salin → credentials.yml)
└── .env.example              # Contoh environment variable (salin → .env)
```

### Intent yang didukung RAG

`cara_bayar`, `daftar_ulang`, `tanya_beasiswa`, `tanya_biaya`, `tanya_jadwal`, `tanya_jalur`, `tanya_kontak`, `tanya_kuota`, `tanya_link`, `tanya_pascasarjana`, `tanya_persyaratan`, `tanya_prodi_fakultas`, `tanya_profesi`, `tanya_prosedur_pendaftaran`, `tanya_ujian`

---

## Alur Pengolahan Dokumen

1. **`dokumen_raw/`**: dokumen resmi PMB dalam bentuk PDF, DOCX, dan gambar.
2. **`ekstraksi_teks_dokumen/`**: ekstraksi teks (`batch_pdf_to_txt.py`, `batch_docx_to_txt.py`, `batch_image_to_txt.py`) ke `dokumen_pmb/data_txt/`.
3. **`dokumen_pmb/data_txt_clean/`**: teks yang sudah dibersihkan.
4. **`dokumen_pmb/data_md_tagged/`**: teks dalam format markdown yang diberi tag intent.
5. **`preparation_rag/`**: `parse_md.py` memecah dokumen menjadi chunk (`chunks/chunks.json`), lalu `build_faiss.py` membuat indeks FAISS ke `actions/vectorstores/`.

---

## Persiapan

### Prasyarat
- Python **3.10** (Rasa 3.6 belum mendukung Python 3.11 ke atas)
- API key **Google Gemini** dari [Google AI Studio](https://aistudio.google.com/)
- Token bot **Telegram** dari [@BotFather](https://t.me/BotFather)
- Docker & Docker Compose (untuk deployment)

### Konfigurasi rahasia

File rahasia **tidak** disertakan di repo. Salin dari contohnya lalu isi:

```bash
cp .env.example .env
cp credentials.example.yml credentials.yml
```

- `.env`: isi `GEMINI_API_KEY`
- `credentials.yml`: isi token bot Telegram, nama bot, dan URL webhook

> Jangan pernah meng-commit `.env` atau `credentials.yml`. Keduanya sudah tercantum di `.gitignore`.

---

## Menjalankan Secara Lokal

```bash
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# source venv/bin/activate        # Linux / macOS

pip install -r requirements.txt
pip install -r actions/requirements.txt
```

`endpoints.yml` diatur untuk Docker (`http://action_server:5055/webhook`). Untuk menjalankan tanpa Docker, ubah sementara menjadi:

```yaml
action_endpoint:
  url: "http://localhost:5055/webhook"
```

Lalu jalankan di dua terminal terpisah:

```bash
# Terminal 1: action server
rasa run actions

# Terminal 2: uji lewat terminal
rasa shell
```

Untuk melatih ulang model:

```bash
rasa train
```

---

## Deployment dengan Docker

1. Arahkan domain ke IP server, lalu sesuaikan domain di `Caddyfile`.
2. Pastikan `.env` dan `credentials.yml` sudah ada di server (`webhook_url` mengarah ke domain tersebut).
3. Jalankan:

```bash
docker compose up -d --build
```

Layanan yang berjalan:
- `caddy`: HTTPS dan reverse proxy (port 80/443)
- `rasa`: Rasa server (port internal 5005)
- `action_server`: custom actions + RAG (port internal 5055)

---

## Evaluasi

Proyek ini dievaluasi dengan:
- **Evaluasi NLU** Rasa untuk klasifikasi intent
- **RAGAS** untuk kualitas jawaban RAG

Data uji dan hasil evaluasi tidak disertakan di repositori ini.

---

## Penulis

**Annisa Tasya Ferina** ([@Nisatf10](https://github.com/Nisatf10))
