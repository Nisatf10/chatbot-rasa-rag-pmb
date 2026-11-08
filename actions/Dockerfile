# Pakai image bawaan Rasa SDK versi 3.6.2 yang persis dengan laptopmu
FROM rasa/rasa-sdk:3.6.2

# Set folder kerja di dalam server nanti
WORKDIR /app

# Pindah jadi 'admin' server sementara buat nginstall library
USER root

# Bawa daftar library (requirements.txt) ke dalam server
COPY requirements.txt ./

# Install semua library RAG kamu (Gemini, FAISS, dkk)
RUN pip install --no-cache-dir -r requirements.txt

# Bawa semua file kodingan di folder actions kamu ke dalam server
COPY . /app/actions

# Balikin aksesnya ke user normal Rasa biar sistemnya aman
USER 1001

# TAMBAHKAN BARIS INI:
ENTRYPOINT ["python", "-m", "rasa_sdk", "--actions", "actions"]
