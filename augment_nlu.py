import os
from dotenv import load_dotenv
import google.generativeai as genai

# load .env
load_dotenv()

# ambil API key
API_KEY = os.getenv("GEMINI_API_KEY")

# konfigurasi Gemini
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

intent = "tanya_profesi"

examples = """
    - Selamat pagi kak. Saya Gledys. Izin bertanya, kalo saya mau lanjut [ners](prodi) apa saja yang harus dibayar? dan berapa biayanya?
    - Min untuk biaya profesi [apoteker](prodi) nya berapa ya persemester
    - Mau nanya perihal [PPG prajabatan](prodi)
    - Mau bertanya ka disini ada profesi [apoteker](prodi) gak ya?
    - saya sebelumnya baru aja lulus [S1](jalur_seleksi_jenjang) di swasta jurusan [keperawatan](prodi), apakah saya bisa melanjutkan profesi [ners](prodi) di PTN?
    - Waalaikumsalam permisi ka saya mau tanya, saya sebelumnya baru aja lulus [S1](jalur_seleksi_jenjang) di swasta jurusan [keperawatan](prodi), apakah saya bisa melanjutkan profesi [ners](prodi) di PTN?
    - Assalamualaikum warahmatullahi wa barakatuh. Selamat siang min, izin bertanya untuk prodi profesi [apoteker](prodi) apakah tahun ini di UIN buka? Terimakasih

"""

prompt = f"""
Kamu adalah generator dataset Rasa NLU bahasa Indonesia.

Tugas:
1. Tambahkan variasi kalimat baru hingga total 50 varian (jumlah dgn yg lama).
2. Parafrase natural seperti chat calon mahasiswa ke admin kampus.
3. Jangan mengubah makna intent.
4. Pertahankan format entity Rasa jika ada.
5. Jangan mengulang contoh yang sudah ada.
6. Gunakan bahasa informal alami.
7. Sesekali boleh typo ringan.
8. Variasikan panjang kalimat.
9. Output hanya list YAML examples tanpa penjelasan.

tambahan info : untuk tanya_profesi ini lebih ke nanya info umum profesi aja bisa ga soalnya kalo syarat biaya jadwal ada intent lainnya takutnya nanti bingung chatbotnya
terus profesi di uin cuman ada apoteker ppg ners dokter
Intent: {intent}

Contoh existing:
{examples}


"""

response = model.generate_content(prompt)

print(response.text)