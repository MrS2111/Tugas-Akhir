# lyric_scraper.py
import pandas as pd
import lyricsgenius
import time
import os
import re

# === Konfigurasi ===
ACCESS_TOKEN = "bRawa4_y8bq_XGdxewxXGtPU4RlPniRHTL7uKJ8Xpzv3VFrja7jpbNzxS1iXIDM1"
INPUT_FILE = "top_10_artists_songs_with_the_most_monthly_listeners_spotify_Filtered.xlsx"
OUTPUT_FILE = "spotify_pop_music_dengan_lirik.xlsx"
SLEEP_TIME = 2  # detik antar permintaan ke API Genius

# === Fungsi pembersihan judul lagu (untuk pencarian) ===
def bersihkan_judul(judul):
    judul = re.sub(r'\(.*?\)', '', judul)
    judul = re.sub(r'\[.*?\]', '', judul)
    judul = re.sub(r'[-–_]', ' ', judul)
    judul = re.sub(r'feat\..*', '', judul, flags=re.IGNORECASE)
    judul = re.sub(r'ft\..*', '', judul, flags=re.IGNORECASE)
    judul = re.sub(r'[^\w\s]', '', judul)
    return judul.strip()

# === Fungsi ambil lirik dari Genius ===
def ambil_lirik_genius(judul, artis):
    try:
        genius = lyricsgenius.Genius(
            ACCESS_TOKEN,
            timeout=10,
            retries=3,
            verbose=False,
        )
        song = genius.search_song(title=judul, artist=artis)
        if song and song.lyrics:
            return song.lyrics.strip()
        else:
            return "Lirik tidak ditemukan"
    except Exception as e:
        print(f"⚠️ Gagal ambil lirik {judul} - {artis}: {e}")
        return "Error"

# === Load dataset Spotify ===
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"File {INPUT_FILE} tidak ditemukan.")

df = pd.read_excel(INPUT_FILE)

# Pastikan kolom utama ada
required_cols = ["artis", "judul_lagu"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom {col} tidak ditemukan dalam file input!")

# Tambah kolom lirik jika belum ada
if "lirik" not in df.columns:
    df["lirik"] = ""

# === Proses ambil lirik ===
genius = lyricsgenius.Genius(
    ACCESS_TOKEN,
    timeout=10,
    retries=3,
    verbose=False,
)

seen_titles = set()

for idx, row in df.iterrows():
    artis = str(row["artis"]).strip()
    judul_asli = str(row["judul_lagu"]).strip()
    judul_bersih = bersihkan_judul(judul_asli).lower()

    # Skip jika judul sudah diproses atau kosong
    if not judul_bersih or judul_bersih in seen_titles:
        print(f"🔁 Melewati duplikat/non-valid: {judul_asli}")
        continue

    # Skip jika sudah ada lirik valid
    if isinstance(row["lirik"], str) and row["lirik"].strip() not in ["", "Lirik tidak ditemukan", "Error"]:
        print(f"✅ Lirik sudah tersedia untuk: {judul_asli}")
        seen_titles.add(judul_bersih)
        continue

    # Ambil lirik
    print(f"🎵 Mengambil lirik: {judul_asli} - {artis} ...")
    lirik = ambil_lirik_genius(judul_bersih, artis)
    df.at[idx, "lirik"] = lirik
    seen_titles.add(judul_bersih)

    # Hindari rate limiting
    time.sleep(SLEEP_TIME)

# === Simpan hasil ===
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nProses selesai! Lirik disimpan ke: {OUTPUT_FILE}")
print(f"Total lagu dengan lirik: {df['lirik'].apply(lambda x: x not in ['', 'Lirik tidak ditemukan', 'Error']).sum()}/{len(df)}")
