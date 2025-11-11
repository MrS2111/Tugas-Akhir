import pandas as pd
import lyricsgenius
import time

# === Konfigurasi ===
ACCESS_TOKEN = "bRawa4_y8bq_XGdxewxXGtPU4RlPniRHTL7uKJ8Xpzv3VFrja7jpbNzxS1iXIDM1"
INPUT_FILE = "top_10_artists_songs_with_the_most_monthly_listeners_spotify_Filtered.xlsx"
OUTPUT_FILE = "spotify_pop_music_dengan_lirik.xlsx"
SLEEP_TIME = 2  # detik antar permintaan ke API Genius

# === Inisialisasi Genius API ===
genius = lyricsgenius.Genius(
    ACCESS_TOKEN,
    timeout=10,
    retries=3,
    verbose=False,
)

# === Fungsi ambil lirik dari Genius ===
def ambil_lirik_genius(judul, artis):
    try:
        song = genius.search_song(title=judul, artist=artis)
        if song and song.lyrics:
            return song.lyrics.strip()
        else:
            return "Lirik tidak ditemukan"
    except Exception as e:
        print(f"Gagal ambil lirik {judul} - {artis}: {e}")
        return "Error"

# === Load dataset ===
df = pd.read_excel(INPUT_FILE)

# Tambah kolom 'lirik' jika belum ada
if "lirik" not in df.columns:
    df["lirik"] = ""

# === Ambil lirik untuk setiap lagu ===
for idx, row in df.iterrows():
    artis = str(row["artis"]).strip()
    judul = str(row["judul_lagu"]).strip()

    print(f"Mengambil lirik: {judul} - {artis} ...")
    lirik = ambil_lirik_genius(judul, artis)
    df.at[idx, "lirik"] = lirik

    # Hindari rate limiting
    time.sleep(SLEEP_TIME)

# === Simpan hasil ke file ===
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nProses selesai! Lirik disimpan ke: {OUTPUT_FILE}")