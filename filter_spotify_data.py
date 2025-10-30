# filter_spotify_data_strict_v3.py
import pandas as pd
import re

# --- File input dan output ---
input_file = "top_10_artists_songs_with_the_most_monthly_listeners_spotify.xlsx"
output_file = "top_10_artists_songs_with_the_most_monthly_listeners_spotify_Filtered.xlsx"

# --- Baca file ---
df = pd.read_excel(input_file)
print(f"Jumlah data awal: {len(df)}")

# --- Fungsi pembersihan judul lagu ---
def clean_title(title):
    title = str(title).lower()
    title = re.sub(r'\(.*?\)', '', title)
    title = re.sub(r'\[.*?\]', '', title)
    title = re.sub(r'[-–_]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

df['judul_lagu_bersih'] = df['judul_lagu'].apply(clean_title)

# --- Pola regex super ketat termasuk slowed, acapella, speed up ---
pattern_non_original = (
    r'(?:remix|mix|club|dub|edit|rework|live|acoustic|instrumental|version|'
    r'remaster|re-recorded|remake|radio|extended|bootleg|session|dj|karaoke|cover|performance|'
    r'spotify singles|slowed|speed up|sped up|acapella|feat\.|featuring)'
)

# --- Pola judul dengan strip diikuti teks non-original ---
pattern_dash_suffix = (
    r'\s*-\s*.*?(?:remix|mix|club|dub|edit|rework|live|acoustic|instrumental|version|'
    r'remaster|re-recorded|remake|radio|bootleg|session|dj|karaoke|cover|performance|'
    r'slowed|speed up|sped up|acapella)\b'
)

# --- Gabungkan dua filter ---
mask_non_original = (
    df['judul_lagu_bersih'].str.contains(pattern_non_original, case=False, regex=True) |
    df['judul_lagu'].str.contains(pattern_dash_suffix, case=False, regex=True)
)

filtered_df = df[~mask_non_original]

# --- Hapus duplikat berdasarkan artis + judul bersih ---
filtered_df = filtered_df.drop_duplicates(subset=['artis', 'judul_lagu_bersih'], keep='first')

# --- Pertahankan semua kolom relevan (termasuk popularitas) ---
if 'popularitas' in filtered_df.columns:
    kolom_dipertahankan = [col for col in filtered_df.columns if col != 'judul_lagu_bersih']
    filtered_df = filtered_df[kolom_dipertahankan]
else:
    print("Kolom 'popularitas' tidak ditemukan. Semua kolom selain 'judul_lagu_bersih' akan disimpan.")
    filtered_df = filtered_df.drop(columns=['judul_lagu_bersih'], errors='ignore')

# --- Simpan hasil ---
filtered_df.to_excel(output_file, index=False)
print(f"Selesai! Jumlah data akhir: {len(filtered_df)}")
print(f"File tersimpan sebagai: {output_file}")

# --- Tampilkan contoh lagu yang difilter ---
removed_examples = df[mask_non_original].head(10)
print("\nContoh lagu yang difilter:")
print(removed_examples[['artis', 'judul_lagu']])