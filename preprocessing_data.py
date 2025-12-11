import pandas as pd
import re
import string

# ==============================
# 1. FILTER SPOTIFY DATA
# ==============================

def filter_spotify(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Jumlah data awal: {len(df)}")

    # Pola lagu non-orisinal di judul
    pattern_non_original = (
        r'(?:remix|mix|club|dub|edit|rework|live|acoustic|instrumental|version|'
        r'remaster|re-recorded|remake|radio|extended|bootleg|session|dj|karaoke|cover|performance|'
        r'spotify singles|slowed|speed up|sped up|acapella|feat\.|featuring)'
    )

    pattern_dash_suffix = (
        r'\s*-\s*.*?(?:remix|mix|club|dub|edit|rework|live|acoustic|instrumental|version|'
        r'remaster|re-recorded|remake|radio|bootleg|session|dj|karaoke|cover|performance|'
        r'slowed|speed up|sped up|acapella)\b'
    )

    # Deteksi judul yang non-orisinal
    mask_non_original = (
        df['judul_lagu'].astype(str).str.contains(pattern_non_original, case=False, regex=True) |
        df['judul_lagu'].astype(str).str.contains(pattern_dash_suffix, case=False, regex=True)
    )

    # Buang lagu non-orisinal
    filtered_df = df[~mask_non_original].copy()
    print(f"Jumlah data setelah menghapus versi non-orisinal: {len(filtered_df)}")

    # Normalisasi judul lagu untuk hapus duplikat
    def clean_title_for_filter(title):
        title = str(title).lower()
        title = re.sub(r'\(.*?\)', '', title)      # hapus teks dalam ()
        title = re.sub(r'\[.*?\]', '', title)      # hapus teks dalam []
        title = re.sub(r'[-–_]', ' ', title)       # ganti tanda hubung dengan spasi
        title = re.sub(r'\s+', ' ', title).strip() # hapus spasi berlebih
        return title

    filtered_df['judul_lagu_bersih'] = filtered_df['judul_lagu'].apply(clean_title_for_filter)

    # Hapus duplikat artis + judul_lagu_bersih
    filtered_df = filtered_df.drop_duplicates(subset=['artis', 'judul_lagu_bersih'], keep='first')
    print(f"Jumlah data setelah menghapus duplikat: {len(filtered_df)}")

    # Buang kolom sementara
    filtered_df = filtered_df.drop(columns=['judul_lagu_bersih'], errors='ignore')

    return filtered_df

# ==============================
# 2. PREPROCESS LIRIK
# ==============================

def preprocess_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1) Mulai dari tag struktur pertama [Intro]/[Verse]/[Chorus]/dst
    match = re.search(
        r"\[(intro|verse|chorus|pre-chorus|refrain|hook|bridge|outro)[^\]]*\]",
        text,
        re.IGNORECASE
    )
    if match:
        text = text[match.start():]

    # 2) Hapus seluruh blok Chorus / Pre-Chorus / Refrain / Hook
    #    Blok didefinisikan dari tag [Chorus ...] sampai sebelum tag berikutnya atau akhir teks
    text = re.sub(
        r"\[ *?(chorus|chorus \d+|pre-chorus|refrain|hook)[^\]]*\](.*?)(?=\n\[|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 3) Hapus semua tag struktur yang tersisa, misalnya [Intro: ...], [Verse: ...], [Bridge: ...], [Outro: ...]
    text = re.sub(r"\[[^\]]*\]", "", text)

    # 4) Hapus sisa teks iklan/elemen dari Genius (read more, embed, translations, dll)
    text = re.sub(
        r"(you might also like|embed|translations|\d+embed|read\s*more)",
        "",
        text,
        flags=re.IGNORECASE
    )

    # ================= Lanjutan preprocessing biasa =================

    # 5) Lowercase
    text = text.lower()

    # 6) Hapus angka
    text = re.sub(r"\d+", " ", text)

    # 7) Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 8) Hapus karakter non-ASCII
    text = text.encode('ascii', 'ignore').decode()

    # 9) Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Pola pembersihan judul (versi preprocessing lirik)
title_patterns_pre = [
    r' - Radio Edit',
    r' - Live in.*',
    r' - .*Remix',
    r'-\s*Bonus.*',
    r'feat\..*',
    r'ft\..*',
    r'[^\w\s]',
]

def clean_title_for_lyrics(title: str) -> str:
    title = str(title)
    for pattern in title_patterns_pre:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    return title.strip().lower()


def preprocess_lyrics_df(df: pd.DataFrame) -> pd.DataFrame:
    # Pastikan kolom lirik ada
    if 'lirik' not in df.columns:
        raise ValueError("Kolom 'lirik' tidak ditemukan di dataset.")

    # Hapus baris dengan lirik error / tidak ditemukan (kalau ada teks seperti itu)
    df = df[~df['lirik'].astype(str).str.contains("Lirik tidak ditemukan", case=False, na=False)]
    df = df[~df['lirik'].astype(str).str.contains("error", case=False, na=False)]

    # Terapkan preprocessing ke kolom lirik
    print("Melakukan preprocessing pada kolom lirik...")
    df['lirik'] = df['lirik'].astype(str).apply(preprocess_lyrics)

    # Bersihkan judul lagi untuk jaga-jaga duplikat
    df['cleaned_title'] = df['judul_lagu'].astype(str).apply(clean_title_for_lyrics)
    # gunakan kombinasi artis + cleaned_title, supaya judul sama beda artis tidak hilang
    df = df.drop_duplicates(subset=['artis', 'cleaned_title'], keep='first')
    df = df.drop(columns=['cleaned_title'])

    # Hapus lirik kosong / terlalu pendek
    df = df[df['lirik'].str.strip() != ""]
    df = df[df['lirik'].str.split().str.len() > 10]

    print(f"Total lagu setelah preprocessing lirik: {len(df)}")
    return df


# ==============================
# 3. LABELING POPULARITAS
# ==============================

def add_popularity_label(df: pd.DataFrame, threshold: int = 55) -> pd.DataFrame:
    if 'popularitas' not in df.columns:
        raise ValueError("Kolom 'popularitas' tidak ditemukan di dataset.")

    # 1 = populer, 0 = tidak populer
    df['label'] = df['popularitas'].apply(
        lambda x: 1 if x >= threshold else 0
    )

    jumlah_populer = (df['label'] == 1).sum()
    jumlah_tidak_populer = (df['label'] == 0).sum()

    print(f"Lagu populer (>= {threshold}): {jumlah_populer}")
    print(f"Lagu tidak populer (< {threshold}): {jumlah_tidak_populer}")

    return df


# ==============================
# 4. MAIN PIPELINE
# ==============================

def main():
    INPUT_FILE = "spotify_pop_music_dengan_lirik.xlsx" 
    OUTPUT_FILE = "Dataset_labeled.xlsx"

    # Load dataset awal (gabungan metadata + lirik)
    df = pd.read_excel(INPUT_FILE)
    print(f"Dataset awal: {len(df)} baris")

    # Step 1: Filter lagu non-orisinal + hapus duplikat judul
    df = filter_spotify(df)

    # Step 2: Preprocess kolom lirik
    df = preprocess_lyrics_df(df)

    # Step 3: Labeling berdasarkan popularitas
    df = add_popularity_label(df, threshold=55)

    # Simpan hasil akhir
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nSelesai! Dataset final tersimpan sebagai: {OUTPUT_FILE}")
    print(f"Total baris akhir: {len(df)}")


if __name__ == "__main__":
    main()