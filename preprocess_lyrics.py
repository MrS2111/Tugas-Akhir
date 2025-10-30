import pandas as pd
import re
import string

# === Fungsi Preprocessing Lirik ===
def preprocess_lyrics_final(text):
    if not isinstance(text, str):
        return ""

    # Jika ada bagian [Verse], mulai dari sana
    match = re.search(r"\[(intro|verse|pre-chorus|chorus|refrain|hook|bridge|outro)[^\]]*\]", text, re.IGNORECASE)
    if match:
        text = text[match.start():]

    # Potong jika ada "read more"
    text = re.split(r"read\s*more", text, flags=re.IGNORECASE)[0]

    # Hapus tag seperti [Chorus], [Verse], dll.
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Hapus sisa teks iklan Genius
    text = re.sub(r"(you might also like|embed|translations|\d+embed)", "", text, flags=re.IGNORECASE)

    # Lowercase
    text = text.lower()

    # Hapus angka
    text = re.sub(r"\d+", "", text)

    # Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Hapus karakter non-ASCII
    text = text.encode('ascii', 'ignore').decode()

    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    return text


# === Pembersihan Judul Lagu ===
title_patterns = [
    r' - Radio Edit', 
    r' - Live in.*', 
    r' - .*Remix', 
    r'-\s*Bonus.*', 
    r'feat\..*', 
    r'ft\..*', 
    r'[^\w\s]', 
]

def clean_title(title):
    for pattern in title_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    return title.strip().lower()


# === Load Dataset ===
file_path = 'spotify_pop_music_dengan_lirik.xlsx'
df = pd.read_excel(file_path)

# Hapus lirik error / tidak ditemukan
df = df[
    (~df['lirik'].str.contains("Lirik tidak ditemukan", case=False, na=True)) &
    (~df['lirik'].str.contains("error", case=False, na=True))
].copy()

# Preprocessing lirik
df['lirik'] = df['lirik'].apply(preprocess_lyrics_final)

# Bersihkan judul dan hapus duplikat
df['cleaned_title'] = df['judul_lagu'].astype(str).apply(clean_title)
df = df.drop_duplicates(subset=['cleaned_title'], keep='first')
df = df.drop(columns=['cleaned_title'])

# Hapus baris dengan lirik kosong atau terlalu pendek
df = df[df['lirik'].str.strip() != ""]
df = df[df['lirik'].str.split().str.len() > 10]

# Simpan hasil
output_file = 'Dataset_final_clean.xlsx'
df.to_excel(output_file, index=False)

print(f"Proses selesai! File tersimpan sebagai: {output_file}")
print(f"Total lagu akhir: {len(df)}")