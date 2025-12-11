import pandas as pd

# --- File input dan output ---
input_file = "Dataset_final_clean.xlsx"
output_file = "Dataset_labeled.xlsx"

# --- Baca dataset ---
df = pd.read_excel(input_file)
print(f"Jumlah data awal: {len(df)}")

# --- Pastikan kolom popularitas ada ---
if 'popularitas' not in df.columns:
    raise ValueError("Kolom 'popularitas' tidak ditemukan di dataset.")

# --- Tambahkan kolom label teks ---
threshold = 56
df['label'] = df['popularitas'].apply(lambda x: 'populer' if x >= threshold else 'tidak populer')

# --- Statistik hasil labeling ---
jumlah_populer = (df['label'] == 'populer').sum()
jumlah_tidak_populer = (df['label'] == 'tidak populer').sum()
print(f"Lagu populer (>= {threshold}): {jumlah_populer}")
print(f"Lagu tidak populer (< {threshold}): {jumlah_tidak_populer}")

# --- Simpan hasil ---
df.to_excel(output_file, index=False)
print(f"Dataset hasil labeling disimpan sebagai: {output_file}")