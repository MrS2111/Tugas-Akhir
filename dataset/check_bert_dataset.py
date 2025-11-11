# ============================================================
#  CEK KUALITAS DAN KONSISTENSI DATASET BERT
#  (Sebelum digunakan untuk training XGBoost)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. BACA DATASET
# ------------------------------------------------------------
print("Membaca dataset utama...")
df = pd.read_pickle("Dataset_BERT_Embeddings.pkl")

print("\nKolom yang tersedia:")
print(df.columns.tolist())
print(f"\nJumlah total data: {len(df)}")

# ------------------------------------------------------------
# 2. CEK NILAI KOSONG
# ------------------------------------------------------------
print("\n🔍 Mengecek nilai kosong di setiap kolom:")
print(df.isnull().sum())

# Hapus baris tanpa embedding atau lirik kosong
df = df.dropna(subset=["bert_embedding", "lirik"]).reset_index(drop=True)
print(f"\nSetelah pembersihan: {len(df)} data valid tersisa.")

# ------------------------------------------------------------
# 3. CEK TIPE DATA DAN STRUKTUR
# ------------------------------------------------------------
print("\n📊 Tipe data tiap kolom:")
print(df.dtypes)

# Cek dimensi embedding secara acak
print("\nContoh dimensi embedding:")
print(np.array(df["bert_embedding"].iloc[0]).shape)

# ------------------------------------------------------------
# 4. CEK KONSISTENSI LABEL DAN SENTIMEN
# ------------------------------------------------------------
if "label" in df.columns:
    print("\n🏷️ Distribusi label (populer / tidak populer):")
    print(df["label"].value_counts())

if "sentimen_lirik" in df.columns:
    print("\n💬 Distribusi kategori sentimen:")
    print(df["sentimen_lirik"].value_counts())

# ------------------------------------------------------------
# 5. CEK NILAI POPULARITAS
# ------------------------------------------------------------
if "popularitas" in df.columns:
    print("\n🎶 Statistik nilai popularitas Spotify:")
    print(df["popularitas"].describe())

    plt.figure(figsize=(6, 4))
    sns.histplot(df["popularitas"], bins=20, kde=True, color="skyblue")
    plt.title("Distribusi Popularitas Lagu (Spotify)")
    plt.xlabel("Skor Popularitas (0–100)")
    plt.ylabel("Frekuensi")
    plt.show()

# ------------------------------------------------------------
# 6. CEK DISTRIBUSI SENTIMEN PER LABEL
# ------------------------------------------------------------
if "sentimen_lirik" in df.columns and "label" in df.columns:
    print("\n📈 Distribusi sentimen terhadap label popularitas:")
    cross_tab = pd.crosstab(df["sentimen_lirik"], df["label"], normalize="index") * 100
    print(cross_tab.round(2))

    cross_tab.plot(kind="bar", figsize=(7, 4), stacked=True, colormap="coolwarm")
    plt.title("Distribusi Sentimen per Kategori Popularitas")
    plt.ylabel("Persentase (%)")
    plt.xlabel("Kategori Sentimen")
    plt.legend(title="Label Popularitas")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7. CEK DUPLIKAT DATA
# ------------------------------------------------------------
dupes = df.duplicated(subset=["artis", "judul_lagu"]).sum()
print(f"\n📋 Jumlah data duplikat berdasarkan artis & judul lagu: {dupes}")

if dupes > 0:
    print("\nContoh duplikat:")
    print(df[df.duplicated(subset=["artis", "judul_lagu"], keep=False)][["artis", "judul_lagu"]].head())

# ------------------------------------------------------------
# 8. SIMPULAN RINGKAS
# ------------------------------------------------------------
print("\n🧾 Ringkasan dataset akhir:")
print(f"- Jumlah lagu: {len(df)}")
print(f"- Jumlah artis unik: {df['artis'].nunique()}")
print(f"- Jumlah label unik: {df['label'].nunique()}")
if 'sentimen_lirik' in df.columns:
    print(f"- Jumlah kategori sentimen: {df['sentimen_lirik'].nunique()}")
print("\n✅ Dataset siap dianalisis jika tidak ada nilai kosong dan distribusi label seimbang.")