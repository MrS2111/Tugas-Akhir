import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_excel("Dataset_labeled.xlsx")

print("Kolom dalam dataset:")
print(df.columns)
print("\nJumlah total data:", len(df))


# ============================================================
# 2. TAMBAH KOLOM PANJANG LIRIK (JUMLAH KATA)
# ============================================================

df["lyric_length_words"] = df["lirik"].astype(str).str.split().str.len()

print("\nContoh data:")
print(df[["judul_lagu", "label", "lyric_length_words"]].head())


# ============================================================
# 3. DISTRIBUSI KATEGORI POPULER vs TIDAK POPULER
# ============================================================

label_map = {0: "Tidak Populer", 1: "Populer"}
df["label_name"] = df["label"].map(label_map)

label_counts = df["label_name"].value_counts()
print("\nDistribusi kategori:")
print(label_counts)

# --- Plot Bar Chart ---
plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
plt.title("Distribusi Lagu Populer vs Tidak Populer")
plt.xlabel("Kategori")
plt.ylabel("Jumlah Lagu")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 4. DISTRIBUSI POPULARITAS (NILAI POPULARITAS)
# ============================================================

print("\nStatistik Popularitas:")
print(df["popularitas"].describe())

# Plot histogram distribusi popularitas
plt.figure(figsize=(7, 4))
plt.hist(df["popularitas"], bins=30, color="#9467bd")
plt.title("Distribusi Nilai Popularitas Lagu")
plt.xlabel("Nilai Popularitas (0–100)")
plt.ylabel("Jumlah Lagu")
plt.tight_layout()
plt.show()


# ============================================================
# 5. DISTRIBUSI PANJANG LIRIK (HISTOGRAM)
# ============================================================

plt.figure(figsize=(7, 4))
plt.hist(df["lyric_length_words"], bins=30, color="#2ca02c")
plt.title("Distribusi Panjang Lirik (Jumlah Kata)")
plt.xlabel("Jumlah Kata")
plt.ylabel("Jumlah Lagu")
plt.tight_layout()
plt.show()


# ============================================================
# 6. DISTRIBUSI PANJANG LIRIK PER KELAS (POPULER vs TIDAK POPULER)
# ============================================================

populer = df[df["label"] == 1]["lyric_length_words"]
tidak_populer = df[df["label"] == 0]["lyric_length_words"]

plt.figure(figsize=(7, 4))
plt.hist(populer, bins=30, alpha=0.5, label="Populer", color="#ff7f0e")
plt.hist(tidak_populer, bins=30, alpha=0.5, label="Tidak Populer", color="#1f77b4")
plt.title("Distribusi Panjang Lirik per Kategori Popularitas")
plt.xlabel("Jumlah Kata")
plt.ylabel("Jumlah Lagu")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. STATISTIK DESKRIPTIF PANJANG LIRIK PER KELAS
# ============================================================

stats_by_label = df.groupby("label_name")["lyric_length_words"].describe()

print("\nStatistik panjang lirik per kategori:")
print(stats_by_label)

stats_by_label.to_excel("Statistik_Lirik_Per_Kategori.xlsx")


# ============================================================
# 8. CETAK MEAN POPULARITAS SAJA (perintah khusus dari dosen)
# ============================================================

mean_popularity = df["popularitas"].mean()
print(f"\nRata-rata nilai popularitas dataset: {mean_popularity:.2f}")