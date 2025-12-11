import pandas as pd
import matplotlib.pyplot as plt

# === Load dataset hasil preprocessing ===
df = pd.read_excel("Dataset_final_clean.xlsx")

# === Hitung jumlah kata per lagu ===
df["jumlah_kata"] = df["lirik"].apply(lambda x: len(str(x).split()))

# === Statistik umum ===
print("Rata-rata panjang lirik:", df["jumlah_kata"].mean())
print("Median panjang lirik:", df["jumlah_kata"].median())
print("Lirik terpendek:", df["jumlah_kata"].min())
print("Lirik terpanjang:", df["jumlah_kata"].max())

# === Plot distribusi panjang lirik ===
plt.figure(figsize=(10,6))
plt.hist(df["jumlah_kata"], bins=40, edgecolor='black')
plt.title("Distribusi Panjang Lirik Lagu Setelah Preprocessing")
plt.xlabel("Jumlah Kata per Lagu")
plt.ylabel("Frekuensi")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()