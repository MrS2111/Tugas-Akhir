import pandas as pd

df = pd.read_pickle("Dataset_BERT_Embeddings.pkl")

# Periksa struktur embedding
print(df['bert_embedding'].iloc[0][:10])  # Tampilkan 10 nilai pertama
print("Jumlah fitur per lagu:", len(df['bert_embedding'].iloc[0]))

# Cek apakah ada nilai kosong
print("Ada embedding kosong:", df['bert_embedding'].isnull().any())
print("Jumlah total lagu:", len(df))
print(df.head(3))