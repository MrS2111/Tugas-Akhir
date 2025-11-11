import os
import math
import pandas as pd
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

# -------- CONFIG --------
INPUT_FILE = "Dataset_labeled.xlsx"
OUTPUT_PICKLE = "Dataset_BERT_Embeddings_128.pkl"
OUTPUT_META_XLSX = "Dataset_BERT_Metadata_128.xlsx"
BATCH_SIZE = 16 # Ukuran batch untuk pemrosesan BERT
USE_CLS_POOLING = False
DEVICE = "cpu"  
MAX_SEQ_LENGTH = 512 # Panjang maksimum input token
# ------------------------

#--- INISIALISASI MODEL BERT ---
def init_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.to(DEVICE)
    return tokenizer, model

#--- FUNGSI EMBEDDING BERT UNTUK BATCH ---
def bert_embed_batch(texts, tokenizer, model, use_cls=False):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        last = out.last_hidden_state
        if use_cls:
            vecs = last[:, 0, :].cpu().numpy()  # CLS pooling
        else:
            mask = enc['attention_mask'].unsqueeze(-1)
            masked = last * mask
            summed = masked.sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            vecs = (summed / counts).cpu().numpy()  # Mean pooling
    return vecs

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File {INPUT_FILE} tidak ditemukan.")

    df = pd.read_excel(INPUT_FILE)
    if 'lirik' not in df.columns:
        raise ValueError("Kolom 'lirik' tidak ditemukan. Pastikan ada kolom bernama 'lirik'.")

    print(f"Jumlah baris input: {len(df)}")
    print("Inisialisasi model BERT (CPU)...")

    tokenizer, model = init_bert()
    hidden_size = model.config.hidden_size
    print(f"BERT hidden size = {hidden_size}. Device = CPU")

    texts = df['lirik'].fillna("").astype(str).tolist()
    n_batches = math.ceil(len(texts) / BATCH_SIZE)
    embeddings = []

    for i in tqdm(range(n_batches), desc="Proses batch"):
        start, end = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        vecs = bert_embed_batch(batch_texts, tokenizer, model, use_cls=USE_CLS_POOLING)
        embeddings.extend(vecs)

    df['bert_embedding'] = embeddings

    #--- Save output ---
    df.to_pickle(OUTPUT_PICKLE)
    meta_df = df.drop(columns=['bert_embedding'], errors='ignore')
    meta_df.to_excel(OUTPUT_META_XLSX, index=False)

    print("\nSelesai! File disimpan:")
    print(f" - Embedding + data: {OUTPUT_PICKLE}")
    print(f" - Metadata: {OUTPUT_META_XLSX}")

if __name__ == "__main__":
    main()