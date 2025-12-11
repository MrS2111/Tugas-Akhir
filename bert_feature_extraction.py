import os
import math
import pandas as pd
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

# -------- CONFIG --------
INPUT_FILE = "Dataset_labeled.xlsx"
BATCH_SIZE = 16
USE_CLS_POOLING = False
DEVICE = "cpu"

# 5 nilai max sequence length 
MAX_SEQ_LIST = [128, 256, 384, 512]

# --- INISIALISASI MODEL BERT ---
def init_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.to(DEVICE)
    return tokenizer, model

# --- FUNGSI EMBEDDING BERT UNTUK BATCH ---
def bert_embed_batch(texts, tokenizer, model, max_seq, use_cls=False):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        last = out.last_hidden_state
        if use_cls:
            vecs = last[:, 0, :].cpu().numpy()
        else:
            mask = enc['attention_mask'].unsqueeze(-1)
            masked = last * mask
            summed = masked.sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            vecs = (summed / counts).cpu().numpy()
    return vecs

def generate_embedding_for_maxseq(df, tokenizer, model, max_seq):
    texts = df["lirik"].fillna("").astype(str).tolist()
    n_batches = math.ceil(len(texts) / BATCH_SIZE)
    embeddings = []

    print(f"\n=== Memproses MAX_SEQ_LENGTH = {max_seq} ===")
    for i in tqdm(range(n_batches), desc=f"Batch MAXSEQ {max_seq}"):
        start, end = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        vecs = bert_embed_batch(batch_texts, tokenizer, model, max_seq, use_cls=USE_CLS_POOLING)
        embeddings.extend(vecs)

    df["bert_embedding"] = embeddings

    # Simpan file
    output_pickle = f"Dataset_BERT_Embeddings_{max_seq}.pkl"
    output_meta = f"Dataset_BERT_Metadata_{max_seq}.xlsx"

    df.to_pickle(output_pickle)
    df.drop(columns=["bert_embedding"]).to_excel(output_meta, index=False)

    print(f"✔ Selesai MAX_SEQ = {max_seq}")
    print(f"   → Embedding disimpan di : {output_pickle}")
    print(f"   → Metadata disimpan di  : {output_meta}")

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File {INPUT_FILE} tidak ditemukan.")

    df = pd.read_excel(INPUT_FILE)
    if "lirik" not in df.columns:
        raise ValueError("Kolom 'lirik' tidak ditemukan pada dataset.")

    print(f"Jumlah data: {len(df)}")
    print("Inisialisasi BERT...")

    tokenizer, model = init_bert()
    print("Model siap digunakan!\n")

    # Loop 5x untuk setiap konfigurasi max_seq_length
    for max_seq in MAX_SEQ_LIST:
        generate_embedding_for_maxseq(df.copy(), tokenizer, model, max_seq)

    print("\n=== SEMUA PROSES SELESAI ===")

if __name__ == "__main__":
    main()
