from flask import Flask, render_template, request
import numpy as np
import pickle
import torch
from transformers import BertTokenizer, BertModel
from langdetect import detect, LangDetectException  

from preprocessing_data import preprocess_lyrics 

# ==========================
# Konfigurasi BERT & Model
# ==========================

# Device (pakai GPU kalau ada)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & model BERT base
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
bert_model.to(DEVICE)
bert_model.eval()  # set ke eval mode


def is_english(text: str) -> bool:
    """
    Mengembalikan True jika teks terdeteksi sebagai bahasa Inggris.
    Jika gagal mendeteksi (misalnya teks terlalu pendek), dianggap bukan Inggris.
    """
    try:
        lang = detect(text)
        return lang == "en"
    except LangDetectException:
        return False


def embed_text_with_bert(text: str, max_length: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Mengubah 1 teks (lirik) menjadi 1 vektor embedding BERT (mean pooling).
    Output: shape (1, hidden_size) → biasanya (1, 768).

    Akan melempar ValueError jika teks menjadi kosong setelah preprocessing.
    """

    # 1. Preprocessing teks
    cleaned_text = preprocess_lyrics(text)

    if not cleaned_text or cleaned_text.strip() == "":
        # Misalnya semua karakter non-alfabetik hilang, atau hanya tag yang terhapus.
        raise ValueError(
            "Lirik tidak valid setelah preprocessing. "
            "Pastikan lirik berisi kalimat yang jelas."
        )

    # 2. Tokenisasi
    encoded = tokenizer(
        cleaned_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    # 3. Forward BERT
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]

    # 4. Mean pooling dengan perhatian terhadap mask
    mask = attention_mask.unsqueeze(-1)              # [batch, seq_len, 1]
    summed = (last_hidden_state * mask).sum(dim=1)   # [batch, hidden]
    counts = mask.sum(dim=1)                         # [batch, 1]
    counts = counts.clamp(min=1)                     # hindari pembagian dengan nol
    mean_pooled = summed / counts                    # [batch, hidden]

    emb = mean_pooled.cpu().numpy()
    return emb


# ==========================
# Load Model XGBoost
# ==========================

XGB_MODEL_PATH = "models/xgboost_bert_optuna_128.pkl"

xgb_model = None
try:
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
except Exception as e:
    # Kalau gagal load model, dicetak ke log saja.
    print(f"[ERROR] Gagal memuat model XGBoost dari '{XGB_MODEL_PATH}': {e}")
    xgb_model = None


# ==========================
# Aplikasi Flask
# ==========================

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    probability_popular = None
    probability_not_popular = None
    raw_lyrics = ""
    error_message = None

    if request.method == "POST":
        raw_lyrics = request.form.get("lyrics", "").strip()

        # Error handling input lirik
        if not raw_lyrics:
            error_message = "Lirik tidak boleh kosong."
        elif len(raw_lyrics.split()) < 5:
            error_message = "Lirik terlalu pendek untuk dianalisis (minimal sekitar 5 kata)."
        elif not is_english(raw_lyrics):
            error_message = "Lirik harus berbahasa Inggris agar dapat diproses oleh model."
        elif len(raw_lyrics) > 5000:
            error_message = "Lirik terlalu panjang (maksimal 5000 karakter)."
        elif xgb_model is None:
            error_message = "Model prediksi belum tersedia di server. Hubungi pengelola sistem."
        else:
            # ---------- Proses prediksi + error handling ----------
            try:
                # 1. Embedding BERT
                emb = embed_text_with_bert(raw_lyrics, max_length=MAX_SEQ_LEN)  # shape (1, d)

                # 2. Prediksi dengan model
                proba = xgb_model.predict_proba(emb)[0]
                prob_not_popular = float(proba[0])
                prob_popular = float(proba[1])

                pred_class = int(xgb_model.predict(emb)[0])

                prediction_label = "Populer" if pred_class == 1 else "Tidak Populer"

                probability_popular = round(prob_popular * 100, 2)
                probability_not_popular = round(prob_not_popular * 100, 2)

            except ValueError as ve:
                error_message = str(ve)
            except Exception as e:
                print(f"[ERROR] Terjadi exception saat prediksi: {e}")
                error_message = (
                    "Terjadi kesalahan internal saat memproses lirik. "
                    "Silakan coba lagi beberapa saat atau gunakan lirik lain."
                )

    return render_template(
        "index.html",
        prediction_label=prediction_label,
        probability_popular=probability_popular,
        probability_not_popular=probability_not_popular,
        raw_lyrics=raw_lyrics,
        error_message=error_message,
        model_name=BERT_MODEL_NAME,
        max_seq_len=MAX_SEQ_LEN,
        device=str(DEVICE),
    )


if __name__ == "__main__":
    app.run(debug=True)