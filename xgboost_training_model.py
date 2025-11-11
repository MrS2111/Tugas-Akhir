import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.visualization import plot_optimization_history, plot_param_importances

# === Konfigurasi dasar ===
DATA_PATH = "Dataset_BERT_Embeddings.pkl"
N_SPLITS = 5
RANDOM_STATE = 42

# === Load dataset ===
print("Memuat dataset...")
df = pd.read_pickle(DATA_PATH)
print(f"Jumlah data: {len(df)}")

# === Pisahkan fitur dan label ===
X = np.vstack(df['bert_embedding'].values)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

print(f"Dimensi fitur: {X.shape}")

# === Definisikan fungsi objective untuk Optuna ===
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': 0.3,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_lambda': 1.0,
        'reg_alpha': 0.0,
        'gamma': 0.0,
        'eval_metric': 'logloss', 
        'random_state': 0,
        'n_jobs': 0
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)

    return np.mean(f1_scores)

# === Jalankan Optuna untuk mencari hyperparameter terbaik ===
print("\nMenjalankan Optimasi Hyperparameter dengan Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=3600)

print("\nHyperparameter terbaik ditemukan:")
print(study.best_params)

# === Visualisasi hasil Optuna ===
print("\nMenampilkan visualisasi hasil optimasi...")
fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig1.show()
fig2.show()

# === Evaluasi ulang menggunakan parameter terbaik ===
best_params = study.best_params
best_params.update({
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
})

xgb_model = XGBClassifier(**best_params)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
print("\nMemulai Training & Evaluasi dengan 5-Fold Cross Validation...\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"🔹 Fold {fold}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    fold_results.append((acc, prec, rec, f1))

    print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    print("-" * 60)

# === Rata-rata hasil ===
results_mean = np.mean(fold_results, axis=0)
print("\nHasil Rata-rata dari 5-Fold Cross Validation:")
print(f"Accuracy : {results_mean[0]:.4f}")
print(f"Precision: {results_mean[1]:.4f}")
print(f"Recall   : {results_mean[2]:.4f}")
print(f"F1-Score : {results_mean[3]:.4f}")

# === Latih model akhir dengan seluruh data ===
print("\nTraining model akhir dengan seluruh data...")
xgb_model.fit(X, y)
y_pred_full = xgb_model.predict(X)

# === Confusion Matrix Full Data ===
cm_full = pd.crosstab(y, y_pred_full, rownames=['Aktual'], colnames=['Prediksi'])
plt.figure(figsize=(5, 4))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Model Akhir XGBoost")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# === Simpan model ===
joblib.dump(xgb_model, "xgboost_bert_optuna.pkl")
print("\nModel XGBoost (Optuna-tuned) disimpan!")

# === Ringkasan ===
print("\nRingkasan Akhir:")
print(f"- Jumlah data   : {len(df)}")
print(f"- Jumlah fitur  : {X.shape[1]}")
print(f"- Jumlah label  : {len(np.unique(y))}")
print(f"- Distribusi label: {dict(zip(label_encoder.classes_, np.bincount(y)))}")