import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.visualization import plot_optimization_history, plot_param_importances
import os

EMBEDDING_FILES = {
    128: "Dataset_BERT_Embeddings_128.pkl",
    256: "Dataset_BERT_Embeddings_256.pkl",
    384: "Dataset_BERT_Embeddings_384.pkl",
    512: "Dataset_BERT_Embeddings_512.pkl",
}

N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS = 200

for emb_dim, DATA_PATH in EMBEDDING_FILES.items():
    print("=" * 80)
    print(f"🔹 MEMPROSES EMBEDDING DIMENSI {emb_dim} | FILE: {DATA_PATH}")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        print(f"File {DATA_PATH} tidak ditemukan, dilewati.\n")
        continue

    # === Load dataset ===
    print("Memuat dataset...")
    df = pd.read_pickle(DATA_PATH)
    print(f"Jumlah data: {len(df)}")

    X = np.vstack(df['bert_embedding'].values)
    y = df['label'].astype(int).values

    print(f"Dimensi fitur: {X.shape}")
    print(f"Contoh nilai unik label: {np.unique(y)}")

    # === 80% train, 20% test (stratified) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Jumlah data train: {X_train.shape[0]}")
    print(f"Jumlah data test : {X_test.shape[0]}")

    trial_results = []

    # === Objective Optuna: CV hanya di TRAIN ===
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

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        acc_scores = []
        f1_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)

            y_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')

            acc_scores.append(acc)
            f1_scores.append(f1)

        mean_acc = float(np.mean(acc_scores))
        mean_f1 = float(np.mean(f1_scores))

        trial.set_user_attr("mean_accuracy", mean_acc)
        trial.set_user_attr("mean_f1", mean_f1)

        trial_results.append({
            "embedding_dim": emb_dim,
            "trial_number": trial.number,
            "max_depth": params["max_depth"],
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "reg_lambda": params["reg_lambda"],
            "reg_alpha": params["reg_alpha"],
            "gamma": params["gamma"],
            "mean_accuracy": mean_acc,
            "mean_f1": mean_f1,
        })

        return mean_f1  

    print("\nMenjalankan Optimasi Hyperparameter dengan Optuna (CV di TRAIN)...")
    study = optuna.create_study(direction="maximize", study_name=f"xgb_bert_{emb_dim}")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nHyperparameter terbaik ditemukan:")
    print(study.best_params)

    # Simpan hasil setiap trial
    df_trials = pd.DataFrame(trial_results)
    output_excel = f"optuna_xgb_trials_{emb_dim}.xlsx"
    df_trials.to_excel(output_excel, index=False)
    print(f"\nHasil lengkap setiap trial disimpan ke: {output_excel}")

    # (Opsional) Visualisasi Optuna
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show()
    fig2.show()

    # === TRAIN FINAL MODEL DI TRAIN, UJI DI TEST ===
    best_params = study.best_params
    best_params.update({
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    })

    xgb_model = XGBClassifier(**best_params)
    print(f"\nTraining model akhir dengan seluruh data TRAIN untuk dimensi {emb_dim}...")
    xgb_model.fit(X_train, y_train)

    # Evaluasi di TEST set (20%)
    y_pred_test = xgb_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, average='weighted')
    rec = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    print(f"\n📊 HASIL EVALUASI DI TEST SET (20%) dimensi {emb_dim}:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Confusion matrix di TEST set
    cm_test = pd.crosstab(y_test, y_pred_test, rownames=['Aktual'], colnames=['Prediksi'])

    plt.figure(figsize=(5, 4))
    xticklabels = ['Tidak Populer (0)', 'Populer (1)']
    yticklabels = ['Tidak Populer (0)', 'Populer (1)']

    sns.heatmap(
        cm_test,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=xticklabels,
        yticklabels=yticklabels
    )
    plt.title(f"Confusion Matrix - Test Set XGBoost ({emb_dim}-dim)")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.show()

    # Simpan model
    model_path = f"xgboost_bert_optuna_{emb_dim}.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"\n Model XGBoost (Optuna-tuned) untuk dimensi {emb_dim} disimpan sebagai: {model_path}")

    print("\nRingkasan Akhir:")
    print(f"- Dimensi embedding : {emb_dim}")
    print(f"- Jumlah data total : {len(df)}")
    print(f"- Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
    print(f"- Distribusi label total : {df['label'].value_counts().to_dict()}")
    print("\n\n")