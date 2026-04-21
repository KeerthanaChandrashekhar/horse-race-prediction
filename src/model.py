import pandas as pd
import numpy as np
import joblib
import os
import sys

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess
from src.features import add_features, select_features


# ---------------------------
# LOAD DATA
# ---------------------------
def load_training_data(data_dir="data/"):
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith("horses_") and f.endswith(".csv")
    ]

    if not files:
        raise ValueError("No training files found")

    dfs = []
    for file in files:
        print(f"[INFO] Loading {file}")
        year = int(file.split("_")[1].split(".")[0])
        df = pd.read_csv(os.path.join(data_dir, file))
        df["year"] = year
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # sample for speed
    df = df.sample(n=300000, random_state=42)

    print("[INFO] Shape:", df.shape)
    return df


# ---------------------------
# TRAIN FUNCTION
# ---------------------------
def train(data_dir="data/", model_path="model.pkl"):

    # ── Load & preprocess ─────────────────────────
    df_raw = load_training_data(data_dir)
    df, pos_col, horse_col, jockey_col, race_col, le_map = preprocess(df_raw)

    # ── Feature engineering ───────────────────────
    df = add_features(df, horse_col, jockey_col, race_col)
    feature_cols = select_features(df, horse_col, jockey_col, race_col)

    X = df[feature_cols].copy()
    y = df["win"].copy()

    X = X.fillna(0)

    print(f"\n[MODEL] Dataset: {len(X)} rows | Wins: {y.sum()} ({y.mean()*100:.2f}%)")

    # ---------------------------
    # 🕒 TIME-BASED SPLIT (CRITICAL)
    # ---------------------------
    time_col = None
    for col in ["date", "race_date", "datetime", "year"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df = df.sort_values(time_col)
        split_idx = int(len(df) * 0.8)

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[feature_cols]
        y_train = train_df["win"]

        X_test = test_df[feature_cols]
        y_test = test_df["win"]

        print(f"[SPLIT] Time-based split using {time_col}")
    else:
        # fallback (not ideal)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("[SPLIT] Random split (fallback)")

    # ---------------------------
    # ⚖️ HANDLE IMBALANCE
    # ---------------------------
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"[INFO] scale_pos_weight: {scale_pos_weight:.2f}")

    # ---------------------------
    # 🚀 XGBOOST MODEL
    # ---------------------------
    print("\n[MODEL] Training XGBoost...")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # ---------------------------
    # 🎯 CALIBRATION (VERY IMPORTANT)
    # ---------------------------
    model = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    # ---------------------------
    # 📊 EVALUATION
    # ---------------------------
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)

    print(f"\n[MODEL] Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["No Win", "Win"]))

    # ---------------------------
    # 💾 SAVE MODEL
    # ---------------------------
    
# 🔥 keep full races instead of random rows
    # 🔥 better sampling: pick races with FULL data
    race_counts = df[race_col].value_counts()

    # keep races with enough horses (>=5)
    valid_races = race_counts[race_counts >= 5].index[:100]

    df_sample = df[df[race_col].isin(valid_races)].copy()

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "horse_col": horse_col,
        "jockey_col": jockey_col,
        "race_col": race_col,
        "pos_col": pos_col,
        "le_map": le_map,
        "accuracy": acc,
        "df_sample": df_sample,
    }

    joblib.dump(artifact, model_path)
    print(f"\n[SAVED] Model → {model_path}")

    return artifact


if __name__ == "__main__":
    train()