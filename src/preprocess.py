import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(data_dir="data/"):
    """Load horse racing dataset - tries common filenames."""
    candidates = [
        "horse_racing.csv", "horses.csv", "races.csv",
        "horse-racing.csv", "data.csv", "results.csv"
    ]
    for fname in candidates:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print(f"[INFO] Loading: {path}")
            df = pd.read_csv(path)
            print(f"[INFO] Shape: {df.shape}")
            print(f"[INFO] Columns: {list(df.columns)}")
            return df

    # Try loading any CSV in the directory
    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if csvs:
        path = os.path.join(data_dir, csvs[0])
        print(f"[INFO] Loading first CSV found: {path}")
        df = pd.read_csv(path)
        print(f"[INFO] Shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
        return df

    raise FileNotFoundError(f"No CSV found in {data_dir}. Please place your Kaggle dataset there.")


def normalize_columns(df):
    """Lowercase and strip column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def detect_position_col(df):
    """Detect position/finishing column."""
    candidates = ["position", "pos", "finish", "finishing_position", "place", "fin_pos", "fin"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find position column. Columns: {list(df.columns)}")


def detect_horse_col(df):
    candidates = ["horse", "horse_name", "name", "animal", "runner"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_jockey_col(df):
    candidates = ["jockey", "jockey_name", "rider", "jock"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_race_col(df):
    candidates = ["race_id", "race", "race_number", "race_no", "raceid", "course", "meeting"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def preprocess(df):
    df = normalize_columns(df)
    print(f"[PREPROCESS] Normalized columns: {list(df.columns)}")

    # Detect key columns
    pos_col = detect_position_col(df)
    horse_col = detect_horse_col(df)
    jockey_col = detect_jockey_col(df)
    race_col = detect_race_col(df)
    
    # 🔥 FIX: Force correct column mapping for this dataset
    if 'horsename' in df.columns:
        horse_col = 'horsename'

    if 'jockeyname' in df.columns:
        jockey_col = 'jockeyname'

    if 'rid' in df.columns:
        race_col = 'rid'

    print(f"[PREPROCESS] pos={pos_col}, horse={horse_col}, jockey={jockey_col}, race={race_col}")

    # Create target: win = 1 if position == 1
    df["position_raw"] = df[pos_col].astype(str).str.strip()
    df["position_num"] = pd.to_numeric(df["position_raw"], errors="coerce")
    df["win"] = (df["position_num"] == 1).astype(int)
    print(f"[PREPROCESS] Win rate: {df['win'].mean():.3f} | Wins: {df['win'].sum()}")

    # Drop rows where position is unknown
    df = df.dropna(subset=["position_num"]).copy()

    # Numeric columns: fill missing with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if c not in ["win", "position_num"]:
            df[c] = df[c].fillna(df[c].median())

    # Categorical encoding
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le_map = {}
    for c in cat_cols:
        df[c] = df[c].fillna("unknown")
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c].astype(str))
        le_map[c] = le

    return df, pos_col, horse_col, jockey_col, race_col, le_map
