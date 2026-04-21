import pandas as pd
import numpy as np


def add_features(df, horse_col, jockey_col, race_col):
    """
    Add strong engineered features (leakage-safe, dataset-only).
    """

    # ---------------------------
    # 🕒 SORT BY TIME (CRITICAL)
    # ---------------------------
    time_col = None
    for col in ["date", "race_date", "datetime"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df = df.sort_values(time_col)
    else:
        df = df.sort_index()  # fallback (not ideal)

    # ---------------------------
    # 🧠 BASIC FEATURES
    # ---------------------------
    if "runners" in df.columns:
        df["race_size"] = df["runners"]
    elif race_col and horse_col:
        df["race_size"] = df.groupby(race_col)[horse_col].transform("count")
    else:
        df["race_size"] = 0

    # ---------------------------
    # 🔥 ODDS FEATURES (MOST IMPORTANT)
    # ---------------------------
    if "decimalprice" in df.columns:
        df["decimalprice"] = df["decimalprice"].replace(0, np.nan)
        df["log_odds"] = np.log(df["decimalprice"])
        df["inv_odds"] = 1 / df["decimalprice"]
    else:
        df["log_odds"] = 0
        df["inv_odds"] = 0

    # ---------------------------
    # 🐎 HORSE EXPERIENCE
    # ---------------------------
    if horse_col:
        df["horse_races"] = df.groupby(horse_col).cumcount()
        df["horse_wins"] = df.groupby(horse_col)["win"].cumsum()

        df["horse_win_rate"] = df["horse_wins"] / (df["horse_races"] + 1)
    else:
        df["horse_races"] = 0
        df["horse_wins"] = 0
        df["horse_win_rate"] = 0

    # ---------------------------
    # 🧑‍✈️ JOCKEY EXPERIENCE
    # ---------------------------
    if jockey_col:
        df["jockey_races"] = df.groupby(jockey_col).cumcount()
        df["jockey_wins"] = df.groupby(jockey_col)["win"].cumsum()

        df["jockey_win_rate"] = df["jockey_wins"] / (df["jockey_races"] + 1)
    else:
        df["jockey_races"] = 0
        df["jockey_wins"] = 0
        df["jockey_win_rate"] = 0

    # ---------------------------
    # 📊 RECENT FORM (LAST 5 RACES)
    # ---------------------------
    if horse_col and "position" in df.columns:
        df["last_5_avg_pos"] = (
            df.groupby(horse_col)["position"]
            .rolling(5)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df["last_5_avg_pos"] = 0

    # ---------------------------
    # ⚔️ RELATIVE RACE FEATURES
    # ---------------------------
    if race_col:
        if "decimalprice" in df.columns:
            df["rank_odds"] = df.groupby(race_col)["decimalprice"].rank()
        else:
            df["rank_odds"] = 0

        if "weight" in df.columns:
            df["rank_weight"] = df.groupby(race_col)["weight"].rank()
        else:
            df["rank_weight"] = 0
    else:
        df["rank_odds"] = 0
        df["rank_weight"] = 0

    # ---------------------------
    # 🧹 CLEANUP
    # ---------------------------
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    print("✅ Feature engineering completed")
    print(f"[INFO] Columns added: horse_win_rate, jockey_win_rate, odds, experience, recent form")

    return df


def select_features(df, horse_col, jockey_col, race_col):
    """
    Select final feature columns for training.
    """

    # 🔥 CORE FEATURES (manually enforced)
    feature_cols = [
        "horse_win_rate",
        "jockey_win_rate",
        "race_size",
        "log_odds",
        "inv_odds",
        "horse_races",
        "jockey_races",
        "last_5_avg_pos",
        "rank_odds",
        "rank_weight"
    ]

    # ---------------------------
    # 🚫 REMOVE LEAKAGE
    # ---------------------------
    exclude = {
        "win",
        "position",
        "position_num",
        "position_raw",
        "res_win",
        "res_place",
    }

    exclude.update([c for c in df.columns if "position" in c.lower()])

    if race_col:
        exclude.add(race_col)
    if horse_col:
        exclude.add(horse_col)
    if jockey_col:
        exclude.add(jockey_col)

    # ---------------------------
    # ➕ ADD SAFE NUMERIC FEATURES
    # ---------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    extra_num = [
        c for c in num_cols
        if c not in feature_cols and c not in exclude
    ]

    feature_cols += extra_num

    # remove duplicates & keep valid columns
    final = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))

    print(f"[FEATURES] Using {len(final)} features")
    print(f"[FEATURES SAMPLE] {final[:15]}{'...' if len(final)>15 else ''}")

    return final