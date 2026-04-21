def run_pipeline(df, preprocess, add_features, feature_cols, model):
    # Preprocess
    df, _, horse_col, jockey_col, race_col, _ = preprocess(df)

    # Feature engineering
    df = add_features(df, horse_col, jockey_col, race_col)

    # Ensure features match training
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)

    # Predict
    df["win_prob"] = model.predict_proba(X)[:, 1]

    return df, horse_col, jockey_col, race_col