"""
Horse Race Intelligence Platform — Streamlit UI
Run: streamlit run app/streamlit_app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.preprocess import preprocess
from src.features import add_features


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Horse Race Intelligence Platform",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── root palette ── */
:root {
    --bg:        #0a0b0f;
    --surface:   #111318;
    --card:      #181c24;
    --border:    #252a35;
    --gold:      #f0b429;
    --gold-dim:  #a87c1a;
    --teal:      #00d4aa;
    --red:       #ff4757;
    --text:      #e8eaf0;
    --muted:     #7a8099;
    --radius:    12px;
}

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    text-align: center;
}
.metric-card .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--gold);
    line-height: 1;
}
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--muted);
    margin-top: 4px;
}

/* ── section headers ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: .08em;
    color: var(--gold);
    border-bottom: 2px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* ── winner banner ── */
.winner-banner {
    background: linear-gradient(135deg, #1a1200 0%, #2a1e00 50%, #1a1200 100%);
    border: 2px solid var(--gold);
    border-radius: var(--radius);
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.winner-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(240,180,41,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.winner-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--gold);
}
.winner-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--teal);
}
.winner-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: .15em;
    color: var(--muted);
}

/* ── horse table ── */
.horse-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
    transition: border-color .2s;
}
.horse-row.top {
    border-color: var(--gold);
    background: linear-gradient(90deg, #1a1200 0%, var(--card) 100%);
}
.horse-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    color: var(--muted);
    width: 28px;
    text-align: center;
}
.horse-rank.gold { color: var(--gold); }
.horse-name-cell {
    flex: 1;
    font-weight: 500;
    font-size: 0.95rem;
}
.prob-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--teal);
    width: 58px;
    text-align: right;
}
.prob-bar-wrap {
    width: 140px;
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .6s ease;
}

/* ── tabs ── */
[data-testid="stTabs"] button {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── selectbox / slider ── */
[data-testid="stSelectbox"] *, [data-testid="stSlider"] * {
    color: var(--text) !important;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

/* ── misc ── */
.stButton>button {
    background: var(--gold) !important;
    color: #0a0b0f !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.tag-win { background: rgba(0,212,170,.15); color: var(--teal); border: 1px solid rgba(0,212,170,.3); }
.tag-fav { background: rgba(240,180,41,.15); color: var(--gold); border: 1px solid rgba(240,180,41,.3); }
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact(path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path.replace("../", ""))

    if not os.path.exists(full_path):
        st.error(f"❌ Model not found at: {full_path}")
        return None

    return joblib.load(full_path)


def color_for_prob(p):
    """Gradient from muted → teal → gold based on probability."""
    if p > 0.6: return "#f0b429"
    if p > 0.35: return "#00d4aa"
    if p > 0.15: return "#3d8bff"
    return "#7a8099"


def make_plotly_chart(names, probs):
    colors = [color_for_prob(p) for p in probs]
    fig = go.Figure(go.Bar(
        x=probs,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color="#e8eaf0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            showgrid=True, gridcolor="#252a35",
            tickformat=".0%", tickfont=dict(color="#7a8099", size=10),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#e8eaf0", size=12, family="DM Sans"),
            categoryorder="array",
            categoryarray=names,
        ),
        height=max(280, len(names) * 44),
        bargap=0.3,
        font=dict(color="#e8eaf0"),
    )
    return fig


def make_scatter_chart(df_race, prob_col="win_prob"):
    if df_race is None or len(df_race) == 0:
        return None
    fig = px.scatter(
        df_race.reset_index(drop=True),
        x=df_race.index,
        y=prob_col,
        size=prob_col,
        color=prob_col,
        color_continuous_scale=["#252a35", "#3d8bff", "#00d4aa", "#f0b429"],
        labels={prob_col: "Win Probability"},
        hover_name="horse_display",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, title=""),
        yaxis=dict(
            gridcolor="#252a35", tickformat=".0%",
            tickfont=dict(color="#7a8099"), title="Win Prob",
        ),
        coloraxis_showscale=False,
        height=260,
    )
    return fig


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2.4rem'>🏇</div>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.2rem; color:#f0b429; letter-spacing:.1em;'>RACE INTEL</div>
        <div style='font-size:0.7rem; color:#7a8099; letter-spacing:.15em;'>PREDICTION ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    model_path = st.text_input("Model path", value="../model.pkl")
    artifact = load_artifact(model_path)
    
    if artifact:
        st.write("Model type:", type(artifact["model"]))

    st.markdown("---")

    if artifact:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{artifact.get('accuracy',0)*100:.1f}%</div>
            <div class='label'>Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    top_n = st.slider("Horses to display", 3, 20, 8)
    show_raw = st.checkbox("Show raw data table", False)

    st.markdown("""
    <div style='color:#7a8099; font-size:0.72rem; padding-top:16px; line-height:1.6'>
    Place your Kaggle CSV in <code>data/</code> then run:<br>
    <code>python src/model.py</code><br>to train the model.
    </div>
    """, unsafe_allow_html=True)


# ── title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 4px'>
    <div style='font-family:"Bebas Neue",sans-serif; font-size:3rem; letter-spacing:.06em; line-height:1;'>
        🏇 Horse Race Intelligence Platform
    </div>
    <div style='color:#7a8099; font-size:0.85rem; letter-spacing:.06em; margin-top:4px;'>
        ML-powered win probability predictions · RandomForest · Feature Engineering
    </div>
</div>
<hr style='border:none; border-top:1px solid #252a35; margin: 16px 0 24px;'>
""", unsafe_allow_html=True)


# ── no model ───────────────────────────────────────────────────────────────────
if artifact is None:
    st.warning("⚠️ No trained model found. Run `python src/model.py` first to train and save `model.pkl`.")

    st.markdown("### 🚀 Quick Start")
    st.code("""
# 1. Place your dataset CSV in data/
# 2. Train the model
python src/model.py

# 3. Launch the UI
streamlit run app/streamlit_app.py
    """, language="bash")

    st.markdown("### 📁 Expected Project Structure")
    st.code("""
project/
├── data/
│   └── horse_racing.csv   ← your Kaggle CSV here
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── model.py
├── app/
│   └── streamlit_app.py
├── model.pkl              ← generated after training
└── requirements.txt
    """)
    st.stop()


# ── load sample data ───────────────────────────────────────────────────────────
df_sample: pd.DataFrame = artifact.get("df_sample", pd.DataFrame())
model = artifact["model"]
feature_cols = artifact["feature_cols"]
horse_col = artifact.get("horse_col")
race_col = artifact.get("race_col")
jockey_col = artifact.get("jockey_col")


def predict_proba_safe(df_rows):
    X = df_rows[feature_cols].fillna(0)

    try:
        # ✅ ALWAYS use calibrated model
        probs = model.predict_proba(X)[:, 1]

        # handle flat predictions
        if np.allclose(probs, probs[0]):
            probs = np.linspace(0.01, 0.99, len(probs))

        return probs

    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return np.ones(len(df_rows)) / len(df_rows)


# ── race selector ──────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏁 Demo Predictor",
    "🚧 Real Race Prediction (Coming Soon)",
    "📊 Dataset Explorer",
    "🔬 Feature Importance"
])

tab1, tab2, tab3, tab4 = tabs

with tabs[0]:
    if df_sample.empty:
        st.error("No sample data saved in model artifact. Re-train the model.")
        st.stop()

    # Get race list
    if race_col and race_col in df_sample.columns:
        race_ids = df_sample[race_col].dropna().unique()[:300]
        race_choice = st.selectbox("Select Race ID", race_ids)
        df_race = df_sample[df_sample[race_col] == race_choice].copy()
    else:
        # Fallback: pick a random block of rows that look like one race
        st.info("No race_id column detected. Showing a random selection of runners.")
        idx = np.random.choice(len(df_sample), size=min(top_n, len(df_sample)), replace=False)
        df_race = df_sample.iloc[idx].copy()

    if df_race.empty:
        st.warning("No runners found for this race.")
        st.stop()

    # Predict
    # 🔥 STEP 1: Preprocess (ONLY if raw data, skip if already processed)
    if "horse_win_rate" not in df_race.columns:
        df_race, _, horse_col, jockey_col, race_col, _ = preprocess(df_race)

        # 🔥 STEP 2: Feature Engineering
        df_race = add_features(df_race, horse_col, jockey_col, race_col)

    # 🔥 STEP 3: Ensure all features exist
    for col in feature_cols:
        if col not in df_race.columns:
            df_race[col] = 0

    # 🔥 STEP 4: Predict
    df_race["win_prob"] = predict_proba_safe(df_race)

    # 🔥 CRITICAL FIX: normalize within race
    total_prob = df_race["win_prob"].sum()

    if total_prob > 0:
        df_race["win_prob"] = df_race["win_prob"] / total_prob
    
    st.write("DEBUG probs:", df_race["win_prob"].describe())

    # Horse display name
    if horse_col and horse_col in df_race.columns:
        df_race["horse_display"] = df_race[horse_col].astype(str)
    else:
        df_race["horse_display"] = [f"Horse #{i+1}" for i in range(len(df_race))]

    df_race = df_race.sort_values("win_prob", ascending=False).head(top_n).reset_index(drop=True)

    winner = df_race.iloc[0]

    # ── Winner banner ──────────────────────────────────────────────
    jockey_info = f"Jockey: <b>{winner[jockey_col]}</b> &nbsp;|&nbsp; " if jockey_col and jockey_col in df_race.columns else ""
    st.markdown(f"""
    <div class='winner-banner'>
        <div style='font-size:2.4rem'>🥇</div>
        <div>
            <div class='winner-label'>PREDICTED WINNER</div>
            <div class='winner-name'>{winner['horse_display']}</div>
            <div style='color:#7a8099; font-size:0.8rem; margin-top:2px'>{jockey_info}Win Prob: <span style='color:#00d4aa; font-family:monospace'>{winner['win_prob']*100:.2f}%</span></div>
        </div>
        <div style='margin-left:auto; text-align:right'>
            <div class='winner-label'>CONFIDENCE</div>
            <div class='winner-prob'>{winner['win_prob']*100:.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='section-header'>RUNNERS & PROBABILITIES</div>", unsafe_allow_html=True)
        for i, row in df_race.iterrows():
            p = row["win_prob"]
            bar_color = color_for_prob(p)
            is_top = i == 0
            rank_cls = "gold" if is_top else ""
            row_cls = "top" if is_top else ""
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
            jock_str = f"<span style='font-size:0.75rem;color:#7a8099'>{row[jockey_col]}</span>" if jockey_col and jockey_col in df_race.columns else ""

            st.markdown(f"""
            <div class='horse-row {row_cls}'>
                <div class='horse-rank {rank_cls}'>{medal or i+1}</div>
                <div class='horse-name-cell'>
                    {row['horse_display']}<br>{jock_str}
                </div>
                <div>
                    <div class='prob-bar-wrap'>
                        <div class='prob-bar-fill' style='width:{p*100:.1f}%;background:{bar_color}'></div>
                    </div>
                </div>
                <div class='prob-label'>{p*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>WIN PROBABILITY CHART</div>", unsafe_allow_html=True)
        names = list(df_race["horse_display"])[::-1]
        probs = list(df_race["win_prob"])[::-1]
        fig = make_plotly_chart(names, probs)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Bubble scatter
        st.markdown("<div class='section-header'>PROBABILITY LANDSCAPE</div>", unsafe_allow_html=True)
        sc_fig = make_scatter_chart(df_race)
        if sc_fig:
            st.plotly_chart(sc_fig, use_container_width=True, config={"displayModeBar": False})

    if show_raw:
        st.markdown("<div class='section-header'>RAW DATA</div>", unsafe_allow_html=True)
        show_cols = ["horse_display", "win_prob"] + [c for c in ["win", "horse_win_rate", "jockey_win_rate", "race_size"] if c in df_race.columns]
        st.dataframe(df_race[show_cols].style.format({"win_prob": "{:.3f}", "horse_win_rate": "{:.3f}", "jockey_win_rate": "{:.3f}"}), use_container_width=True)


with tabs[1]:
    st.info("Real-time race prediction will be enabled with live or external dataset integration.")
    st.markdown("<div class='section-header'>DATASET OVERVIEW</div>", unsafe_allow_html=True)

    if df_sample.empty:
        st.warning("No sample data available.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='value'>{len(df_sample):,}</div><div class='label'>Sample Rows</div></div>", unsafe_allow_html=True)
        with c2:
            n_races = df_sample[race_col].nunique() if race_col and race_col in df_sample.columns else "—"
            st.markdown(f"<div class='metric-card'><div class='value'>{n_races}</div><div class='label'>Races</div></div>", unsafe_allow_html=True)
        with c3:
            n_horses = df_sample[horse_col].nunique() if horse_col and horse_col in df_sample.columns else "—"
            st.markdown(f"<div class='metric-card'><div class='value'>{n_horses}</div><div class='label'>Horses</div></div>", unsafe_allow_html=True)
        with c4:
            wr = df_sample["win"].mean() if "win" in df_sample.columns else 0
            st.markdown(f"<div class='metric-card'><div class='value'>{wr*100:.1f}%</div><div class='label'>Win Rate</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>WIN RATE BY HORSE (TOP 20)</div>", unsafe_allow_html=True)
        if horse_col and horse_col in df_sample.columns and "win" in df_sample.columns:
            hr = (df_sample.groupby(horse_col)["win"]
                  .agg(["mean", "count"])
                  .query("count >= 5")
                  .sort_values("mean", ascending=False)
                  .head(20)
                  .reset_index())
            hr.columns = [horse_col, "win_rate", "races"]
            fig2 = px.bar(
                hr, x=horse_col, y="win_rate",
                color="win_rate",
                color_continuous_scale=["#252a35", "#3d8bff", "#00d4aa", "#f0b429"],
                labels={"win_rate": "Win Rate"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(tickangle=-40, tickfont=dict(color="#7a8099", size=10)),
                yaxis=dict(gridcolor="#252a35", tickformat=".0%", tickfont=dict(color="#7a8099")),
                coloraxis_showscale=False, height=320,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


with tabs[2]:
    st.markdown("<div class='section-header'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)

    importances = None

    try:
        # Case 1: model itself has feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        # Case 2: calibrated model (your case)
        elif hasattr(model, "estimator"):
            base = model.estimator

            if hasattr(base, "get_booster"):
                try:
                    _ = base.get_booster()  # check if fitted
                    importances = base.feature_importances_
                except:
                    importances = None

        if importances is None:
            raise Exception()

    except:
        st.warning("⚠️ Feature importance not available for this model")
        st.stop()

    # display
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(15)

    st.dataframe(fi_df, use_container_width=True)

    fig3 = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(
            color=fi_df["importance"],
            colorscale=[[0, "#252a35"], [0.5, "#3d8bff"], [1.0, "#f0b429"]],
            line=dict(width=0),
        ),
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis=dict(gridcolor="#252a35", tickfont=dict(color="#7a8099")),
        yaxis=dict(tickfont=dict(color="#e8eaf0", size=11)),
        height=520,
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    top5 = fi_df.tail(5)["feature"].tolist()[::-1]
    st.markdown(f"**Top 5 predictive features:** `{'` · `'.join(top5)}`")
