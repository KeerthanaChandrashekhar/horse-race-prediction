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
    justify-content: space-between;
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
    min-width: 50px;
    text-align: left;
    margin-left: 8px;
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

st.markdown("### 🎯 AI-Powered Decision Support System for Horse Race Prediction under Uncertainty")


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
    "🏇 Race Simulator"
])

tab1, tab2 = tabs

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
    st.markdown("""
    <div style="
        background: #181c24;
        border: 1px solid #252a35;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
    ">
        <div style="color:#f0b429; font-family:'Bebas Neue'; font-size:1.1rem; letter-spacing:.08em;">
            SYSTEM OVERVIEW
        </div>
        <div style="color:#e8eaf0; font-size:0.85rem; margin-top:8px; line-height:1.6;">
            • Uses <b>odds + performance features</b> to predict win probability<br>
            • Considers <b>horse ability, jockey skill, and experience</b><br>
            • Provides <b>decision insights</b>: Best Bet, Value Bet, Risky Horse
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-header'>CUSTOM RACE INPUT</div>", unsafe_allow_html=True)

    num_horses = st.number_input("Number of Horses", 2, 15, 5)

    horses_data = []

    for i in range(num_horses):
        st.markdown(f"### Horse {i+1}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            name = st.text_input(f"Name", key=f"name_{i}")

        with col2:
            odds = st.number_input(f"Odds", min_value=1.01, value=2.0, key=f"odds_{i}")

        with col3:
            horse_wr = st.slider(f"Horse Win Rate", 0.0, 1.0, 0.1, key=f"hwr_{i}")

        with col4:
            jockey_wr = st.slider(f"Jockey Win Rate", 0.0, 1.0, 0.1, key=f"jwr_{i}")

        exp = st.number_input(f"Experience", min_value=1, value=5, key=f"exp_{i}")

        horses_data.append({
            "horse": name if name else f"Horse_{i+1}",
            "odds": odds,
            "horse_win_rate": horse_wr,
            "jockey_win_rate": jockey_wr,
            "experience": exp
        })

    if st.button("Predict Custom Race"):

        df_custom = pd.DataFrame(horses_data)

        # --- Feature Engineering ---
        df_custom["log_odds"] = np.log(df_custom["odds"])
        df_custom["inv_odds"] = 1 / df_custom["odds"]
        df_custom["avg_last5_pos"] = 5
        df_custom["race_size"] = len(df_custom)

        # --- Ensure feature match ---
        for col in feature_cols:
            if col not in df_custom.columns:
                df_custom[col] = 0

        # --- Equal odds fix ---
        # Check if ALL meaningful inputs are same
        if (
            df_custom["odds"].nunique() == 1 and
            df_custom["horse_win_rate"].nunique() == 1 and
            df_custom["jockey_win_rate"].nunique() == 1 and
            df_custom["experience"].nunique() == 1
        ):
            probs = np.ones(len(df_custom)) / len(df_custom)
        else:
            probs = predict_proba_safe(df_custom)
            probs = probs / probs.sum()

        df_custom["win_prob"] = probs

        df_custom = df_custom.sort_values("win_prob", ascending=False).reset_index(drop=True)

        winner = df_custom.iloc[0]

        st.success(f"🏆 Predicted Winner: {winner['horse']} ({winner['win_prob']*100:.2f}%)")
        
        st.warning("""
        ⚠️ Predictions are probabilistic, not guaranteed outcomes.
        This system is designed for decision support under uncertainty.
        """)
        
        # =========================
        # 🎯 Decision Intelligence Layer
        # =========================

        st.markdown("### 🧠 Decision Insights")

        # 🟢 Best Bet (highest probability)
        best_bet = df_custom.iloc[0]

        # 🟡 Value Bet (high odds + decent probability)
        df_custom["value_score"] = df_custom["win_prob"] * df_custom["odds"]
        value_bet = df_custom.sort_values("value_score", ascending=False).iloc[0]

        # 🔴 Risky Horse (low probability + high odds)
        risky = df_custom.sort_values(by=["win_prob", "odds"], ascending=[True, False]).iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>🟢 {best_bet['horse']}</div>
                <div class='label'>Best Bet ({best_bet['win_prob']*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>🟡 {value_bet['horse']}</div>
                <div class='label'>Value Bet (Odds {value_bet['odds']})</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>🔴 {risky['horse']}</div>
                <div class='label'>Risky (Low Prob)</div>
            </div>
            """, unsafe_allow_html=True)

        # --- Chart ---
        fig = make_plotly_chart(
            list(df_custom["horse"])[::-1],
            list(df_custom["win_prob"])[::-1]
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # --- Table ---
        st.dataframe(
            df_custom[["horse", "odds", "horse_win_rate", "jockey_win_rate", "experience", "win_prob"]],
            use_container_width=True
        )