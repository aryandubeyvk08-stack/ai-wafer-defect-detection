# ╔══════════════════════════════════════════════════════════════════════╗
# ║        AI-Based Wafer Defect Detection — Streamlit GUI App          ║
# ║        Semiconductor Yield Prediction Tool                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Industrial / Clean Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Main background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(88,166,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.header-sub {
    font-size: 0.9rem;
    color: #8b949e;
    font-weight: 300;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8b949e;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
}
.metric-sub {
    font-size: 0.75rem;
    color: #6e7681;
    margin-top: 4px;
}

/* Prediction result */
.pred-good {
    background: linear-gradient(135deg, #0d2818, #1a4731);
    border: 2px solid #2ea043;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
}
.pred-defective {
    background: linear-gradient(135deg, #2d1117, #4d1f28);
    border: 2px solid #f85149;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
}
.pred-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.pred-conf {
    font-size: 0.9rem;
    color: #8b949e;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Input labels */
label { color: #c9d1d9 !important; font-size: 0.85rem !important; }

/* Buttons */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 10px 24px !important;
    transition: background 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #388bfd !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: #161b22 !important;
    color: #58a6ff !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    width: 100%;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #58a6ff !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}

/* Progress / info boxes */
.stAlert { border-radius: 8px !important; }
.stInfo { background: #132236 !important; border-color: #58a6ff !important; }

/* Feature importance bars */
.fi-bar-wrap { margin: 6px 0; }
.fi-label {
    font-size: 0.8rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 3px;
}
.fi-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 10px;
    width: 100%;
}
.fi-bar-fill {
    height: 10px;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.fi-score {
    font-size: 0.75rem;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    text-align: right;
    margin-top: 2px;
}

/* Divider */
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — DATA GENERATION
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generate_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthesise realistic 200mm CMOS wafer process data."""
    rng = np.random.default_rng(seed)
    thickness      = rng.normal(100.0, 2.5,   n)
    doping_conc    = rng.normal(1e15,  5e13,  n)
    temperature    = rng.normal(300.0, 5.0,   n)
    defect_density = np.abs(rng.normal(0.01,  0.008, n))

    label = (
        (np.abs(thickness - 100.0) > 5.0)     |
        (np.abs(doping_conc - 1e15) > 1.5e14) |
        (np.abs(temperature - 300.0) > 12.0)  |
        (defect_density > 0.025)
    ).astype(int)

    return pd.DataFrame({
        "thickness":      thickness,
        "doping_conc":    doping_conc,
        "temperature":    temperature,
        "defect_density": defect_density,
        "defective":      label,
    })


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
FEATURES = ["thickness", "doping_conc", "temperature", "defect_density"]

@st.cache_resource(show_spinner=False)
def train_models(seed: int = 42):
    """Train Random Forest & Logistic Regression; return models + metrics."""
    df = generate_data(seed=seed)
    X = df[FEATURES].values
    y = df["defective"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    metrics = {
        "acc_rf": accuracy_score(y_test, y_pred_rf),
        "acc_lr": accuracy_score(y_test, y_pred_lr),
        "report_rf": classification_report(
            y_test, y_pred_rf, target_names=["Good", "Defective"], output_dict=True
        ),
        "report_lr": classification_report(
            y_test, y_pred_lr, target_names=["Good", "Defective"], output_dict=True
        ),
        "cm_rf":  confusion_matrix(y_test, y_pred_rf),
        "cm_lr":  confusion_matrix(y_test, y_pred_lr),
        "importances": rf.feature_importances_,
        "feature_labels": ["Thickness", "Doping Conc.", "Temperature", "Defect Density"],
    }
    return rf, lr, scaler, metrics


# ═══════════════════════════════════════════════════════════════════
# MODULE 3 — PREDICTION
# ═══════════════════════════════════════════════════════════════════
def predict_wafer(rf, lr, scaler, thickness, doping, temperature, defect_density):
    """Return predictions + probabilities for both models."""
    x = np.array([[thickness, doping, temperature, defect_density]])
    x_sc = scaler.transform(x)

    pred_rf  = rf.predict(x)[0]
    prob_rf  = rf.predict_proba(x)[0]
    pred_lr  = lr.predict(x_sc)[0]
    prob_lr  = lr.predict_proba(x_sc)[0]

    return {
        "rf_label":      "DEFECTIVE" if pred_rf else "GOOD",
        "rf_confidence": float(np.max(prob_rf)) * 100,
        "rf_prob_good":  float(prob_rf[0]) * 100,
        "rf_prob_def":   float(prob_rf[1]) * 100,
        "lr_label":      "DEFECTIVE" if pred_lr else "GOOD",
        "lr_confidence": float(np.max(prob_lr)) * 100,
        "lr_prob_good":  float(prob_lr[0]) * 100,
        "lr_prob_def":   float(prob_lr[1]) * 100,
    }


# ═══════════════════════════════════════════════════════════════════
# MODULE 4 — VISUALISATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    "good":       "#2ea043",
    "defective":  "#f85149",
    "rf":         "#58a6ff",
    "lr":         "#e67e22",
    "bg":         "#0d1117",
    "card":       "#161b22",
    "border":     "#30363d",
    "text":       "#c9d1d9",
    "muted":      "#8b949e",
}

def build_dashboard(df: pd.DataFrame, metrics: dict, pred_result: dict | None = None):
    """Render the full matplotlib results dashboard and return a Figure."""
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["card"],
        "axes.edgecolor":    COLORS["border"],
        "axes.labelcolor":   COLORS["text"],
        "xtick.color":       COLORS["muted"],
        "ytick.color":       COLORS["muted"],
        "text.color":        COLORS["text"],
        "grid.color":        COLORS["border"],
        "grid.alpha":        0.5,
        "font.family":       "monospace",
    })

    rows = 3 if pred_result else 2
    fig = plt.figure(figsize=(18, rows * 6))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(
        "AI-Based Wafer Defect Detection  ·  Results Dashboard",
        fontsize=14, fontweight="bold", color=COLORS["rf"], y=0.99,
        fontfamily="monospace",
    )

    gs = gridspec.GridSpec(rows, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── 1. Pie — Wafer quality split ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["defective"].value_counts().sort_index()
    wedge_colors = [COLORS["good"], COLORS["defective"]]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=["Good", "Defective"],
        colors=wedge_colors, autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 2},
        textprops={"color": COLORS["text"], "fontsize": 10},
    )
    for at in autotexts:
        at.set_color(COLORS["bg"]); at.set_fontweight("bold")
    ax1.set_title("Wafer Quality Distribution", fontweight="bold",
                  color=COLORS["text"], pad=12)

    # ── 2. Bar H — Feature importance ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    importances = metrics["importances"]
    labels = metrics["feature_labels"]
    idx = np.argsort(importances)
    bar_cols = [COLORS["defective"] if i == idx[-1] else COLORS["rf"]
                for i in range(len(importances))]
    bars = ax2.barh(
        [labels[i] for i in idx], importances[idx],
        color=[bar_cols[i] for i in idx], height=0.55,
    )
    ax2.set_xlabel("Importance Score", fontsize=9)
    ax2.set_title("Feature Importance (RF)", fontweight="bold",
                  color=COLORS["text"], pad=12)
    ax2.axvline(importances.mean(), color=COLORS["muted"],
                linestyle="--", linewidth=0.9, label="Mean")
    ax2.legend(fontsize=8, facecolor=COLORS["card"], labelcolor=COLORS["text"])
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f}", va="center", fontsize=8, color=COLORS["muted"])

    # ── 3. Bar — Accuracy comparison ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    model_names = ["Random\nForest", "Logistic\nRegression"]
    accs = [metrics["acc_rf"] * 100, metrics["acc_lr"] * 100]
    xpos = np.arange(len(model_names))
    bars2 = ax3.bar(xpos, accs,
                    color=[COLORS["rf"], COLORS["lr"]],
                    edgecolor=COLORS["bg"], width=0.45)
    ax3.set_ylim(75, 102)
    ax3.set_xticks(xpos); ax3.set_xticklabels(model_names, fontsize=9)
    ax3.set_ylabel("Accuracy (%)", fontsize=9)
    ax3.set_title("Model Accuracy Comparison", fontweight="bold",
                  color=COLORS["text"], pad=12)
    for bar, acc in zip(bars2, accs):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{acc:.2f}%", ha="center", va="bottom",
                 fontweight="bold", fontsize=10, color=COLORS["text"])

    # ── 4. Confusion matrix — RF ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    cm_rf = metrics["cm_rf"]
    im = ax4.imshow(cm_rf, cmap="Blues", aspect="auto")
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Good", "Defective"], fontsize=9)
    ax4.set_yticklabels(["Good", "Defective"], fontsize=9)
    ax4.set_xlabel("Predicted", fontsize=9)
    ax4.set_ylabel("Actual", fontsize=9)
    ax4.set_title("Confusion Matrix — RF", fontweight="bold",
                  color=COLORS["text"], pad=12)
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm_rf[i, j]), ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if cm_rf[i, j] > cm_rf.max() / 2 else COLORS["text"])

    # ── 5. Histogram — Defect density ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    good_dd = df.loc[df["defective"] == 0, "defect_density"]
    bad_dd  = df.loc[df["defective"] == 1, "defect_density"]
    ax5.hist(good_dd, bins=40, alpha=0.75, color=COLORS["good"],
             label="Good", density=True)
    ax5.hist(bad_dd, bins=40, alpha=0.75, color=COLORS["defective"],
             label="Defective", density=True)
    ax5.axvline(0.025, color="white", linestyle="--",
                linewidth=1.2, label="Threshold (0.025)")
    ax5.set_xlabel("Defect Density (cm⁻²)", fontsize=9)
    ax5.set_ylabel("Probability Density", fontsize=9)
    ax5.set_title("Defect Density Distribution", fontweight="bold",
                  color=COLORS["text"], pad=12)
    ax5.legend(fontsize=8, facecolor=COLORS["card"], labelcolor=COLORS["text"])

    # ── 6. Scatter — Thickness vs Temperature ─────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    for cls, color, lbl in [(0, COLORS["good"], "Good"),
                             (1, COLORS["defective"], "Defective")]:
        mask = df["defective"] == cls
        ax6.scatter(df.loc[mask, "thickness"],
                    df.loc[mask, "temperature"],
                    c=color, alpha=0.3, s=6, label=lbl)
    ax6.set_xlabel("Thickness (Å)", fontsize=9)
    ax6.set_ylabel("Temperature (K)", fontsize=9)
    ax6.set_title("Thickness vs Temperature", fontweight="bold",
                  color=COLORS["text"], pad=12)
    ax6.legend(fontsize=8, facecolor=COLORS["card"], labelcolor=COLORS["text"])

    # ── Optional row 3 — new wafer prediction viz ─────────────
    if pred_result:
        ax7 = fig.add_subplot(gs[2, :])
        ax7.set_facecolor(COLORS["bg"])
        ax7.axis("off")

        result_color = COLORS["defective"] if pred_result["rf_label"] == "DEFECTIVE" else COLORS["good"]
        ax7.text(0.5, 0.85,
                 f"NEW WAFER  →  {pred_result['rf_label']}",
                 ha="center", va="center", fontsize=18, fontweight="bold",
                 color=result_color, transform=ax7.transAxes,
                 fontfamily="monospace")
        ax7.text(0.5, 0.55,
                 f"Random Forest confidence: {pred_result['rf_confidence']:.1f}%   |   "
                 f"Logistic Regression confidence: {pred_result['lr_confidence']:.1f}%",
                 ha="center", va="center", fontsize=11,
                 color=COLORS["muted"], transform=ax7.transAxes)
        ax7.text(0.5, 0.25,
                 f"P(Good) = {pred_result['rf_prob_good']:.1f}%   ·   "
                 f"P(Defective) = {pred_result['rf_prob_def']:.1f}%",
                 ha="center", va="center", fontsize=10,
                 color=COLORS["text"], transform=ax7.transAxes)

    return fig


def fig_to_bytes(fig) -> bytes:
    """Convert a matplotlib Figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════
# APP — BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════
rf, lr, scaler, metrics = train_models()
df = generate_data()

# ─── HEADER ──────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div class="header-title">🔬 Wafer Defect Detection</div>
  <div class="header-sub">Semiconductor Yield Prediction  ·  200mm CMOS Process  ·  AI-Powered Inline Inspection</div>
</div>
""", unsafe_allow_html=True)

# ─── QUICK STATS ROW ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
defect_rate = df["defective"].mean() * 100
total = len(df)

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Training Samples</div>
      <div class="metric-value">{total:,}</div>
      <div class="metric-sub">Synthetic 200mm wafers</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">RF Accuracy</div>
      <div class="metric-value">{metrics['acc_rf']*100:.2f}%</div>
      <div class="metric-sub">Random Forest (200 trees)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">LR Accuracy</div>
      <div class="metric-value">{metrics['acc_lr']*100:.2f}%</div>
      <div class="metric-sub">Logistic Regression baseline</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Dataset Defect Rate</div>
      <div class="metric-value">{defect_rate:.1f}%</div>
      <div class="metric-sub">Of synthetic wafers</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT — SIDEBAR + CONTENT
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙ Wafer Parameters")
    st.markdown("<small style='color:#8b949e'>Enter process measurement values below. Defaults represent a sample wafer from the fab floor.</small>", unsafe_allow_html=True)
    st.markdown("---")

    thickness = st.number_input(
        "Thickness (Å)",
        min_value=85.0, max_value=115.0, value=101.2, step=0.1,
        format="%.1f",
        help="Target: 100.0 Å  ·  Spec limit: ±5 Å"
    )
    doping = st.number_input(
        "Doping Concentration (×10¹⁵ cm⁻³)",
        min_value=0.5, max_value=1.5, value=1.02, step=0.01,
        format="%.3f",
        help="Value is multiplied by 1×10¹⁵. Target: 1.000  ·  Limit: ±0.15"
    )
    temperature = st.number_input(
        "Temperature (K)",
        min_value=265.0, max_value=335.0, value=308.5, step=0.1,
        format="%.1f",
        help="Target: 300 K  ·  Spec limit: ±12 K"
    )
    defect_density = st.number_input(
        "Defect Density (cm⁻²)",
        min_value=0.000, max_value=0.100, value=0.031, step=0.001,
        format="%.4f",
        help="Good wafer: < 0.025 cm⁻²"
    )

    st.markdown("---")

    # Spec check indicators
    st.markdown("**📋 Spec Check**")
    checks = {
        "Thickness": abs(thickness - 100.0) <= 5.0,
        "Doping":    abs(doping * 1e15 - 1e15) <= 1.5e14,
        "Temperature": abs(temperature - 300.0) <= 12.0,
        "Defect Density": defect_density <= 0.025,
    }
    for param, ok in checks.items():
        icon = "✅" if ok else "⚠️"
        color = "#2ea043" if ok else "#f85149"
        st.markdown(
            f"<span style='color:{color};font-size:0.85rem'>{icon} {param}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Wafer Quality", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**ℹ️ Spec Limits**")
    st.markdown("""
<small style='color:#6e7681;line-height:1.9'>
Thickness: 95–105 Å<br>
Doping: (0.85–1.15) ×10¹⁵<br>
Temperature: 288–312 K<br>
Defect Density: < 0.025 cm⁻²
</small>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PREDICTION OUTPUT
# ─────────────────────────────────────────────────────────────────
pred_result = None

if predict_btn:
    doping_actual = doping * 1e15
    pred_result = predict_wafer(rf, lr, scaler, thickness, doping_actual, temperature, defect_density)

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    col_rf, col_lr = st.columns(2)
    with col_rf:
        label_rf = pred_result["rf_label"]
        css_cls  = "pred-defective" if label_rf == "DEFECTIVE" else "pred-good"
        icon     = "⚠️" if label_rf == "DEFECTIVE" else "✅"
        color    = "#f85149" if label_rf == "DEFECTIVE" else "#2ea043"
        st.markdown(f"""
        <div class="{css_cls}">
          <div class="pred-label" style="color:{color}">{icon} {label_rf}</div>
          <div class="pred-conf">Random Forest  ·  Confidence: {pred_result['rf_confidence']:.1f}%</div>
          <div style="margin-top:10px;font-size:0.82rem;color:#8b949e">
            P(Good) = {pred_result['rf_prob_good']:.1f}% &nbsp;|&nbsp; P(Defective) = {pred_result['rf_prob_def']:.1f}%
          </div>
        </div>""", unsafe_allow_html=True)

    with col_lr:
        label_lr = pred_result["lr_label"]
        css_cls2  = "pred-defective" if label_lr == "DEFECTIVE" else "pred-good"
        icon2     = "⚠️" if label_lr == "DEFECTIVE" else "✅"
        color2    = "#f85149" if label_lr == "DEFECTIVE" else "#2ea043"
        st.markdown(f"""
        <div class="{css_cls2}">
          <div class="pred-label" style="color:{color2}">{icon2} {label_lr}</div>
          <div class="pred-conf">Logistic Regression  ·  Confidence: {pred_result['lr_confidence']:.1f}%</div>
          <div style="margin-top:10px;font-size:0.82rem;color:#8b949e">
            P(Good) = {pred_result['lr_prob_good']:.1f}% &nbsp;|&nbsp; P(Defective) = {pred_result['lr_prob_def']:.1f}%
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Agreement banner
    agree = pred_result["rf_label"] == pred_result["lr_label"]
    if agree:
        st.success(f"✅ Both models agree: **{pred_result['rf_label']}**")
    else:
        st.warning("⚠️ Models disagree. Random Forest result is primary — review manually.")

    st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# DASHBOARD + FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Results Dashboard", "📈 Feature Importance", "📋 Model Report"])

with tab1:
    st.markdown('<div class="section-title">Process Analytics Dashboard</div>', unsafe_allow_html=True)

    with st.spinner("Rendering dashboard..."):
        fig = build_dashboard(df, metrics, pred_result)
        st.pyplot(fig, use_container_width=True)
        png_bytes = fig_to_bytes(fig)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇ Download Dashboard (PNG)",
        data=png_bytes,
        file_name="wafer_defect_dashboard.png",
        mime="image/png",
        use_container_width=True,
    )

with tab2:
    st.markdown('<div class="section-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    importances = metrics["importances"]
    labels      = metrics["feature_labels"]
    total_imp   = importances.sum()
    idx_sorted  = np.argsort(importances)[::-1]

    bar_palette = [COLORS["defective"], COLORS["rf"], "#e67e22", "#2ea043"]
    for rank, i in enumerate(idx_sorted):
        pct  = importances[i] / total_imp * 100
        norm = importances[i] / importances.max() * 100
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
            <div class="fi-bar-wrap">
              <div class="fi-label">#{rank+1} &nbsp; {labels[i]}</div>
              <div class="fi-bar-bg">
                <div class="fi-bar-fill" style="width:{norm:.1f}%;background:{bar_palette[rank]}"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="padding-top:18px">
              <span class="fi-score">{importances[i]:.4f}</span><br>
              <span style="font-size:0.72rem;color:#6e7681">{pct:.1f}% of total</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)

    def report_table(report_dict, title):
        rows = []
        for cls in ["Good", "Defective"]:
            r = report_dict[cls]
            rows.append({
                "Class": cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1-Score":  f"{r['f1-score']:.3f}",
                "Support":   int(r['support']),
            })
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_t1:
        report_table(metrics["report_rf"], "Random Forest")
    with col_t2:
        report_table(metrics["report_lr"], "Logistic Regression")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#6e7681;font-size:0.78rem;font-family:'IBM Plex Mono',monospace;padding:12px 0">
  AI-Based Wafer Defect Detection  ·  200mm CMOS Process Simulation  ·  Random Forest + Logistic Regression
  <br>Built with Streamlit &amp; scikit-learn  ·  Synthetic data for demonstration
</div>
""", unsafe_allow_html=True)
