import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
        border: 1px solid #2a2f3e;
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    .hero h1 { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero p  { color: #8892a4; font-size: 1rem; margin-top: 8px; }

    .card {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .card h3 { color: #ffffff; font-size: 1rem; font-weight: 600; margin: 0 0 4px 0; }
    .card p  { color: #8892a4; font-size: 0.85rem; margin: 0; }

    .result-fraud {
        background: linear-gradient(135deg, #3b1a1a, #1a0f0f);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-legit {
        background: linear-gradient(135deg, #0f2a1a, #0f1a12);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-title { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .result-sub   { color: #8892a4; font-size: 0.9rem; margin-top: 8px; }

    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-label { color: #8892a4; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }

    .stSlider > div > div { background: #2a2f3e !important; }
    div[data-testid="stNumberInput"] input { background: #1a1f2e !important; color: #fff !important; border-color: #2a2f3e !important; }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    .section-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2f3e;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💳 Credit Card Fraud Detection</h1>
    <p>ML-powered transaction analysis using Random Forest · AUC-PR 0.80 · Precision 91%</p>
</div>
""", unsafe_allow_html=True)

# ── Model stats ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("AUC-PR",    "0.8038", c1),
    ("Precision", "91%",    c2),
    ("Recall",    "76%",    c3),
    ("Threshold", "0.40",   c4),
]
for label, value, col in stats:
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 About")
    st.markdown("""
    This app uses a **Random Forest** model trained on 284,807 real credit card transactions.

    **Pipeline:**
    - SMOTE for class imbalance
    - Stratified K-Fold CV
    - SHAP explainability
    - Threshold tuning (0.40)

    **Key insight:** Only 0.17% of transactions are fraud — accuracy is meaningless. We optimize for **AUC-PR**.
    """)
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("📂 [GitHub Repository](https://github.com/zyadragab/credit-card-fraud-detection)")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Transaction Input</div>', unsafe_allow_html=True)
st.markdown("Enter the transaction features below. V1–V28 are PCA-transformed (confidential). Adjust `Time` and `Amount` directly.")

with st.form("transaction_form"):
    col_time, col_amount = st.columns(2)
    with col_time:
        time_val   = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
    with col_amount:
        amount_val = st.number_input("Amount ($)", value=0.0, step=0.01, min_value=0.0)

    st.markdown("**PCA Features (V1 – V28)**")

    v_vals = []
    cols_per_row = 4
    v_features = [f"V{i}" for i in range(1, 29)]
    rows = [v_features[i:i+cols_per_row] for i in range(0, len(v_features), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, feat in zip(cols, row):
            with col:
                v_vals.append(st.number_input(feat, value=0.0, step=0.01, key=feat))

    submitted = st.form_submit_button("🔍 Analyze Transaction")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Build input row
    input_data = pd.DataFrame([[time_val, amount_val] + v_vals],
                               columns=['Time', 'Amount'] + v_features)

    # Scale Time and Amount
    input_data[['Time', 'Amount']] = scaler.transform(input_data[['Time', 'Amount']])

    # Reorder columns to match training
    feature_order = ['Time'] + v_features + ['Amount']
    input_data = input_data[feature_order]

    # Predict
    prob      = model.predict_proba(input_data)[0][1]
    threshold = 0.40
    prediction = int(prob >= threshold)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Result</div>', unsafe_allow_html=True)

    res_col, gauge_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-fraud">
                <div class="result-title" style="color:#ef4444;">🚨 FRAUD DETECTED</div>
                <div class="result-sub">This transaction has been flagged as fraudulent</div>
                <br>
                <div style="color:#ef4444; font-size:2rem; font-weight:700;">{prob*100:.1f}%</div>
                <div style="color:#8892a4; font-size:0.85rem;">Fraud Probability</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legit">
                <div class="result-title" style="color:#22c55e;">✅ LEGITIMATE</div>
                <div class="result-sub">This transaction appears to be legitimate</div>
                <br>
                <div style="color:#22c55e; font-size:2rem; font-weight:700;">{(1-prob)*100:.1f}%</div>
                <div style="color:#8892a4; font-size:0.85rem;">Legit Probability</div>
            </div>""", unsafe_allow_html=True)

    with gauge_col:
        # Probability gauge
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')

        bar_color = '#ef4444' if prediction == 1 else '#22c55e'
        ax.barh(['Fraud Probability'], [prob],        color=bar_color, height=0.4, alpha=0.9)
        ax.barh(['Fraud Probability'], [1 - prob],    color='#2a2f3e', height=0.4,
                left=prob, alpha=0.5)
        ax.axvline(x=threshold, color='#f59e0b', linestyle='--', linewidth=1.5,
                   label=f'Threshold ({threshold})')

        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability', color='#8892a4', fontsize=9)
        ax.tick_params(colors='#8892a4', labelsize=8)
        ax.spines[:].set_color('#2a2f3e')
        ax.legend(fontsize=8, facecolor='#1a1f2e', labelcolor='#8892a4',
                  edgecolor='#2a2f3e')
        ax.set_title('Fraud Probability Score', color='#ffffff', fontsize=10, pad=10)

        st.pyplot(fig)
        plt.close()

    # Summary metrics
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Fraud Probability</div>
            <div class="metric-value" style="color:{'#ef4444' if prediction==1 else '#22c55e'};">{prob*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Decision Threshold</div>
            <div class="metric-value" style="color:#f59e0b;">0.40</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Verdict</div>
            <div class="metric-value" style="color:{'#ef4444' if prediction==1 else '#22c55e'};">{'FRAUD' if prediction==1 else 'LEGIT'}</div>
        </div>""", unsafe_allow_html=True)
