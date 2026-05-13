import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

/* ══ SKY BLUE EVERYWHERE ══ */
.stApp,
html, body,
.block-container,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: #87CEEB !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 800px !important;
}

/* ══ ALL TEXT WHITE BY DEFAULT ══ */
p, span, div, li, label, caption,
.stMarkdown, .stText,
[class*="css"] {
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}

/* ══ HEADINGS: BOLD GREEN ══ */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    font-family: 'Poppins', sans-serif !important;
    color: #00c853 !important;
    font-weight: 800 !important;
    -webkit-text-fill-color: #00c853 !important;
}

/* ══ HERO BANNER ══ */
.hero-banner {
    background: linear-gradient(135deg, #1565c0 0%, #0d8c4a 100%);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    border: 2px solid rgba(255,255,255,0.3);
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #00e676 !important;
    -webkit-text-fill-color: #00e676 !important;
    margin: 0 0 0.4rem 0;
    line-height: 1.25;
}
.hero-sub {
    color: #ffffff !important;
    font-size: 0.98rem;
    opacity: 0.95;
    margin: 0;
}

/* ══ SECTION CARDS ══ */
.section-card {
    background: rgba(255,255,255,0.15);
    border: 2px solid rgba(255,255,255,0.35);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(6px);
}
.card-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #00e676 !important;
    border-bottom: 2px solid rgba(0,230,118,0.4);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ══ RESULT CARDS ══ */
.result-positive {
    background: rgba(0, 200, 83, 0.25);
    border: 2.5px solid #00c853;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin: 0.8rem 0;
}
.result-positive .res-label {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #00e676 !important;
}
.result-positive .res-sub {
    color: #ccffe0 !important;
    font-size: 0.9rem;
    margin-top: 0.2rem;
}
.result-negative {
    background: rgba(229, 57, 53, 0.2);
    border: 2.5px solid #ef5350;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin: 0.8rem 0;
}
.result-negative .res-label {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #ff8a80 !important;
}
.result-negative .res-sub {
    color: #ffd0cc !important;
    font-size: 0.9rem;
    margin-top: 0.2rem;
}

/* ══ CONFIDENCE BAR ══ */
.conf-wrap {
    background: rgba(255,255,255,0.2);
    border-radius: 30px;
    height: 14px;
    width: 100%;
    margin: 0.6rem 0 0.2rem 0;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.3);
}
.conf-fill {
    height: 14px;
    border-radius: 30px;
    background: linear-gradient(90deg, #1565c0, #00c853);
}
.conf-text {
    font-family: 'Poppins', sans-serif;
    font-size: 0.82rem;
    color: #ffffff !important;
    font-weight: 600;
    text-align: right;
    margin-top: 2px;
}

/* ══ INPUTS ══ */
.stTextArea textarea {
    background: rgba(255,255,255,0.9) !important;
    border: 2px solid rgba(255,255,255,0.7) !important;
    border-radius: 12px !important;
    color: #0d1b2a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}
.stTextArea textarea::placeholder {
    color: #546e7a !important;
}
.stTextArea textarea:focus {
    border-color: #00c853 !important;
    box-shadow: 0 0 0 3px rgba(0,200,83,0.2) !important;
}
/* selectbox container */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.9) !important;
    border: 2px solid rgba(255,255,255,0.7) !important;
    border-radius: 10px !important;
    color: #0d1b2a !important;
}
/* dropdown list */
[data-baseweb="select"] ul,
[data-baseweb="popover"] {
    background: #1a4a7a !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] li {
    color: #ffffff !important;
}
[data-baseweb="select"] li:hover {
    background: #00c853 !important;
}

/* ══ BUTTON ══ */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1565c0 0%, #00c853 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.35) !important;
}

/* ══ METRICS ══ */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.18) !important;
    border: 2px solid rgba(255,255,255,0.4) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}
[data-testid="stMetricValue"] {
    color: #00e676 !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800 !important;
    -webkit-text-fill-color: #00e676 !important;
}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d47a1 0%, #1565c0 50%, #0a6e3a 100%) !important;
    border-right: 2px solid rgba(255,255,255,0.2) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] caption {
    color: #ffffff !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #00e676 !important;
    -webkit-text-fill-color: #00e676 !important;
    font-weight: 800 !important;
}

/* ══ EXPANDER ══ */
details > summary {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
}
details > summary:hover {
    background: rgba(255,255,255,0.25) !important;
}

/* ══ TABLE ══ */
table { border-collapse: collapse; width: 100%; border-radius: 10px; overflow: hidden; }
th {
    background: linear-gradient(90deg,#1565c0,#0d8c4a) !important;
    color: #ffffff !important;
    padding: 10px 14px !important;
    font-family:'Poppins',sans-serif;
    font-weight: 700;
}
td {
    background: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid rgba(255,255,255,0.2) !important;
}

/* ══ WARNINGS / ALERTS ══ */
.stAlert p { color: #ffffff !important; }
.stWarning { background: rgba(255,193,7,0.25) !important; border-left: 4px solid #ffc107 !important; border-radius: 10px !important; }
.stError { background: rgba(229,57,53,0.2) !important; border-left: 4px solid #ef5350 !important; border-radius: 10px !important; }

/* ══ SPINNER ══ */
.stSpinner > div { border-top-color: #00c853 !important; }

/* ══ LABEL FIX ══ */
label, .stTextArea label, .stSelectbox label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'outputs', 'models', 'sentiment_model.pkl')
    if not os.path.exists(model_path):
        model_path = 'sentiment_model.pkl'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Project Info")
    st.markdown("""
**Project:** NLP Sentiment Analysis  
**Dataset:** IMDB 50K Movie Reviews  
**Model:** LinearSVC + TF-IDF  
**Accuracy:** ~93.8% F1 Score  

---
**Phases Completed:**  
✅ Phase A — Data Setup  
✅ Phase B — EDA & Visualization  
✅ Phase C — NLP Preprocessing  
✅ Phase D — Model Training  
✅ Phase E — Streamlit Deployment  

---
**Healthcare Application:**  
This NLP technique can analyze patient feedback and healthcare survey responses to identify satisfaction trends.
""")
    st.markdown("---")
    st.caption("Built by Karthikeyan | Internship 2026")

# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎬 IMDB Movie Review Sentiment Analyzer</div>
    <div class="hero-sub">Powered by LinearSVC + TF-IDF &nbsp;|&nbsp; 93.8% F1 Score &nbsp;|&nbsp; 50K IMDB Reviews</div>
</div>
""", unsafe_allow_html=True)

# ─── Model check ─────────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model file not found! Run Phase D first to generate `outputs/models/sentiment_model.pkl`.")
    st.stop()

# ─── Input Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="card-title">✍️ Enter a Movie Review</div>', unsafe_allow_html=True)

example_reviews = {
    "Select an example...": "",
    "😊 Positive example": "This movie was absolutely brilliant! The acting was superb and the story kept me on the edge of my seat. I laughed, I cried. Highly recommended!",
    "😞 Negative example": "Terrible film. Complete waste of time and money. The acting was awful, the plot made no sense, and the ending was deeply disappointing.",
    "🤔 Mixed example": "It was okay I guess. Some parts were interesting but overall it felt average. Nothing special but not completely bad either."
}

selected = st.selectbox("Try an example or write your own:", list(example_reviews.keys()))
user_input = st.text_area(
    "Your review:",
    value=example_reviews[selected],
    height=140,
    placeholder="Type your movie review here..."
)
st.markdown('</div>', unsafe_allow_html=True)

# ─── Button ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# ─── Result ───────────────────────────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing your review..."):
            cleaned = clean_text(user_input)
            prediction = model.predict([cleaned])[0]
            try:
                score = model.decision_function([cleaned])[0]
                confidence = min(100, int(abs(score) * 25 + 60))
            except Exception:
                confidence = 85

        is_positive = (prediction == 'positive' or prediction == 1)

        st.markdown('<div class="section-card"><div class="card-title">📊 Result</div>', unsafe_allow_html=True)

        if is_positive:
            st.markdown("""
            <div class="result-positive">
                <div class="res-label">😊 POSITIVE Sentiment</div>
                <div class="res-sub">The model predicts this review is positive.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-negative">
                <div class="res-label">😞 NEGATIVE Sentiment</div>
                <div class="res-sub">The model predicts this review is negative.</div>
            </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Sentiment", "😊 POSITIVE" if is_positive else "😞 NEGATIVE")
        with col_b:
            st.metric("Confidence", f"{confidence}%")

        st.markdown(f"""
        <div style="margin-top:0.8rem;">
            <div style="font-size:0.88rem;color:#ffffff;font-weight:600;margin-bottom:4px;">
                Confidence Score: {confidence}%
            </div>
            <div class="conf-wrap">
                <div class="conf-fill" style="width:{confidence}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 See cleaned text that was analyzed"):
            st.code(cleaned if cleaned else "(empty after cleaning)", language=None)

        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("💡 How it works: HTML removed → lowercased → stopwords removed → lemmatized → LinearSVC predicts from TF-IDF patterns.")

# ─── Performance Table ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📈 Model Performance Summary"):
    st.markdown("""
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 88.5% | 88.4% |
| Multinomial Naive Bayes | 85.2% | 85.1% |
| **LinearSVC (Selected)** | **93.9%** | **93.8%** |

**Winner: LinearSVC** — Best F1 score after GridSearchCV tuning.  
*Trained on 40,000 reviews · Tested on 10,000 reviews*
""")
