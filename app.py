"""
CSI324: Text Analytics - Streamlit Web Application
Sentiment Analysis using Trained ML Pipeline
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import time

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Sentiment Analyzer | CSI324",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Custom CSS for Premium Design
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Title */
    .hero-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-top: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #8b8fa3;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        letter-spacing: 0.02em;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border: 1px solid rgba(102, 126, 234, 0.3);
        text-align: center;
        margin: 0 auto;
        display: block;
        width: fit-content;
    }
    
    /* Input Card */
    .input-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .input-label {
        color: #c4c7d4;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
    }
    
    /* Result Cards */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .result-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .result-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .result-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .result-positive .result-label { color: #10b981; }
    .result-negative .result-label { color: #ef4444; }
    
    .confidence-text {
        color: #8b8fa3;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    .confidence-value {
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .result-positive .confidence-value { color: #34d399; }
    .result-negative .confidence-value { color: #f87171; }
    
    /* Confidence Bar */
    .confidence-bar-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        height: 10px;
        margin: 1rem auto;
        max-width: 300px;
        overflow: hidden;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 50px;
        transition: width 0.8s ease-out;
    }
    
    .bar-positive { background: linear-gradient(90deg, #10b981, #34d399); }
    .bar-negative { background: linear-gradient(90deg, #ef4444, #f87171); }
    
    /* Pipeline Info */
    .pipeline-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .pipeline-title {
        color: #667eea;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    
    .pipeline-step {
        display: flex;
        align-items: center;
        padding: 0.6rem 0;
        color: #9ca0b4;
        font-size: 0.9rem;
    }
    
    .step-number {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.8rem;
        flex-shrink: 0;
    }
    
    /* Textarea styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #e2e4ed !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #5a5e72 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Examples section */
    .examples-title {
        color: #6b6f85;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.8rem;
        margin-top: 1.5rem;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: #4a4e63;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metrics Row */
    .metrics-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        text-align: center;
    }
    
    .metric-value {
        color: #e2e4ed;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    .metric-label {
        color: #6b6f85;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Preprocessing Function
# ============================================================
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text: lowercase, remove noise, tokenize, remove stopwords"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)


# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained ML pipeline from model.pkl"""
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ============================================================
# App Layout
# ============================================================

# Badge
st.markdown('<div class="badge">CSI324 · Text Analytics</div>', unsafe_allow_html=True)

# Hero
st.markdown('<h1 class="hero-title">Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Powered by TF-IDF & Logistic Regression · Trained on 50K tweets</p>', unsafe_allow_html=True)

# Model Status
if not model_loaded:
    st.error(f"⚠️ Model not found. Please run `python train_model.py` first to generate `model.pkl`.\n\nError: {model_error}")
    st.stop()

# Input Section
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="input-label">📝 Enter your text</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="Text input",
    placeholder="Type or paste any text here to analyze its sentiment...",
    height=120,
    label_visibility="collapsed",
    key="text_input"
)

# Example buttons
st.markdown('<div class="examples-title">💡 Try these examples</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😊 Positive", key="pos_ex", use_container_width=True):
        st.session_state.text_input = "I absolutely love this! Best experience ever, would highly recommend to everyone."
        st.rerun()
with col2:
    if st.button("😞 Negative", key="neg_ex", use_container_width=True):
        st.session_state.text_input = "Terrible and disappointing. Complete waste of time and money, never again."
        st.rerun()
with col3:
    if st.button("🤔 Neutral", key="neu_ex", use_container_width=True):
        st.session_state.text_input = "The package arrived today. It contains the items I ordered last week."
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Analyze Button
analyze = st.button("🔍  Analyze Sentiment", key="analyze_btn", use_container_width=True)

# ============================================================
# Prediction & Results
# ============================================================
if analyze and user_input.strip():
    # Preprocess
    cleaned_text = preprocess_text(user_input)
    
    if cleaned_text.strip() == '':
        st.warning("⚠️ After preprocessing, no meaningful text remains. Please enter more descriptive text.")
    else:
        # Predict
        with st.spinner(""):
            time.sleep(0.3)  # Brief pause for UX
            prediction = model.predict([cleaned_text])[0]
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = max(probabilities) * 100
            neg_prob = probabilities[0] * 100
            pos_prob = probabilities[1] * 100
        
        # Display Result
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-positive">
                <div class="result-emoji">😊</div>
                <div class="result-label">Positive Sentiment</div>
                <div class="confidence-text">Confidence: <span class="confidence-value">{confidence:.1f}%</span></div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar bar-positive" style="width: {confidence}%"></div>
                </div>
                <div class="metrics-row">
                    <div class="metric-item">
                        <div class="metric-value" style="color: #10b981;">{pos_prob:.1f}%</div>
                        <div class="metric-label">Positive</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" style="color: #ef4444;">{neg_prob:.1f}%</div>
                        <div class="metric-label">Negative</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-negative">
                <div class="result-emoji">😞</div>
                <div class="result-label">Negative Sentiment</div>
                <div class="confidence-text">Confidence: <span class="confidence-value">{confidence:.1f}%</span></div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar bar-negative" style="width: {confidence}%"></div>
                </div>
                <div class="metrics-row">
                    <div class="metric-item">
                        <div class="metric-value" style="color: #10b981;">{pos_prob:.1f}%</div>
                        <div class="metric-label">Positive</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" style="color: #ef4444;">{neg_prob:.1f}%</div>
                        <div class="metric-label">Negative</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show preprocessing details in expander
        with st.expander("🔬 View Preprocessing Details"):
            st.markdown(f"**Original Text:**")
            st.code(user_input, language=None)
            st.markdown(f"**After Preprocessing:**")
            st.code(cleaned_text, language=None)
            st.markdown(f"**Steps Applied:**")
            st.markdown("""
            1. ✅ Lowercasing  
            2. ✅ URL & mention removal  
            3. ✅ Special character removal  
            4. ✅ Tokenization (NLTK)  
            5. ✅ Stopword removal  
            6. ✅ TF-IDF vectorization  
            """)

elif analyze and not user_input.strip():
    st.warning("⚠️ Please enter some text to analyze.")


# ============================================================
# Pipeline Architecture Section
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-card">
    <div class="pipeline-title">⚙️ ML Pipeline Architecture</div>
    <div class="pipeline-step">
        <span class="step-number">1</span>
        <span>Text Input → Lowercasing & Cleaning</span>
    </div>
    <div class="pipeline-step">
        <span class="step-number">2</span>
        <span>Tokenization (NLTK word_tokenize)</span>
    </div>
    <div class="pipeline-step">
        <span class="step-number">3</span>
        <span>Stopword Removal (English)</span>
    </div>
    <div class="pipeline-step">
        <span class="step-number">4</span>
        <span>TF-IDF Vectorization (50K features, bigrams)</span>
    </div>
    <div class="pipeline-step">
        <span class="step-number">5</span>
        <span>Logistic Regression Classifier</span>
    </div>
    <div class="pipeline-step">
        <span class="step-number">6</span>
        <span>Sentiment Prediction (Positive / Negative)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    CSI324: Text Analytics · Practical Exam<br>
    Dataset: Sentiment140 (1.6M tweets) · Model: TF-IDF + Logistic Regression<br>
    Built with Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
