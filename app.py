import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (required on Streamlit Cloud)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stTextArea textarea {
        border-radius: 10px;
        font-size: 16px;
    }
    .result-positive {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        color: #155724;
    }
    .result-negative {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎭 Sentiment Analyzer")
st.markdown("Enter any text below to predict whether the sentiment is **Positive** or **Negative**.")
st.markdown("---")

# Examples
with st.expander("💡 Try an example"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state.example_text = "I absolutely love this! It made my day so much better."
    with col2:
        if st.button("😞 Negative Example"):
            st.session_state.example_text = "This is terrible. I'm really disappointed and frustrated."

# Input area
default_text = st.session_state.get("example_text", "")
user_input = st.text_area(
    "Enter your text:",
    value=default_text,
    height=150,
    placeholder="Type or paste any text here..."
)

# Predict
if st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            cleaned = preprocess(user_input)
            prediction = model.predict([cleaned])[0]
            proba = model.predict_proba([cleaned])[0]
            confidence = max(proba) * 100

        if prediction == 1:
            st.markdown(f'<div class="result-positive">✅ Positive Sentiment</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-negative">❌ Negative Sentiment</div>', unsafe_allow_html=True)

        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.progress(confidence / 100)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:0.85rem;'>"
    "Trained on Sentiment140 dataset · Logistic Regression + TF-IDF · ~76% Accuracy"
    "</p>",
    unsafe_allow_html=True
)
