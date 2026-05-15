import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
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

# App
st.title("Sentiment Analyzer")
st.write("Enter text below to predict if it's **Positive** or **Negative**")

st.markdown("---")
st.caption("CA2 | Anmol Mishra | Reg No: 12313922")
st.markdown("---")

# Example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Try Positive Example"):
        st.session_state.example_text = "I am so happy today! Everything is going great and I love it."
with col2:
    if st.button("Try Negative Example"):
        st.session_state.example_text = "This is really bad. I am so disappointed and upset."

default_text = st.session_state.get("example_text", "")
user_input = st.text_area("Enter text:", value=default_text)

if st.button("Predict"):
    if user_input.strip():
        cleaned = preprocess(user_input)
        prediction = model.predict([cleaned])[0]
        result = "Positive" if prediction == 1 else "Negative"
        st.subheader(f"Result: {result}")
    else:
        st.warning("Please enter some text.")
