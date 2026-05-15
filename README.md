# 🧠 Sentiment Analyzer — CSI324 Text Analytics

An end-to-end Machine Learning pipeline for **Twitter Sentiment Classification** deployed as a **Streamlit web application**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)

---

## 📋 Project Overview

| Component | Details |
|-----------|---------|
| **Dataset** | Sentiment140 (1.6M tweets) |
| **Task** | Binary Sentiment Classification (Positive / Negative) |
| **Preprocessing** | Lowercasing, Tokenization (NLTK), Stopword Removal |
| **Feature Extraction** | TF-IDF Vectorizer (50K features, bigrams) |
| **Model** | Logistic Regression |
| **Deployment** | Streamlit Web App |

---

## 🗂️ Project Structure

```
├── train_model.py      # ML pipeline: data loading, preprocessing, training, saving
├── app.py              # Streamlit web application
├── model.pkl           # Trained ML pipeline (generated after training)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd CA
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train_model.py
```
This will:
- Download the Sentiment140 dataset via KaggleHub
- Preprocess 50,000 tweet samples
- Train a TF-IDF + Logistic Regression pipeline
- Save the model as `model.pkl`

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🔧 ML Pipeline Architecture

```
Text Input
    │
    ▼
┌─────────────────────┐
│  1. Lowercasing      │
│  2. URL Removal      │
│  3. @Mention Removal │
│  4. Special Chars    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. Tokenization     │
│     (NLTK)           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  6. Stopword         │
│     Removal          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  7. TF-IDF           │
│     Vectorization    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  8. Logistic         │
│     Regression       │
└─────────┬───────────┘
          │
          ▼
    Sentiment Output
   (Positive/Negative)
```

---

## 📊 Model Performance

- **Training Samples**: 50,000 tweets
- **Test Split**: 80/20
- **Algorithm**: Logistic Regression (solver=lbfgs, C=1.0)
- **Features**: TF-IDF with unigrams + bigrams (max 50K features)

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Pandas** — Data manipulation
- **NLTK** — Text preprocessing (tokenization, stopwords)
- **Scikit-learn** — TF-IDF, Logistic Regression, Pipeline
- **Streamlit** — Web app deployment
- **Pickle** — Model serialization
- **KaggleHub** — Dataset download

---

## 📝 CSI324: Text Analytics — Practical Exam

**Course**: CSI324 - Text Analytics  
**Task**: Build an end-to-end ML pipeline for text classification and deploy using Streamlit

---

## 📜 License

This project is for academic purposes (CSI324 Practical Exam).
