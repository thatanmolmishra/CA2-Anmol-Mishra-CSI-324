# 🎭 Sentiment Analyzer

A machine learning web app that predicts whether a given text has a **Positive** or **Negative** sentiment.

🔗 **[Live Demo →](https://your-app-url.streamlit.app)**

---

## 📌 About

This app uses a **Logistic Regression** model trained on the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset — 1.6 million tweets — to classify text sentiment in real time.

## 🚀 Features

- Real-time sentiment prediction
- Confidence score with progress bar
- Pre-loaded example texts
- Clean, responsive UI

## 🧠 Model Details

| Component | Details |
|-----------|---------|
| Dataset | Sentiment140 (50k samples) |
| Vectorizer | TF-IDF (50k features) |
| Classifier | Logistic Regression |
| Accuracy | ~76% |

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NLTK
- Streamlit

## 🖥️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** `model.pkl` is pre-trained and included in the repo. You don't need to re-run `train_model.py` to use the app.
>
> To retrain: `python train_model.py` (requires `kagglehub` and a Kaggle account)
