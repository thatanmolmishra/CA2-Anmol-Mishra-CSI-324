"""
CSI324: Text Analytics - Practical Exam
End-to-End ML Pipeline for Sentiment Classification
Dataset: Sentiment140 (Twitter Sentiment Analysis)
"""

import pandas as pd
import numpy as np
import re
import pickle
import time

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download NLTK data (if not already present)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============================================================
# STEP 1: Load the Dataset
# ============================================================
print("=" * 60)
print("STEP 1: Loading Sentiment140 Dataset")
print("=" * 60)

# Sentiment140 columns: target, id, date, flag, user, text
# target: 0 = negative, 4 = positive
COLUMN_NAMES = ['target', 'id', 'date', 'flag', 'user', 'text']

import kagglehub
dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
csv_path = f"{dataset_path}/training.1600000.processed.noemoticon.csv"

df = pd.read_csv(csv_path, encoding='latin-1', header=None, names=COLUMN_NAMES)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSentiment distribution:")
print(df['target'].value_counts())

# Convert target: 0 -> 0 (Negative), 4 -> 1 (Positive)
df['target'] = df['target'].replace(4, 1)

print(f"\nAfter mapping (0=Negative, 1=Positive):")
print(df['target'].value_counts())

# ============================================================
# STEP 2: Text Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Text Preprocessing")
print("=" * 60)

# Use a sample for faster training (50,000 samples)
SAMPLE_SIZE = 50000
df_sampled = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"Using {SAMPLE_SIZE} samples for training")

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text with:
    - Lowercasing
    - Remove URLs, mentions, hashtags, special characters
    - Tokenization
    - Stopword removal
    """
    # Lowercasing
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol (keep the word)
    text = re.sub(r'#', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return ' '.join(tokens)

print("Preprocessing text data...")
start_time = time.time()
df_sampled['cleaned_text'] = df_sampled['text'].apply(preprocess_text)
elapsed = time.time() - start_time
print(f"Preprocessing completed in {elapsed:.1f} seconds")

# Show sample
print("\nSample preprocessed texts:")
for i in range(3):
    print(f"\n  Original:  {df_sampled['text'][i][:80]}...")
    print(f"  Cleaned:   {df_sampled['cleaned_text'][i][:80]}...")

# Remove empty texts after cleaning
df_sampled = df_sampled[df_sampled['cleaned_text'].str.strip() != ''].reset_index(drop=True)
print(f"\nDataset after cleaning: {len(df_sampled)} samples")

# ============================================================
# STEP 3: Train-Test Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Train-Test Split")
print("=" * 60)

X = df_sampled['cleaned_text']
y = df_sampled['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples")

# ============================================================
# STEP 4: Build ML Pipeline (TF-IDF + Logistic Regression)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Building ML Pipeline")
print("=" * 60)

# Create Pipeline: TF-IDF Vectorizer -> Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ('classifier', LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42
    ))
])

print("Pipeline architecture:")
print(pipeline)

# ============================================================
# STEP 5: Train the Model
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Training the Model")
print("=" * 60)

start_time = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training completed in {train_time:.1f} seconds")

# ============================================================
# STEP 6: Evaluate the Model
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Model Evaluation")
print("=" * 60)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================
# STEP 7: Save the Model as model.pkl
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving Model Pipeline")
print("=" * 60)

model_path = 'model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

import os
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"Model saved to: {model_path}")
print(f"Model file size: {model_size:.2f} MB")

# ============================================================
# STEP 8: Quick Test - Load and Predict
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Quick Verification Test")
print("=" * 60)

# Load the saved model
with open(model_path, 'rb') as f:
    loaded_pipeline = pickle.load(f)

test_texts = [
    "I love this product, it's absolutely amazing!",
    "Terrible experience, worst purchase ever",
    "The weather is nice today, feeling great",
    "I'm so disappointed and frustrated with the service",
    "Best movie I have ever watched, highly recommend"
]

print("\nTest Predictions:")
for text in test_texts:
    cleaned = preprocess_text(text)
    prediction = loaded_pipeline.predict([cleaned])[0]
    proba = loaded_pipeline.predict_proba([cleaned])[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = max(proba) * 100
    print(f"  '{text}'")
    print(f"    → {sentiment} (Confidence: {confidence:.1f}%)")
    print()

print("=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
