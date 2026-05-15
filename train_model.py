import pandas as pd
import re
import pickle
import nltk
import kagglehub
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Step 1: Load Dataset
print("Loading dataset...")
path = kagglehub.dataset_download("kazanova/sentiment140")
df = pd.read_csv(f"{path}/training.1600000.processed.noemoticon.csv",
                 encoding='latin-1', header=None,
                 names=['target', 'id', 'date', 'flag', 'user', 'text'])

# Map target: 0 = Negative, 4 -> 1 = Positive
df['target'] = df['target'].replace(4, 1)

# Using 50k samples for speed
df = df.sample(n=50000, random_state=42).reset_index(drop=True)
print(f"Using {len(df)} samples")

# Step 2: Preprocessing the data
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()                                    # Lowercasing
    text = re.sub(r'http\S+|@\w+|[^a-zA-Z\s]', '', text)   # Removing URLs, mentions, and special characters
    tokens = word_tokenize(text)                           # Tokenization
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]  # Stopword removal
    return ' '.join(tokens)

print("Preprocessing...")
df['clean'] = df['text'].apply(preprocess)
df = df[df['clean'].str.strip() != '']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['target'], test_size=0.2, random_state=42
)

# Step 4: Build Pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Step 5: Training the model
print("Training...")
pipeline.fit(X_train, y_train)

# Step 6: Evaluating the model
y_pred = pipeline.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Step 7: Saving the model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Model saved as model.pkl")
