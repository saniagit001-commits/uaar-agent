"""
UAAR Info Agent — Training & Evaluation Script
Run:  python train.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "uaar_dataset.csv")

print("=" * 60)
print("  UAAR Info Agent — Naive Bayes Training & Evaluation")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\n[1] Dataset: {len(df)} rows, {df['intent'].nunique()} intent classes")
print(f"    Classes: {sorted(df['intent'].unique())}")

corpus = []
for _, row in df.iterrows():
    tags = str(row.get("tags", "")).replace(",", " ")
    corpus.append(f"{row['question']} {tags}".lower())

le = LabelEncoder()
y  = le.fit_transform(df["intent"])

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                              max_df=0.95, min_df=1)
X = vectorizer.fit_transform(corpus)
print(f"\n[2] TF-IDF matrix: {X.shape}, Vocabulary: {len(vectorizer.vocabulary_)} n-grams")

# Use stratify only when every class has >= 2 samples
import collections
class_counts = collections.Counter(y)
can_stratify = all(c >= 2 for c in class_counts.values())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42,
    stratify=y if can_stratify else None)
if not can_stratify:
    print("    [Note] Stratification skipped — some classes have < 2 samples.")
print(f"\n[3] Split: {X_train.shape[0]} train / {X_test.shape[0]} test (80/20)")

model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)
print("\n[4] Naive Bayes trained (Laplace smoothing alpha=0.5)")

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\n[5] Test Accuracy: {acc * 100:.1f}%")

print("\n[6] Classification Report:")
print("-" * 55)
present_labels = sorted(set(y_test) | set(y_pred))
present_names  = le.inverse_transform(present_labels)
print(classification_report(y_test, y_pred,
      labels=present_labels, target_names=present_names, zero_division=0))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"[7] 5-Fold Cross-Validation: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"    Folds: {[f'{s*100:.1f}%' for s in cv_scores]}")

print("\n[8] Live Demo Predictions:")
print("-" * 55)
demos = [
    "When is the entry test?",
    "Eligibility for Faculty of Sciences?",
    "Tell me about Dr. Yaser Hafeez",
    "What scholarships does UAAR offer?",
    "How do I apply for admission?",
    "Who is the Vice Chancellor?",
    "What facilities are in the hostel?",
    "Can I do PhD at UAAR?",
]
for q in demos:
    vec   = vectorizer.transform([q.lower()])
    probs = np.exp(model.predict_log_proba(vec))[0]
    pred  = le.inverse_transform([np.argmax(probs)])[0]
    conf  = float(np.max(probs))
    print(f"  Q: {q}")
    print(f"     Intent: {pred:<20}  Confidence: {conf:.2%}\n")

print("=" * 60)
print("  Done! Now run:  python visualize.py  for graphs")
print("  Then run:       python app.py        for chat")
print("=" * 60)