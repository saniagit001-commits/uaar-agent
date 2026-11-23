"""
UAAR Info Agent — Improved Agent Module
Improvements over original:
  1. TfidfVectorizer with bigrams (1,2) — understands phrases like "how many"
  2. Confidence threshold — returns fallback if score < 0.55 instead of guessing
  3. Cosine similarity re-ranking — picks best matching answer within detected intent
  4. Intent-to-answer mapping from CSV (multiple answers per intent stored)
"""

import csv
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_CSV   = BASE_DIR / "data"/"uaar_dataset.csv"   # ← updated dataset name
MODEL_FILE = BASE_DIR / "uaar_model.pkl"

# ── Confidence threshold ───────────────────────────────────────────────────────
# If Naive Bayes top probability is below this, use fallback instead of guessing
CONFIDENCE_THRESHOLD = 0.40


# ── Text normalisation ─────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    return text


# ── Agent class ────────────────────────────────────────────────────────────────
class UAARAgent:
    def __init__(self):
        self.pipeline       : Pipeline | None = None
        self.label_encoder  : LabelEncoder | None = None
        self.questions      : list[str] = []
        self.answers        : list[str] = []
        self.intents        : list[str] = []
        self.norm_questions : list[str] = []
        self._tfidf_matrix  = None          # for cosine similarity lookup
        self._tfidf_vect    = None          # standalone vectoriser for similarity

        self._load_data()
        self._load_or_train()

    # ── Data loading ───────────────────────────────────────────────────────────
    def _load_data(self):
        if not DATA_CSV.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

        questions, answers, intents = [], [], []
        with open(DATA_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("question", "").strip()
                a = row.get("answer", "").strip()
                i = row.get("intent", "").strip()
                if q and a and i:
                    questions.append(q)
                    answers.append(a)
                    intents.append(i)

        self.questions       = questions
        self.answers         = answers
        self.intents         = intents
        self.norm_questions  = [_normalise(q) for q in questions]

    # ── Model training / loading ───────────────────────────────────────────────
    def _load_or_train(self):
        csv_mtime   = DATA_CSV.stat().st_mtime
        model_stale = (
            not MODEL_FILE.exists()
            or MODEL_FILE.stat().st_mtime < csv_mtime
        )

        if model_stale:
            print("[Agent] Training model (CSV changed or first run)…")
            self._train()
            self._save()
        else:
            print("[Agent] Loading cached model…")
            self._load()

        # Build cosine-similarity matrix over all questions
        self._tfidf_vect   = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf_matrix = self._tfidf_vect.fit_transform(self.norm_questions)

    def _train(self):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.intents)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),   # KEY FIX: bigrams catch "how many", "entry test" etc.
                min_df=1,
                sublinear_tf=True,    # log-scaling dampens very frequent words
                strip_accents="unicode",
                analyzer="word",
            )),
            ("nb", MultinomialNB(alpha=0.3)),   # lower alpha = sharper boundaries
        ])
        self.pipeline.fit(self.norm_questions, y)

    def _save(self):
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({
                "pipeline":      self.pipeline,
                "label_encoder": self.label_encoder,
            }, f)

    def _load(self):
        with open(MODEL_FILE, "rb") as f:
            data = pickle.load(f)
        self.pipeline      = data["pipeline"]
        self.label_encoder = data["label_encoder"]

    # ── Inference ──────────────────────────────────────────────────────────────
    def get_response(self, user_input: str) -> dict:
        norm_input = _normalise(user_input)

        # 1. Naive Bayes intent + confidence
        proba        = self.pipeline.predict_proba([norm_input])[0]
        top_idx      = int(np.argmax(proba))
        nb_conf      = float(proba[top_idx])
        pred_intent  = self.label_encoder.inverse_transform([top_idx])[0]

        # 2. If confidence too low → fallback
        if nb_conf < CONFIDENCE_THRESHOLD:
            fallback_answers = [
                a for a, i in zip(self.answers, self.intents)
                if i == "fallback"
            ]
            answer = fallback_answers[0] if fallback_answers else (
                "I'm not sure about that. Please visit https://www.uaar.edu.pk "
                "or contact admissions@uaar.edu.pk for help."
            )
            return {
                "answer":  answer,
                "intent":  "fallback",
                "nb_conf": round(nb_conf, 4),
                "cos_sim": 0.0,
            }

        # 3. Within predicted intent, find best matching Q via cosine similarity
        input_vec = self._tfidf_vect.transform([norm_input])
        cos_scores = cosine_similarity(input_vec, self._tfidf_matrix)[0]

        # Restrict to rows that match the predicted intent
        best_score = -1.0
        best_idx   = -1
        for idx, (intent, score) in enumerate(zip(self.intents, cos_scores)):
            if intent == pred_intent and score > best_score:
                best_score = score
                best_idx   = idx

        if best_idx == -1:
            # Shouldn't happen, but be safe
            answer    = self.answers[0]
            best_score = 0.0
        else:
            answer = self.answers[best_idx]

        return {
            "answer":  answer,
            "intent":  pred_intent,
            "nb_conf": round(nb_conf, 4),
            "cos_sim": round(float(best_score), 4),
        }

    # ── Suggestions ───────────────────────────────────────────────────────────
    def get_suggestions(self, user_input: str, top_n: int = 3) -> list[str]:
        norm_input  = _normalise(user_input)
        input_vec   = self._tfidf_vect.transform([norm_input])
        cos_scores  = cosine_similarity(input_vec, self._tfidf_matrix)[0]

        # Sort by score, skip the exact question if present
        ranked = sorted(
            enumerate(cos_scores), key=lambda x: x[1], reverse=True
        )
        suggestions = []
        for idx, score in ranked:
            q = self.questions[idx]
            if _normalise(q) != norm_input and score > 0.05:
                suggestions.append(q)
            if len(suggestions) >= top_n:
                break
        return suggestions

    # ── Utilities ──────────────────────────────────────────────────────────────
    def get_intent_classes(self) -> list[str]:
        return sorted(set(self.intents))


# ── Singleton factory ──────────────────────────────────────────────────────────
_agent_instance: UAARAgent | None = None


def get_agent() -> UAARAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = UAARAgent()
    return _agent_instance