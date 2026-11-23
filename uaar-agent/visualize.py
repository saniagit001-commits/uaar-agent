"""
UAAR Info Agent — ML Graphs & Visualization Script

Generates 5 professional graphs and saves them as PNG images
inside the  graphs/  folder.

Usage:
    python visualize.py

Output files (inside graphs/ folder):
    1_intent_distribution.png
    2_confusion_matrix.png
    3_precision_recall_f1.png
    4_cross_validation.png
    5_confidence_scores.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.preprocessing import LabelEncoder

# ── colours ──────────────────────────────────────────────────────────
GREEN_DARK  = "#1a5c2a"
GREEN_MID   = "#2e7d3e"
GREEN_LIGHT = "#a8d5b5"
GOLD        = "#f0a500"
BLUE        = "#3a7ebf"
BG          = "#f5f7fa"

GRAPHS_DIR  = os.path.join(os.path.dirname(__file__), "graphs")
DATASET_PATH= os.path.join(os.path.dirname(__file__), "data", "uaar_dataset.csv")

os.makedirs(GRAPHS_DIR, exist_ok=True)

INTENT_LABELS = {
    "admissions"  : "Admissions",
    "eligibility" : "Eligibility",
    "entry_test"  : "Entry Test",
    "departments" : "Departments",
    "faculty_info": "Faculty Info",
    "contact_info": "Contact Info",
    "facilities"  : "Facilities",
    "scholarships": "Scholarships",
    "general"     : "General",
    "greet"       : "Greet",
    "fallback"    : "Fallback",
    "research"    : "Research",
}

def label(code):
    return INTENT_LABELS.get(code, code)


# ── 0. Load & train ──────────────────────────────────────────────────
print("[1/5] Loading dataset and training Naive Bayes model …")

df = pd.read_csv(DATASET_PATH)

corpus = []
for _, row in df.iterrows():
    tags = str(row.get("tags", "")).replace(",", " ")
    corpus.append(f"{row['question']} {tags}".lower())

le   = LabelEncoder()
y    = le.fit_transform(df["intent"])

vec  = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                       max_df=0.95, min_df=1)
X    = vec.fit_transform(corpus)

import collections as _collections
_class_counts = _collections.Counter(y)
_can_stratify = all(c >= 2 for c in _class_counts.values())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42,
    stratify=y if _can_stratify else None)

model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

classes      = le.classes_
class_labels = [label(c) for c in classes]
# Only use labels present in test/pred to avoid size mismatch
_present       = sorted(set(y_test) | set(y_pred))
_present_names = le.inverse_transform(_present)
report_dict    = classification_report(y_test, y_pred,
                                       labels=_present,
                                       target_names=_present_names,
                                       zero_division=0,
                                       output_dict=True)
cv_scores   = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print(f"    Test accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")


# ── helpers ───────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(GRAPHS_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    Saved → graphs/{name}")


def styled_fig(w=10, h=6, title=""):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold",
                     color=GREEN_DARK, y=1.01)
    return fig, ax


# ════════════════════════════════════════════════════════════════════
# GRAPH 1 — Intent Distribution (how many Q&As per intent)
# ════════════════════════════════════════════════════════════════════
print("[2/5] Generating Graph 1 — Intent Distribution …")

intent_counts = df["intent"].value_counts()
readable      = [label(i) for i in intent_counts.index]
colors        = [GREEN_DARK if i % 2 == 0 else GREEN_MID
                 for i in range(len(intent_counts))]

fig, ax = styled_fig(12, 5, "Graph 1 — Dataset: Number of Q&As per Intent Class")
bars = ax.bar(readable, intent_counts.values, color=colors,
              edgecolor="white", linewidth=0.8, width=0.65)

for bar, val in zip(bars, intent_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(val), ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=GREEN_DARK)

ax.set_xlabel("Intent Class", fontsize=12, color="#333")
ax.set_ylabel("Number of Questions", fontsize=12, color="#333")
ax.set_ylim(0, max(intent_counts.values) + 8)
ax.tick_params(axis="x", rotation=30, labelsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

total_patch = mpatches.Patch(color=GREEN_DARK,
                              label=f"Total: {len(df)} Q&A pairs")
ax.legend(handles=[total_patch], fontsize=10, loc="upper right")

save(fig, "1_intent_distribution.png")


# ════════════════════════════════════════════════════════════════════
# GRAPH 2 — Confusion Matrix
# ════════════════════════════════════════════════════════════════════
print("[3/5] Generating Graph 2 — Confusion Matrix …")

cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(11, 8), facecolor=BG)
ax.set_facecolor(BG)

sns.heatmap(cm,
            annot=True, fmt="d",
            xticklabels=class_labels,
            yticklabels=class_labels,
            cmap=sns.light_palette(GREEN_DARK, as_cmap=True),
            linewidths=0.5, linecolor="#ddd",
            ax=ax, cbar_kws={"shrink": 0.75})

ax.set_xlabel("Predicted Intent", fontsize=12, color="#333", labelpad=10)
ax.set_ylabel("Actual Intent",    fontsize=12, color="#333", labelpad=10)
ax.set_title("Graph 2 — Confusion Matrix (Naive Bayes, 20% Test Set)",
             fontsize=13, fontweight="bold", color=GREEN_DARK, pad=14)
ax.tick_params(axis="x", rotation=35, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=9)

save(fig, "2_confusion_matrix.png")


# ════════════════════════════════════════════════════════════════════
# GRAPH 3 — Precision / Recall / F1 per Intent
# ════════════════════════════════════════════════════════════════════
print("[4/5] Generating Graph 3 — Precision / Recall / F1 …")

metrics = {"Precision": [], "Recall": [], "F1-Score": []}
valid_classes = []
for c in classes:
    if c in report_dict and c not in ("accuracy","macro avg","weighted avg"):
        metrics["Precision"].append(report_dict[c]["precision"])
        metrics["Recall"].append(report_dict[c]["recall"])
        metrics["F1-Score"].append(report_dict[c]["f1-score"])
        valid_classes.append(label(c))

x   = np.arange(len(valid_classes))
w   = 0.25
pal = [GREEN_DARK, GOLD, BLUE]

fig, ax = styled_fig(13, 6,
    "Graph 3 — Naive Bayes: Precision, Recall & F1-Score per Intent")

for i, (metric, vals) in enumerate(metrics.items()):
    bars = ax.bar(x + i * w, vals, width=w, label=metric,
                  color=pal[i], edgecolor="white", linewidth=0.7)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#333")

ax.set_xticks(x + w)
ax.set_xticklabels(valid_classes, rotation=30, ha="right", fontsize=9.5)
ax.set_ylabel("Score (0.0 – 1.0)", fontsize=12)
ax.set_ylim(0, 1.18)
ax.legend(fontsize=11, loc="upper right")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35)

save(fig, "3_precision_recall_f1.png")


# ════════════════════════════════════════════════════════════════════
# GRAPH 4 — 5-Fold Cross-Validation Accuracy
# ════════════════════════════════════════════════════════════════════
print("[5/5] Generating Graph 4 — Cross-Validation Accuracy …")

fold_labels = [f"Fold {i+1}" for i in range(len(cv_scores))]
colors_cv   = [GREEN_DARK if s >= cv_scores.mean() else GREEN_LIGHT
               for s in cv_scores]

fig, ax = styled_fig(9, 5,
    "Graph 4 — 5-Fold Cross-Validation Accuracy (Naive Bayes)")

bars = ax.bar(fold_labels, cv_scores * 100,
              color=colors_cv, edgecolor="white", linewidth=0.8, width=0.5)

for bar, s in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{s*100:.1f}%", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=GREEN_DARK)

mean_line = cv_scores.mean() * 100
ax.axhline(mean_line, color=GOLD, linewidth=2,
           linestyle="--", label=f"Mean = {mean_line:.1f}%")

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(max(0, mean_line - 20), 105)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.35)

save(fig, "4_cross_validation.png")


# ════════════════════════════════════════════════════════════════════
# GRAPH 5 — Confidence Scores for Demo Predictions
# ════════════════════════════════════════════════════════════════════
print("[5/5] Generating Graph 5 — Demo Prediction Confidence Scores …")

demo_questions = [
    "How do I apply for admission?",
    "When is the entry test?",
    "Eligibility for Faculty of Sciences?",
    "Tell me about Dr. Yaser Hafeez",
    "What scholarships are available?",
    "Who is the Vice Chancellor?",
    "Does UAAR have a hostel?",
    "Can I do PhD at UAAR?",
    "What programs does UIIT offer?",
    "What is the Cholistan quota?",
]

pred_intents, conf_scores = [], []
for q in demo_questions:
    qvec  = vec.transform([q.lower()])
    probs = np.exp(model.predict_log_proba(qvec))[0]
    best  = int(np.argmax(probs))
    pred_intents.append(label(le.inverse_transform([best])[0]))
    conf_scores.append(float(np.max(probs)) * 100)

short_qs = [q[:38] + "…" if len(q) > 38 else q for q in demo_questions]
bar_colors= [GREEN_DARK if c >= 70 else GOLD if c >= 40 else "#e06060"
             for c in conf_scores]

fig, ax = styled_fig(13, 6,
    "Graph 5 — Naive Bayes Confidence Score for Sample Predictions")

bars = ax.barh(range(len(short_qs)), conf_scores,
               color=bar_colors, edgecolor="white", linewidth=0.7, height=0.6)

for i, (bar, c, intent) in enumerate(zip(bars, conf_scores, pred_intents)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{c:.1f}%  [{intent}]", va="center", fontsize=9.5, color="#333")

ax.set_yticks(range(len(short_qs)))
ax.set_yticklabels(short_qs, fontsize=10)
ax.set_xlabel("Confidence Score (%)", fontsize=12)
ax.set_xlim(0, 130)
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.axvline(70, color=GOLD, linewidth=1.5, linestyle="--",
           label="70% threshold")

patches = [
    mpatches.Patch(color=GREEN_DARK, label="High confidence (≥70%)"),
    mpatches.Patch(color=GOLD,       label="Medium (40–69%)"),
    mpatches.Patch(color="#e06060",  label="Low (<40%)"),
]
ax.legend(handles=patches, fontsize=9.5, loc="lower right")

save(fig, "5_confidence_scores.png")


# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  All 5 graphs saved inside the  graphs/  folder:")
print("    1_intent_distribution.png")
print("    2_confusion_matrix.png")
print("    3_precision_recall_f1.png")
print("    4_cross_validation.png")
print("    5_confidence_scores.png")
print(f"\n  Model Test Accuracy : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"  CV Mean Accuracy    : {cv_scores.mean()*100:.1f}%")
print("=" * 55)