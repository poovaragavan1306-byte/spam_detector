"""
train_model.py
Purpose: load data -> clean -> split -> TF-IDF -> train Naive Bayes -> evaluate -> save files.
Run once with:  python train_model.py
"""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw charts to files (no pop-up window)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from preprocess import clean_text

# ---------- 1. Load the dataset ----------
# Works with our spam.csv (label,message) and the popular Kaggle/UCI
# SMS Spam file (columns v1,v2 in latin-1 encoding).
try:
    df = pd.read_csv("spam.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("spam.csv", encoding="latin-1")

if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
df = df[["label", "message"]].dropna().drop_duplicates()

print("Dataset size:", df.shape)
print(df["label"].value_counts(), "\n")

# ---------- 2. Preprocess ----------
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})   # ham=0, spam=1
df["clean"] = df["message"].apply(clean_text)

# ---------- 3. Train/test split (reproducible, 80/20) ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label_num"],
    test_size=0.2, random_state=42, stratify=df["label_num"])

# ---------- 4. TF-IDF (fit ONLY on training data -> no data leakage) ----------
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)   # learn vocabulary + transform
X_test_tfidf = vectorizer.transform(X_test)         # only transform

# ---------- 5. Train Multinomial Naive Bayes ----------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------- 6. Evaluate ----------
y_pred = model.predict(X_test_tfidf)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
}
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
fig.savefig("confusion_matrix.png", dpi=120)

# ---------- 7. Save everything the app needs ----------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
with open("metrics.json", "w") as f:
    json.dump({**metrics, "confusion_matrix": cm.tolist()}, f)

print("Saved: spam_model.pkl, tfidf_vectorizer.pkl, metrics.json, confusion_matrix.png")
