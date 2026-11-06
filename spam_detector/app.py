"""
app.py
Purpose: Streamlit web app. Run with:  streamlit run app.py
Needs the files created by train_model.py.
"""
import json
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from preprocess import clean_text

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

# ---------- Small CSS for a clean look ----------
st.markdown("""
<style>
.result-box {padding:20px; border-radius:12px; text-align:center;
             font-size:26px; font-weight:700; margin-top:10px;}
.spam {background:#fdecea; color:#c62828; border:2px solid #c62828;}
.ham  {background:#e8f5e9; color:#2e7d32; border:2px solid #2e7d32;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl"), joblib.load("tfidf_vectorizer.pkl")


# ---------- Check that the model exists ----------
if not (os.path.exists("spam_model.pkl") and os.path.exists("tfidf_vectorizer.pkl")):
    st.error("Model files not found. Please run  `python train_model.py`  first.")
    st.stop()

model, vectorizer = load_model()

# ---------- Sidebar navigation ----------
st.sidebar.title("📧 Spam Detector")
page = st.sidebar.radio("Navigation", ["🔍 Predict", "📊 Model Performance", "ℹ️ About"])

# ================= PAGE 1: PREDICT =================
if page == "🔍 Predict":
    st.title("🔍 Spam Email Detection")
    st.write("Paste an email or text message below and click **Predict**.")

    message = st.text_area("Email / Message", height=180,
                           placeholder="e.g. Congratulations! You won a free iPhone. Click here to claim...")

    if st.button("Predict", type="primary"):
        if message.strip() == "":
            st.warning("Please enter a message first.")
        else:
            cleaned = clean_text(message)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]   # [P(ham), P(spam)]
            confidence = probabilities[prediction] * 100

            if prediction == 1:
                st.markdown('<div class="result-box spam">🚨 Spam Email</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box ham">✅ Not Spam Email</div>', unsafe_allow_html=True)

            st.write("")
            st.metric("Prediction confidence", f"{confidence:.2f}%")
            st.progress(float(probabilities[prediction]))
            c1, c2 = st.columns(2)
            c1.metric("Not Spam probability", f"{probabilities[0]*100:.2f}%")
            c2.metric("Spam probability", f"{probabilities[1]*100:.2f}%")

# ================= PAGE 2: PERFORMANCE =================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    if not os.path.exists("metrics.json"):
        st.warning("metrics.json not found. Run `python train_model.py` first.")
    else:
        with open("metrics.json") as f:
            m = json.load(f)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        c2.metric("Precision", f"{m['precision']*100:.2f}%")
        c3, c4 = st.columns(2)
        c3.metric("Recall", f"{m['recall']*100:.2f}%")
        c4.metric("F1-score", f"{m['f1_score']*100:.2f}%")
        st.caption(f"Trained on {m['train_size']} messages, tested on {m['test_size']} messages.")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(np.array(m["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ================= PAGE 3: ABOUT =================
else:
    st.title("ℹ️ About")
    st.write("""
    This app classifies a message as **Spam** or **Not Spam (Ham)**.

    **Pipeline:** text cleaning → TF-IDF features → Multinomial Naive Bayes.

    **Tip:** the bundled `spam.csv` is a small sample dataset. For better real-world
    accuracy, replace it with the SMS Spam Collection dataset (Kaggle/UCI) and re-run
    `python train_model.py`.
    """)
