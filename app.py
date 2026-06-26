import streamlit as st
import joblib
import re

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #FBF3E7;
}
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp span {
    color: #4A2F1C !important;
}
[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0);
}
.postcard {
    background-color: #FFFDF8;
    border: 1px solid #C68B59;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(120, 80, 40, 0.08);
}
.stTextArea textarea {
    background-color: #FFFDF8 !important;
    border: 1px solid #C68B59 !important;
    color: #4A2F1C !important;
    border-radius: 8px !important;
}
.stButton > button {
    background-color: #A85C32;
    color: #FFFDF8 !important;
    border-radius: 8px;
    font-weight: 600;
    border: none;
    padding: 0.5rem 1.2rem;
}
.stButton > button:hover {
    background-color: #8A4924;
    color: #FFFDF8 !important;
}
.result-card {
    border-radius: 10px;
    padding: 1.3rem 1.6rem;
    text-align: center;
    margin-top: 0.8rem;
}
.result-spam {
    background-color: #F6E4D7;
    border: 1.5px solid #A85C32;
}
.result-ham {
    background-color: #E9EFE0;
    border: 1.5px solid #6E8B5A;
}
.result-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.result-spam .result-label { color: #8A4924 !important; }
.result-ham .result-label { color: #4F6B3D !important; }
.result-value {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.result-spam .result-value { color: #6E3414 !important; }
.result-ham .result-value { color: #3A5229 !important; }
.result-confidence {
    font-size: 0.85rem;
    color: #6B5A47 !important;
}
</style>
""", unsafe_allow_html=True)


# Load the trained model and vectorizer

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer

    with st.spinner("Training model for the first time, this takes a few seconds..."):
        data = pd.read_csv('data/spam.csv', encoding='latin-1')
        data = data.rename(columns={'Category': 'label', 'Message': 'message'})
        data = data.drop_duplicates()
        data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})
        data['clean_message'] = data['message'].apply(clean_text)

        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        X_vec = vectorizer.fit_transform(data['clean_message'])

        model = MultinomialNB()
        model.fit(X_vec, data['label_num'])

        joblib.dump(model, 'model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

    return model, vectorizer

model, vectorizer = load_model()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# App layout

st.title("📩 Spam Message Classifier")
st.write(
    "Paste any text message below and find out if it's spam or not, "
    "using a Naive Bayes model trained on thousands of real SMS messages."
)

message = st.text_area(
    "Message",
    placeholder="e.g. Congratulations! You've won a free prize, call now to claim...",
    height=120,
    label_visibility="collapsed"
)
check_clicked = st.button("Check message", type="primary")

if check_clicked:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = proba[prediction] * 100

        if prediction == 1:
            st.markdown(
                f'''
                <div class="result-card result-spam">
                    <div class="result-label">Result</div>
                    <div class="result-value">🚩 Spam</div>
                    <div class="result-confidence">{confidence:.1f}% confident</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'''
                <div class="result-card result-ham">
                    <div class="result-label">Result</div>
                    <div class="result-value">✅ Not spam</div>
                    <div class="result-confidence">{confidence:.1f}% confident</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

st.divider()
st.caption("Model: Naive Bayes (TF-IDF) · Trained on the SMS Spam Collection dataset (5,157 messages)")
