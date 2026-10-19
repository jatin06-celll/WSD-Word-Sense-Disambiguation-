import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Page Config for beautiful UI
st.set_page_config(
    page_title="Word Sense Disambiguation",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .title-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    h1 {
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    .predicted-meaning {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00f2fe;
        margin: 1rem 0;
    }
    .accuracy-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        text-align: center;
        margin-top: 0.5rem;
    }
    .acc-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-bottom: 0.3rem;
    }
    .acc-value-train {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4facfe;
    }
    .acc-value-test {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00f2fe;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_train():
    try:
        df = pd.read_csv("dataset.csv")
    except Exception:
        st.error("Error loading dataset.csv")
        return None, None, None, None, None

    models = {}
    vectorizers = {}
    word_senses = {}
    accuracy_data = {}

    grouped = df.groupby(["Language", "Word"])
    for (lang, word), group in grouped:
        sentences = list(group["Sentence"])
        labels = list(group["Meaning"])

        # Check if we have enough samples per class for stratified split
        label_counts = Counter(labels)
        min_count = min(label_counts.values())

        if min_count >= 2:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                sentences, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                sentences, labels, test_size=0.2, random_state=42
            )

        # FeatureUnion: combine word n-grams + character n-grams
        # Character n-grams capture subword patterns (verb endings, suffixes, etc.)
        # This allows the model to generalize to NEW sentences not in the dataset
        vec = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ])

        X_train = vec.fit_transform(X_train_text)
        X_test = vec.transform(X_test_text)

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)

        # Compute accuracies
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        models[(lang, word)] = model
        vectorizers[(lang, word)] = vec
        word_senses[(lang, word)] = list(group["Meaning"].unique())
        accuracy_data[(lang, word)] = {"train": train_acc, "test": test_acc}

    return df, models, vectorizers, word_senses, accuracy_data

# Header
st.markdown("""
<div class="title-box">
    <h1>🧠 AI Word Sense Disambiguation</h1>
    <p style='color: #ccc; font-size: 1.1rem;'>Discover the true meaning of ambiguous words in Indian languages.</p>
    <p style='color: #888; font-size: 0.9rem;'>Works on ANY sentence — not just examples from the dataset!</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Training ML Models..."):
    df, models, vectorizers, word_senses, accuracy_data = load_data_and_train()

if df is not None:
    # UI Layout
    col1, col2 = st.columns(2)

    languages = sorted(list(df['Language'].unique()))
    with col1:
        selected_lang = st.selectbox("🌍 Select Language", languages)

    words_in_lang = sorted(list(df[df['Language'] == selected_lang]['Word'].unique()))
    with col2:
        selected_word = st.selectbox("🔍 Select Ambiguous Word", words_in_lang)

    senses = word_senses.get((selected_lang, selected_word), [])

    st.markdown(f"**Possible Meanings:** `{'` | `'.join(senses)}`")

    # Display Training & Testing Accuracy
    acc = accuracy_data.get((selected_lang, selected_word), {})
    train_pct = acc.get("train", 0) * 100
    test_pct = acc.get("test", 0) * 100

    acc_col1, acc_col2 = st.columns(2)
    with acc_col1:
        st.markdown(f"""
        <div class="accuracy-card">
            <div class="acc-label">📈 Training Accuracy</div>
            <div class="acc-value-train">{train_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with acc_col2:
        st.markdown(f"""
        <div class="accuracy-card">
            <div class="acc-label">📉 Testing Accuracy</div>
            <div class="acc-value-test">{test_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✍️ Enter your sentence:")
    user_sentence = st.text_input("", placeholder=f"Type a sentence containing the word '{selected_word}'...")

    if st.button("🔮 Analyze Meaning", use_container_width=True):
        if user_sentence.strip() == "":
            st.warning("Please enter a sentence first.")
        else:
            # Predict — works on ANY sentence, even those not in the dataset
            vec = vectorizers[(selected_lang, selected_word)]
            model = models[(selected_lang, selected_word)]

            X_test = vec.transform([user_sentence])
            prediction = model.predict(X_test)[0]
            confidence_array = model.predict_proba(X_test)[0]
            confidence = np.max(confidence_array) * 100

            # Display Result
            st.markdown(f"""
            <div class="prediction-card">
                <p style="color: #ccc; font-size: 1.2rem;">The context implies:</p>
                <div class="predicted-meaning">{prediction}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(int(confidence) / 100)
            st.markdown(f"<p style='text-align: center; color: #aaa;'>Confidence Score: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
