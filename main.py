import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# 1. Load Dataset
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("Error: dataset.csv not found.")
    sys.exit(1)

# 2. Train Models and Evaluate
print("\n⚙️  Training and Evaluating WSD Models...")
models = {}
vectorizers = {}
accuracy_data = {}  # Store train/test accuracy for display

grouped = df.groupby(["Language", "Word"])
for (lang, word), group in grouped:
    sentences = list(group["Sentence"])
    labels = list(group["Meaning"])

    # Check if we have enough samples per class for stratified split
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    # Use stratify only if every class has at least 2 samples
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

    # Evaluation — Training Accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Evaluation — Testing Accuracy
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n--- Evaluation for '{word}' ({lang}) ---")
    print(f"📈 Training Accuracy : {train_acc * 100:.2f}%")
    print(f"📉 Testing  Accuracy : {test_acc * 100:.2f}%")
    print("Classification Report (on Test Set):")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    models[(lang, word)] = model
    vectorizers[(lang, word)] = vec
    accuracy_data[(lang, word)] = {"train": train_acc, "test": test_acc}

print("\n✅ Models ready!")
print("\n" + "="*50)
print("🧠 Word Sense Disambiguation (WSD) Terminal App")
print("="*50)

print("\nSupported Words:")
for lang, word in models.keys():
    senses = df[(df["Language"] == lang) & (df["Word"] == word)]["Meaning"].unique()
    train_pct = accuracy_data[(lang, word)]["train"] * 100
    test_pct = accuracy_data[(lang, word)]["test"] * 100
    print(f" - {word} ({lang}) -> Senses: {', '.join(senses)}  |  Train: {train_pct:.1f}%  Test: {test_pct:.1f}%")

print("\n💬 Type 'exit' to quit.")
print("ℹ️  You can enter ANY sentence — it does not need to be from the dataset!")

# 3. Interactive Loop
while True:
    print("-" * 40)
    target_word = input("Enter Ambiguous Word (e.g. कल, हार): ").strip()

    if target_word.lower() == 'exit':
        print("👋 Exiting...")
        break

    # Find language for word
    lang = None
    for l, w in models.keys():
        if w == target_word:
            lang = l
            break

    if not lang:
        print(f"⚠️  Word '{target_word}' is not in our dataset. Try one from the list above.")
        continue

    sentence = input(f"Enter a sentence containing '{target_word}': ").strip()

    if not sentence:
        print("⚠️ Sentence cannot be empty.")
        continue

    vec = vectorizers[(lang, target_word)]
    model = models[(lang, target_word)]

    X_test_interactive = vec.transform([sentence])
    prediction = model.predict(X_test_interactive)[0]
    confidence = np.max(model.predict_proba(X_test_interactive)) * 100

    print(f"\n🌍 Predicted Meaning: {prediction}")
    print(f"📊 Confidence: {confidence:.2f}%")


# streamlit run app.py







# streamlit run app.py