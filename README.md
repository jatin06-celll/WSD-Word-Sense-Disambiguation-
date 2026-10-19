# 🧠 Word Sense Disambiguation (WSD)

A Machine Learning and Natural Language Processing (NLP) system designed to disambiguate polysemous words (words with multiple meanings) across different languages based on sentence context.

Includes an interactive **Streamlit Web Application** with real-time prediction, confidence scoring, and dataset analytics.

---

## 🌟 Overview

In human languages, many words have multiple interpretations depending on context (e.g., in Hindi, **"कल"** can mean *Yesterday* or *Tomorrow*; in English, **"bank"** can mean *River Bank* or *Financial Institution*). 

This project trains specialized models for each polysemous word using **TF-IDF Feature Unions (Word + Character n-grams)** and **Logistic Regression** to accurately predict the intended sense from context.

---

## 🚀 Key Features

- **Multi-Language Support**: Disambiguation across languages and polysemous target words.
- **Advanced Feature Extraction**: Uses `FeatureUnion` combining word-level n-grams and character-level n-grams (`char_wb`) to effectively capture morphological inflections, verb tenses, and subword contexts.
- **Interactive Web App**: Built with **Streamlit** featuring a modern, dark-themed UI.
- **Confidence Scores & Performance**: Real-time prediction probabilities along with train and test evaluation metrics.
- **Dataset Insights**: Interactive exploration of polysemous words, sense distributions, and sample sentences.
- **CLI Mode**: A standalone script (`main.py`) for quick command-line evaluation and classification reports.

---

## 📁 Project Structure

```text
├── app.py              # Streamlit interactive web application
├── main.py             # CLI script for model training and evaluation
├── dataset.csv         # Labeled sentences dataset (Language, Word, Meaning, Sentence)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore configuration
└── README.md           # Project documentation
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Machine Learning & NLP**: `scikit-learn`, `numpy`, `pandas`
- **Web Framework**: `Streamlit`
- **Visualization**: `matplotlib`, `seaborn`

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/jatin06-celll/WSD-Word-Sense-Disambiguation-.git
cd WSD-Word-Sense-Disambiguation-
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the Web Application
Launch the interactive Streamlit interface in your browser:
```bash
streamlit run app.py
```

### Run via Command Line Interface (CLI)
Train models and print classification reports directly in the terminal:
```bash
python main.py
```

---

## 📊 How It Works

1. **Dataset Grouping**: Sentences are grouped by `(Language, Word)`.
2. **Feature Extraction**:
   - **Word n-grams (`ngram_range=(1, 2)`)**: Captures co-occurring contextual words.
   - **Character n-grams (`char_wb`, `ngram_range=(2, 5)`)**: Captures grammatical clues like verb endings and tense markers (crucial for distinguishing tenses such as *था/थी* vs *गा/गी*).
3. **Classification**: Logistic Regression models predict the sense probability for any input sentence containing the ambiguous word.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
