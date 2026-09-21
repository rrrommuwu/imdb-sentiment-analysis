# 🎬 IMDB Sentiment Analysis — NLP Project

> Predicting movie review sentiment (Positive/Negative) using NLP and Machine Learning on the IMDB 50K dataset.

---

## 📊 Model Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 88.5% | 88.4% |
| Multinomial Naive Bayes | 85.2% | 85.1% |
| BernoulliNB | 84.1% | 84.0% |
| **LinearSVC (Final)** | **93.9%** | **93.8%** |

**Winner: LinearSVC** — Tuned with GridSearchCV. Trained on 40K reviews, tested on 10K.

---

## 🔍 Key EDA Insights

1. **Dataset is perfectly balanced** — exactly 25,000 positive and 25,000 negative reviews, so no class imbalance issues.
2. **Positive reviews are longer** — positive reviews average ~1,350 characters vs ~1,200 for negative reviews, suggesting more detailed praise.
3. **Most common positive words:** "great", "best", "love", "well", "life" — emotional and superlative language dominates.

---

## 🏗️ Project Structure

```
imdb-sentiment-analysis/
│
├── data/
│   └── raw/
│       └── IMDB_Dataset.csv          # 50K IMDB reviews from Kaggle
│
├── models/
│   └── sentiment_model.pkl           # Trained LinearSVC model (saved by joblib)
│
├── images/
│   ├── sentiment_distribution.png    # Bar chart of positive vs negative counts
│   ├── review_length_histogram.png   # Distribution of review lengths
│   └── wordcloud_positive.png        # Word cloud for positive reviews
│
├── notebooks/ (or scripts)
│   ├── 01_data_exploration.ipynb     # Phase A: Load & explore IMDB data
│   ├── 02_eda_visualization.py       # Phase B: 8+ charts and EDA
│   ├── 03_nlp_preprocessing.py       # Phase C: Text cleaning pipeline
│   ├── 04_feature_engineering.py     # Phase C Part 2: TF-IDF features
│   └── 05_06_07_model_training.py    # Phase D: Train, compare, tune models
│
├── app.py                            # Phase E: Streamlit web app
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## ⚙️ Tech Stack

| Category | Tools Used |
|----------|-----------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| NLP | NLTK (stopwords, lemmatizer), Regex |
| ML | scikit-learn (LinearSVC, TF-IDF, GridSearchCV) |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Deployment | Streamlit, Streamlit Cloud |
| Model Saving | joblib |
| Version Control | Git, GitHub |

---

## 🏥 Healthcare Application

This exact pipeline can be applied to healthcare:

- **Patient feedback analysis** — Automatically classify patient reviews of hospitals as positive/negative
- **Medical survey responses** — Extract satisfaction signals from open-text feedback forms
- **Clinical note sentiment** — Flag concerning language in doctor's notes for follow-up

The `clean_text()` function and TF-IDF + LinearSVC pipeline requires minimal changes to work on any text classification task.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/imdb-sentiment-analysis.git
cd nlp-sentiment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the IMDB dataset from Kaggle
# Place IMDB_Dataset.csv in data/raw/

# 4. Run all phases in order
python 02_eda_visualization.py        # Phase B: EDA
python 03_nlp_preprocessing.py        # Phase C: NLP
python 04_feature_engineering.py      # Phase C Part 2
python 05_06_07_model_training.py     # Phase D: Train model

# 5. Launch the Streamlit app
streamlit run app.py
```

---

## 📦 requirements.txt

```
pandas
numpy
scikit-learn
nltk
streamlit
matplotlib
seaborn
wordcloud
joblib
scipy
