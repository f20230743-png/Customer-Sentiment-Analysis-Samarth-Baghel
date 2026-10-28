import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.feature_engineering import create_tfidf_features
from src.train_model import split_data
from src.train_model import train_logistic_regression
from src.evaluate import evaluate_model
from src.train_model import train_naive_bayes
from src.train_model import train_linear_svm
from src.train_model import train_random_forest
from src.save_model import save_model
from src.model_comparison import compare_all_models
# ==========================================================
# DOWNLOAD NLTK DATA
# ==========================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")

print("=" * 60)
print("Original Dataset Shape:", df.shape)

# ==========================================================
# DATA CLEANING
# ==========================================================

df.drop(columns=["Unnamed: 0"], inplace=True)

df.dropna(subset=["Review Text"], inplace=True)

df.reset_index(drop=True, inplace=True)

print("Cleaned Dataset Shape:", df.shape)

# ==========================================================
# CREATE SENTIMENT
# ==========================================================

def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Rating"].apply(get_sentiment)

# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

df["Clean Review"] = df["Review Text"].apply(clean_text)

# ==========================================================
# TOKENIZATION + LEMMATIZATION
# ==========================================================

stop_words = set(stopwords.words("english"))

stop_words.discard("not")
stop_words.discard("no")

lemmatizer = WordNetLemmatizer()

def preprocess(text):

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if word not in stop_words and word.isalpha()
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

df["Processed Review"] = df["Clean Review"].apply(preprocess)

# ==========================================================
# REVIEW LENGTH
# ==========================================================

df["Review Length"] = df["Processed Review"].apply(lambda x: len(x.split()))

# ==========================================================
# SAMPLE OUTPUT
# ==========================================================

print("\nSample Data\n")

print(
    df[
        [
            "Review Text",
            "Processed Review",
            "Rating",
            "Sentiment",
        ]
    ].head()
)

# ==========================================================
# EDA
# ==========================================================

# Rating Distribution

plt.figure(figsize=(7,5))

df["Rating"].value_counts().sort_index().plot(kind="bar")

plt.title("Rating Distribution")

plt.xlabel("Rating")

plt.ylabel("Count")

plt.show()

# ----------------------------------------------------------

# Sentiment Distribution

plt.figure(figsize=(7,5))

df["Sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Count")

plt.show()

# ----------------------------------------------------------

# Review Length Distribution

plt.figure(figsize=(8,5))

plt.hist(df["Review Length"], bins=30)

plt.title("Review Length Distribution")

plt.xlabel("Number of Words")

plt.ylabel("Frequency")

plt.show()

# ======================================================
# FEATURE ENGINEERING
# ======================================================

X, y, vectorizer = create_tfidf_features(df)

print("\n" + "="*60)
print("TF-IDF Matrix Shape")
print(X.shape)

print("\nTarget Classes")
print(y.unique())

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = split_data(X, y)

print("\n" + "="*60)
print("Training Data Shape")
print(X_train.shape)

print("\nTesting Data Shape")
print(X_test.shape)

print("\nTraining Labels")
print(y_train.value_counts())

print("\nTesting Labels")
print(y_test.value_counts())

# =====================================================
# LOGISTIC REGRESSION
# =====================================================

logistic_model = train_logistic_regression(
    X_train,
    y_train
)

evaluate_model(
    logistic_model,
    X_test,
    y_test
)
save_model(
    logistic_model,
    vectorizer
)
# =====================================================
# MULTINOMIAL NAIVE BAYES
# =====================================================

naive_bayes_model = train_naive_bayes(
    X_train,
    y_train
)

print("\n" + "="*60)
print("MULTINOMIAL NAIVE BAYES")

evaluate_model(
    naive_bayes_model,
    X_test,
    y_test
)
# =====================================================
# LINEAR SVM
# =====================================================

print("\n" + "=" * 60)
print("LINEAR SVM")
print("=" * 60)

svm_model = train_linear_svm(
    X_train,
    y_train
)

evaluate_model(
    svm_model,
    X_test,
    y_test
)
# =====================================================
# RANDOM FOREST
# =====================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_model = train_random_forest(
    X_train,
    y_train
)

evaluate_model(
    rf_model,
    X_test,
    y_test
)
models = {

    "Logistic Regression": logistic_model,

    "Naive Bayes": naive_bayes_model,

    "Linear SVM": svm_model,

    "Random Forest": rf_model

}
results = compare_all_models(
    models,
    X_test,
    y_test
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results)
results.to_csv(
    "model_results.csv",
    index=False
)

print("\nModel comparison saved successfully!")