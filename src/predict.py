import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# =====================================================
# DOWNLOAD NLTK
# =====================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("models/sentiment_model.pkl")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# =====================================================
# NLP OBJECTS
# =====================================================

stop_words = set(stopwords.words("english"))

stop_words.discard("not")
stop_words.discard("no")

lemmatizer = WordNetLemmatizer()


# =====================================================
# PREPROCESS TEXT
# =====================================================

def preprocess(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

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


# =====================================================
# PREDICT
# =====================================================

def predict_sentiment(review):

    processed = preprocess(review)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)

    return prediction[0]