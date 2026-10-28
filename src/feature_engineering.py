from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(df):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(df["Processed Review"])

    y = df["Sentiment"]

    return X, y, vectorizer