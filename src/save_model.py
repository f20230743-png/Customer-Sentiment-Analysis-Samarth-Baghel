import joblib
import os

def save_model(model, vectorizer):

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/sentiment_model.pkl"
    )

    joblib.dump(
        vectorizer,
        "models/tfidf_vectorizer.pkl"
    )

    print("\n" + "="*60)
    print("Model Saved Successfully!")

    print("Location : models/sentiment_model.pkl")

    print("Vectorizer Saved Successfully!")

    print("Location : models/tfidf_vectorizer.pkl")