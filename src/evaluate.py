from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import pandas as pd


# ======================================================
# Evaluate a Single Model
# ======================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    print("\n" + "=" * 60)
    print("Accuracy")
    print(accuracy_score(y_test, predictions))

    print("\n" + "=" * 60)
    print("Classification Report")
    print(classification_report(y_test, predictions))

    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))


# ======================================================
# Compare Multiple Models
# ======================================================

def compare_models(models, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, predictions), 4),
            "Precision": round(
                precision_score(y_test, predictions, average="weighted"),
                4
            ),
            "Recall": round(
                recall_score(y_test, predictions, average="weighted"),
                4
            ),
            "F1 Score": round(
                f1_score(y_test, predictions, average="weighted"),
                4
            )
        })

    return pd.DataFrame(results)