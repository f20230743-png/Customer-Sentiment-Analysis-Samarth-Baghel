import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def compare_all_models(models, X_test, y_test):

    results = []

    for model_name, model in models.items():

        predictions = model.predict(X_test)

        results.append({

            "Model": model_name,

            "Accuracy": round(
                accuracy_score(y_test, predictions), 4
            ),

            "Precision": round(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted"
                ),
                4
            ),

            "Recall": round(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted"
                ),
                4
            ),

            "F1 Score": round(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted"
                ),
                4
            )

        })

    results_df = pd.DataFrame(results)

    return results_df