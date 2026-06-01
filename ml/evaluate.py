import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


def evaluate_model(model, X_test, y_test):
    """
    Оценка модели на тестовой выборке.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred
        )
    }

    return results


def evaluate_csv(model, df):
    """
    Используется для endpoint /predict-from-csv.
    """

    result_df = df.copy()

    roc_auc = None

    if "loan_status" in result_df.columns:

        y_true = result_df["loan_status"]

        X = result_df.drop(
            columns=["loan_status"]
        )

        y_prob = model.predict_proba(X)[:, 1]

        roc_auc = roc_auc_score(
            y_true,
            y_prob
        )

    else:

        X = result_df

    predictions = model.predict(X)

    result_df["predicted_loan_status"] = predictions

    return {
        "roc_auc": roc_auc,
        "data": result_df
    }