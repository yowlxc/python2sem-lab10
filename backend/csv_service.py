# Чтение CSV, предсказания, расчёт ROC-AUC, формирование ответа.

import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.metrics import roc_auc_score

from backend.model_service import predict_dataframe, predict_probabilities
from backend.preprocess_service import prepare_dataframe


def _prepare_target(target_column):
    """Приведение loan_status к 0/1."""

    if target_column.dtype == "object":
        return target_column.map({
            "одобрено": 1,
            "отказ": 0,
            "approved": 1,
            "rejected": 0,
            "Approved": 1,
            "Rejected": 0,
            "Y": 1,
            "N": 0,
            "Yes": 1,
            "No": 0,
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0
        })

    return target_column


async def process_csv(file: UploadFile):
    """Обработка CSV-файла."""

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть CSV"
        )

    try:
        df = pd.read_csv(file.file)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка чтения CSV: {error}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV-файл пустой"
        )

    roc_auc = None

    if "loan_status" in df.columns:
        y_true = _prepare_target(df["loan_status"])
        X_raw = df.drop(columns=["loan_status"])
    else:
        y_true = None
        X_raw = df

    try:
        X = prepare_dataframe(X_raw)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка предобработки CSV: {error}"
        )

    predictions = predict_dataframe(X)
    probabilities = predict_probabilities(X)

    df["predicted_loan_status"] = predictions

    df["predicted_status_text"] = df["predicted_loan_status"].apply(
        lambda value: "одобрено" if int(value) == 1 else "отказ"
    )

    if y_true is not None and probabilities is not None:
        try:
            roc_auc = roc_auc_score(y_true, probabilities)
        except ValueError:
            roc_auc = None

    return {
        "roc_auc": roc_auc,
        "rows": df.to_dict(orient="records")
    }