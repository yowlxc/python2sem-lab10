# Чтение CSV, предсказания, расчёт ROC-AUC, формирование ответа

import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.metrics import roc_auc_score

from backend import model_service
from backend.model_service import predict_dataframe


def _get_probabilities(input_data):

    # создаем вспомогательную функцию
    # получения вероятностей одобрения ипотеки

    if model_service.model is None:
        raise HTTPException(
            status_code=400,
            detail="Модель не загружена"
        )

    if hasattr(model_service.model, "predict_proba"): # проверяем, умеет ли модель возвращать вероятности
        return model_service.model.predict_proba(
            input_data
        )[:, 1] # нам нужен 2-ой столбец

    return None


async def process_csv(file: UploadFile):

    # создаём асинхронную функцию обработки CSV-файла

    if not file.filename.endswith(".csv"):
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

    roc_auc = None

    if "loan_status" in df.columns:
        y_true = df["loan_status"]
        X = df.drop(columns=["loan_status"])

    else:
        X = df

    predictions = predict_dataframe(X)

    # добавление новых колонок

    df["predicted_loan_status"] = predictions

    df["predicted_status_text"] = (
        df["predicted_loan_status"]
        .apply(
            lambda value:
            "одобрено"
            if int(value) == 1
            else "отказ"
        )
    )

    probabilities = _get_probabilities(X)

    if (
            "loan_status" in df.columns
            and probabilities is not None
    ):
        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )

    return {
        "roc_auc": roc_auc,
        "rows": df.to_dict(
            orient="records"
        )
    }