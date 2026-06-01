#чтение CSV, предсказания, расчёт ROC-AUC, формирование ответа

import pandas as pd
from fastapi import HTTPException, UploadFile
from sklearn.metrics import roc_auc_score

from backend.model_service import model, predict_dataframe


def _get_probabilities(input_data): # cоздаем вспомогательную функцию для получения вероятностей одобрения ипотеки
    if hasattr(model, "predict_proba"): # проверяем, умеет ли модель возвращать вероятности.
        return model.predict_proba(input_data)[:, 1] # получаем вероятность класса 1.

    return None


async def process_csv(file: UploadFile): # cоздаём асинхронную функцию обработки CSV-файла.
    if not file.filename.endswith(".csv"): # валидация формата
        raise HTTPException(status_code=400, detail="Файл должен быть CSV")

    try: # чтение csv
        df = pd.read_csv(file.file)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {error}")

    roc_auc = None

    if "loan_status" in df.columns:
        y_true = df["loan_status"]
        X = df.drop(columns=["loan_status"])
    else:
        X = df

    predictions = predict_dataframe(X)

    df["predicted_loan_status"] = predictions
    df["predicted_status_text"] = df["predicted_loan_status"].apply(
        lambda value: "одобрено" if int(value) == 1 else "отказ"
    )

    probabilities = _get_probabilities(X)

    if "loan_status" in df.columns and probabilities is not None:
        roc_auc = roc_auc_score(y_true, probabilities)

    return {
        "roc_auc": roc_auc,
        "rows": df.to_dict(orient="records")
    }