# Работа с моделью:
# загрузка модели, хранение модели, предсказание.

from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException, UploadFile

MODEL_PATH = Path("models/mortgage_model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)

model = None


def load_model_from_disk():
    """Загрузка модели с диска при запуске приложения."""

    global model

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None

    return model


async def upload_model_file(file: UploadFile):
    """Загрузка модели через API."""

    global model

    if not file.filename or not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть .pkl"
        )

    content = await file.read()

    with open(MODEL_PATH, "wb") as f:
        f.write(content)

    try:
        model = joblib.load(MODEL_PATH)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка загрузки модели: {error}"
        )

    return {
        "status": "success",
        "message": "Модель успешно загружена"
    }


def predict_dataframe(df: pd.DataFrame):
    """Предсказание по DataFrame."""

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Модель не загружена"
        )

    try:
        return model.predict(df)

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка предсказания: {error}"
        )


def predict_probabilities(df: pd.DataFrame):
    """Получение вероятностей, если модель поддерживает predict_proba."""

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Модель не загружена"
        )

    if not hasattr(model, "predict_proba"):
        return None

    try:
        return model.predict_proba(df)[:, 1]

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка расчёта вероятностей: {error}"
        )