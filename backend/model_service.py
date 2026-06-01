# Это работа с моделью.(загрузка модели, хранение модели, предсказание

from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException, UploadFile

MODEL_PATH = Path("models/mortgage_model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)

model = None


def load_model_from_disk():
    global model

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None

    return model


async def upload_model_file(file: UploadFile): # загрузка модели
    global model

    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Файл должен быть .pkl")

    content = await file.read()

    with open(MODEL_PATH, "wb") as f:
        f.write(content)

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {error}")

    return {"status": "success", "message": "Модель успешно загружена"}


def predict_dataframe(df: pd.DataFrame): # получение предсказания
    if model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")

    return model.predict(df)