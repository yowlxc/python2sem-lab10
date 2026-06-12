# Это работа с моделью.
# Загрузка модели, хранение модели, предсказание.

from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException, UploadFile

MODEL_PATH = Path("models/mortgage_model.pkl") # создается путь, где будет храниться модель
MODEL_PATH.parent.mkdir(exist_ok=True)

model = None

# Загрузка модели из диска
def load_model_from_disk():
    global model

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH) # загружаем модель из файла
        except Exception:
            model = None

    return model


# загрузка модели через API
async def upload_model_file(file: UploadFile):
    global model

    if not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть .pkl"
        )

    content = await file.read() # чтение файла в FastAPI выполняется асинхронно

    with open(MODEL_PATH, "wb") as f:
        f.write(content)

    try:
        model = joblib.load(MODEL_PATH) # загружаем модель в память

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка загрузки модели: {error}"
        )

    return {
        "status": "success",
        "message": "Модель успешно загружена"
    }

# Функция предсказания:
def predict_dataframe(df: pd.DataFrame):

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Модель не загружена"
        )

    return model.predict(df)