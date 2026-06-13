# Это главный файл FastAPI.
# Здесь описаны все маршруты API.

from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, File, UploadFile

from backend.csv_service import process_csv
#Импортируются функции для работы с моделью
from backend.model_service import (
    load_model_from_disk, # загружает модель с диска при запуске приложения
    predict_dataframe, # делает предсказание по таблице
    upload_model_file # загружает .pkl модель через API
)
from backend.schemas import ClientData


@asynccontextmanager
async def lifespan(app: FastAPI):
    # при запуске приложения пробуем загрузить модель с диска
    load_model_from_disk()

    yield # yield делит функцию на две части

    # здесь можно освобождать ресурсы при завершении приложения

# Cоздание приложения
app = FastAPI(
    title="Mortgage Approval Service",
    lifespan=lifespan # подключает функцию запуска, где загружается модель
)  # создание приложения

# Создается GET-запрос по адресу(/)
@app.get("/")
def root():
    # тестовый endpoint
    return {
        "message": "Mortgage ML Service is running" # сообщение, что сервер работает
    }

# Загрузка ML-модели
@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    # загрузка модели
    return await upload_model_file(file)

# Предсказания по данным клиента
@app.post("/predict")
def predict(data: List[ClientData]):

    rows = []
   # Цикл проходит по каждому клиенту:
    for item in data:
        row = item.model_dump()

        # loan_percent_income не спрашивается напрямую
        # рассчитывается автоматически
        row["loan_percent_income"] = (
            row["loan_amnt"] / row["person_income"]
        )

        # образование - one-hot признаки
        education = row.pop("person_education")

        row["person_education_Bachelor"] = int(
            education == "Bachelor"
        )

        row["person_education_Doctorate"] = int(
            education == "Doctorate"
        )

        row["person_education_High School"] = int(
            education == "High School"
        )

        row["person_education_Master"] = int(
            education == "Master"
        )

        # пол
        row["person_gender_male"] = int(
            row["person_gender_male"]
        )

        # просрочки
        row["previous_loan_defaults_on_file"] = (
            1
            if row["previous_loan_defaults_on_file"] == "Yes"
            else 0
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    predictions = predict_dataframe(df)

    result = []

    for row, prediction in zip(rows, predictions):
        result.append({
            **row,
            "loan_status": int(prediction),
            "status_text":
                "одобрено"
                if int(prediction) == 1
                else "отказ"
        })

    return result


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    # предсказание по csv с большим количеством клиентов
    return await process_csv(file)