# Это главный файл FastAPI.
# Здесь описаны все маршруты API.

from typing import List

import pandas as pd
from fastapi import FastAPI, File, UploadFile

from backend.csv_service import process_csv
from backend.model_service import load_model_from_disk, predict_dataframe, upload_model_file
from backend.schemas import ClientData

app = FastAPI(title="Mortgage Approval Service") # создания приложения


@app.on_event("startup")
def startup_event():
    load_model_from_disk()


@app.get("/")
def root():
    return {"message": "Mortgage ML Service is running"}


@app.post("/upload-model") # загрузка модели
async def upload_model(file: UploadFile = File(...)):
    return await upload_model_file(file)


@app.post("/predict")
def predict(data: List[ClientData]):
    df = pd.DataFrame([item.model_dump() for item in data])
    predictions = predict_dataframe(df)

    result = []

    for item, prediction in zip(data, predictions):
        result.append({
            **item.model_dump(),
            "loan_status": int(prediction),
            "status_text": "одобрено" if int(prediction) == 1 else "отказ"
        })

    return result


@app.post("/predict-from-csv") # предсказание по csv
async def predict_from_csv(file: UploadFile = File(...)):
    return await process_csv(file)