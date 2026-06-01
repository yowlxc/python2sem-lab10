from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from pathlib import Path
from typing import List
from pydantic import BaseModel
import io

# ============================================
# ИМПОРТ БЭКЕНДА (из папки backend)
# ============================================
from backend.schemas import ClientData
from backend.model_service import load_model_from_disk, predict_dataframe, upload_model_file
from backend.csv_service import process_csv

# ============================================
# НАСТРОЙКА ПРИЛОЖЕНИЯ
# ============================================
app = FastAPI(title="Mortgage ML Service")

# Настройка фронтенда (шаблоны и статика)
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


# ============================================
# ФРОНТЕНД — ГЛАВНАЯ СТРАНИЦА
# ============================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница с интерфейсом"""
    # Получаем список признаков из схемы ClientData
    features = []
    for field_name, field_info in ClientData.model_fields.items():
        features.append({
            "name": field_name,
            "label": field_name.replace("_", " ").title(),
            "type": "number" if field_info.annotation == float or field_info.annotation == int else "text"
        })

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": features
        }
    )


# ============================================
# ЭНДПОИНТЫ БЭКЕНДА (уже есть, но переопределяем для единого приложения)
# ============================================

@app.on_event("startup")
def startup_event():
    load_model_from_disk()


@app.get("/health")
async def health():
    from backend.model_service import model
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/upload-model")
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


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    return await process_csv(file)


# ============================================
# ЗАПУСК
# ============================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)