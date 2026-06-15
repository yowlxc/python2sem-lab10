from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from backend.csv_service import process_csv
from backend.model_service import (
    load_model_from_disk,
    predict_dataframe,
    predict_probabilities,
    upload_model_file
)
from backend.preprocess_service import prepare_client_row
from backend.schemas import ClientData


templates = Jinja2Templates(directory="templates")


FEATURES = [
    {
        "name": "person_income",
        "label": "Годовой доход",
        "type": "number",
        "placeholder": "50000",
        "step": "0.01"
    },
    {
        "name": "person_emp_exp",
        "label": "Стаж работы (лет)",
        "type": "number",
        "placeholder": "5",
        "step": "1"
    },
    {
        "name": "loan_amnt",
        "label": "Сумма кредита",
        "type": "number",
        "placeholder": "10000",
        "step": "0.01"
    },
    {
        "name": "loan_int_rate",
        "label": "Процентная ставка",
        "type": "number",
        "placeholder": "12.5",
        "step": "0.01"
    },
    {
        "name": "cb_person_cred_hist_length",
        "label": "Длина кредитной истории (лет)",
        "type": "number",
        "placeholder": "4",
        "step": "0.1"
    },
    {
        "name": "credit_score",
        "label": "Кредитный рейтинг",
        "type": "number",
        "placeholder": "650",
        "step": "1"
    },
    {
        "name": "previous_loan_defaults_on_file",
        "label": "Были ли просрочки",
        "type": "select",
        "options": ["Нет", "Да"]
    },
    {
        "name": "person_gender",
        "label": "Пол",
        "type": "select",
        "options": ["Женский", "Мужской"]
    },
    {
        "name": "person_education",
        "label": "Образование",
        "type": "select",
        "options": ["Среднее", "Бакалавр", "Магистр", "Доктор наук"]
    },
{
    "name": "person_home_ownership",
    "label": "Тип жилья",
    "type": "select",
    "options": [
        "Собственное жильё",
        "Аренда",
        "Другое",
        "Ипотека"
    ]
},
{
    "name": "loan_intent",
    "label": "Цель кредита",
    "type": "select",
    "options": [
        "Образование",
        "Ремонт",
        "Медицина",
        "Личные нужды",
        "Бизнес"
    ]
}
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_from_disk()
    yield


app = FastAPI(
    title="Mortgage Approval Service",
    lifespan=lifespan
)


app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static"
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "features": FEATURES,
            "form_data": {},
            "model_status": "Модель пока не загружалась через интерфейс",
            "prediction_result": None,
            "csv_result": None
        }
    )


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    return await upload_model_file(file)


@app.post("/predict")
def predict(data: List[ClientData]):
    rows = []

    for item in data:
        row = item.model_dump()
        prepared_row = prepare_client_row(row)
        rows.append(prepared_row)

    df = pd.DataFrame(rows)

    predictions = predict_dataframe(df)
    probabilities = predict_probabilities(df)

    result = []

    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        probability = None

        if probabilities is not None:
            probability = float(probabilities[index])

        result.append({
            **row,
            "loan_status": int(prediction),
            "status_text": "одобрено" if int(prediction) == 1 else "отказ",
            "probability": probability
        })

    return result


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    return await process_csv(file)




@app.post("/upload-model-ui", response_class=HTMLResponse)
async def upload_model_ui(request: Request, file: UploadFile = File(...)):
    try:
        result = await upload_model_file(file)
        model_status = result["message"]

    except Exception as error:
        model_status = f"Ошибка: {error}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "features": FEATURES,
            "form_data": {},
            "model_status": model_status,
            "prediction_result": None,
            "csv_result": None
        }
    )


@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    person_income: float = Form(...),
    person_emp_exp: int = Form(...),
    loan_amnt: float = Form(...),
    loan_int_rate: float = Form(...),
    cb_person_cred_hist_length: float = Form(...),
    credit_score: int = Form(...),
    previous_loan_defaults_on_file: str = Form(...),
    person_gender: str = Form(...),
    person_education: str = Form(...),
    person_home_ownership: str = Form(...),
    loan_intent: str = Form(...)
):
    form_data = {
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent

    }

    try:
        prepared_row = prepare_client_row(form_data.copy())
        df = pd.DataFrame([prepared_row])

        prediction = predict_dataframe(df)[0]
        probabilities = predict_probabilities(df)

        probability = None

        if probabilities is not None:
            probability = float(probabilities[0])

        prediction_result = {
            "loan_status": int(prediction),
            "status_text": "одобрено" if int(prediction) == 1 else "отказ",
            "probability": probability
        }

    except Exception as error:
        prediction_result = {
            "error": str(error)
        }

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "features": FEATURES,
            "form_data": form_data,
            "model_status": "",
            "prediction_result": prediction_result,
            "csv_result": None
        }
    )


@app.post("/predict-csv-ui", response_class=HTMLResponse)
async def predict_csv_ui(request: Request, file: UploadFile = File(...)):
    try:
        result = await process_csv(file)

        csv_result = ""

        if result["roc_auc"] is not None:
            csv_result += f"<p><b>ROC-AUC:</b> {result['roc_auc']:.4f}</p>"

        rows = result["rows"]

        if rows:
            columns = rows[0].keys()

            csv_result += "<table><tr>"

            for column in columns:
                csv_result += f"<th>{column}</th>"

            csv_result += "</tr>"

            for row in rows:
                csv_result += "<tr>"

                for column in columns:
                    csv_result += f"<td>{row[column]}</td>"

                csv_result += "</tr>"

            csv_result += "</table>"

        csv_result += f"<p>Показано строк: {len(rows)}</p>"

    except Exception as error:
        csv_result = f"Ошибка: {error}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "features": FEATURES,
            "form_data": {},
            "model_status": "",
            "prediction_result": None,
            "csv_result": csv_result
        }
    )