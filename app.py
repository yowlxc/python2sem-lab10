from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates  # можно использовать Jinja2, но оставим Mako
from pathlib import Path
import pandas as pd
import joblib
import json
import io
from typing import List, Optional
from templating import render_template  # ваш модуль для Mako

app = FastAPI(title="Mortgage ML Service")

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Путь к модели
MODEL_PATH = Path("models/mortgage_model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)
model = None

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================
def load_model_from_disk():
    global model
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Модель загружена из диска")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            model = None
    return model

@app.on_event("startup")
def startup_event():
    load_model_from_disk()

# ============================================
# ПРИЗНАКИ ДЛЯ ФОРМЫ
# ============================================
FEATURES = [
    {"name": "person_age", "label": "Возраст", "type": "number", "placeholder": "30"},
    {"name": "person_gender", "label": "Пол", "type": "select", "options": ["male", "female"]},
    {"name": "person_education", "label": "Образование", "type": "select",
     "options": ["Bachelor", "Master", "PhD", "High School"]},
    {"name": "person_income", "label": "Доход", "type": "number", "placeholder": "50000"},
    {"name": "person_emp_exp", "label": "Опыт работы (лет)", "type": "number", "placeholder": "5"},
    {"name": "person_home_ownership", "label": "Владение жильём", "type": "select",
     "options": ["RENT", "OWN", "MORTGAGE", "OTHER"]},
    {"name": "loan_amnt", "label": "Сумма кредита", "type": "number", "placeholder": "200000"},
    {"name": "loan_intent", "label": "Цель кредита", "type": "select",
     "options": ["EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]},
    {"name": "loan_int_rate", "label": "Процентная ставка", "type": "number", "step": "0.1", "placeholder": "10.5"},
    {"name": "loan_percent_income", "label": "% дохода на кредит", "type": "number", "step": "0.01", "placeholder": "0.3"},
    {"name": "cb_person_cred_hist_length", "label": "Кредитная история (лет)", "type": "number", "placeholder": "5"},
    {"name": "credit_score", "label": "Кредитный рейтинг", "type": "number", "placeholder": "700"},
    {"name": "previous_loan_defaults_on_file", "label": "Были просрочки", "type": "select", "options": ["Yes", "No"]}
]

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================
def predict_dataframe(df: pd.DataFrame):
    if model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    return model.predict(df)

def render_main_page(request: Request,
                     prediction_result: Optional[dict] = None,
                     model_status_msg: str = None,
                     csv_result_html: str = None,
                     form_data: dict = None):
    """Единая функция рендеринга главной страницы с данными"""
    if model_status_msg is None:
        model_status_msg = "✅ Модель загружена" if model is not None else "❌ Модель не загружена"
    if form_data is None:
        form_data = {}
    return render_template(
        "index.mako",
        features=FEATURES,
        model_status=model_status_msg,
        prediction_result=prediction_result,
        csv_result=csv_result_html,
        request=request,
        form_data=form_data
    )

# ============================================
# UI-ЭНДПОИНТЫ (для отображения HTML)
# ============================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_main_page(request)

@app.post("/upload-model-ui", response_class=HTMLResponse)
async def upload_model_ui(request: Request, file: UploadFile = File(...)):
    global model
    if not file.filename.endswith(".pkl"):
        return render_main_page(request, model_status_msg="❌ Ошибка: файл должен быть .pkl")
    content = await file.read()
    with open(MODEL_PATH, "wb") as f:
        f.write(content)
    try:
        model = joblib.load(MODEL_PATH)
        msg = f"✅ Модель {file.filename} успешно загружена"
    except Exception as e:
        msg = f"❌ Ошибка загрузки модели: {e}"
    return render_main_page(request, model_status_msg=msg)

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request):
    if model is None:
        return render_main_page(request, model_status_msg="❌ Модель не загружена. Сначала загрузите модель.")
    form_data = await request.form()
    # Преобразуем данные в DataFrame
    data_dict = {}
    for f in FEATURES:
        val = form_data.get(f["name"], "")
        if f["type"] == "number" and val != "":
            val = float(val)
        data_dict[f["name"]] = val
    df = pd.DataFrame([data_dict])
    try:
        prediction = int(model.predict(df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[0][1])
        result = {
            "loan_status": prediction,
            "probability": probability
        }
        return render_main_page(request, prediction_result=result, form_data=dict(form_data))
    except Exception as e:
        return render_main_page(request, model_status_msg=f"❌ Ошибка предсказания: {e}")

@app.post("/predict-csv-ui", response_class=HTMLResponse)
async def predict_csv_ui(request: Request, file: UploadFile = File(...)):
    if model is None:
        return render_main_page(request, model_status_msg="❌ Модель не загружена")
    if not file.filename.endswith(".csv"):
        return render_main_page(request, model_status_msg="❌ Файл должен быть CSV")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        roc_auc = None
        if "loan_status" in df.columns:
            y_true = df["loan_status"]
            X = df.drop(columns=["loan_status"])
        else:
            X = df
        predictions = predict_dataframe(X)
        df["predicted_loan_status"] = predictions
        df["predicted_status_text"] = df["predicted_loan_status"].apply(lambda x: "одобрено" if int(x) == 1 else "отказ")
        if "loan_status" in df.columns and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(y_true, probabilities)
        # Формируем HTML-таблицу
        html = f"<p><strong>📈 ROC-AUC:</strong> {roc_auc:.4f}</p>" if roc_auc is not None else "<p>⚠️ ROC-AUC не рассчитан</p>"
        html += f"<p><strong>🧾 Количество строк:</strong> {len(df)}</p>"
        if len(df) > 0:
            html += '<div style="overflow-x: auto;"><table border="1"><thead><tr>'
            for col in df.columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"
            for _, row in df.head(20).iterrows():
                html += "<tr>"
                for val in row:
                    html += f"<td>{val}</td>"
                html += "</tr>"
            if len(df) > 20:
                html += f"<tr><td colspan='100'>... и ещё {len(df)-20} строк</td></tr>"
            html += "</tbody></table></div>"
        return render_main_page(request, csv_result_html=html)
    except Exception as e:
        return render_main_page(request, model_status_msg=f"❌ Ошибка обработки CSV: {e}")

# ============================================
# API-ЭНДПОИНТЫ (по заданию, возвращают JSON)
# ============================================
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/upload-model")
async def upload_model_api(file: UploadFile = File(...)):
    global model
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Файл должен быть .pkl")
    content = await file.read()
    with open(MODEL_PATH, "wb") as f:
        f.write(content)
    try:
        model = joblib.load(MODEL_PATH)
        return {"status": "success", "message": f"Модель {file.filename} загружена"}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {error}")

@app.post("/predict")
async def predict_api(data: List[dict]):
    if model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    df = pd.DataFrame(data)
    predictions = predict_dataframe(df)
    result = []
    for i, pred in enumerate(predictions):
        result.append({
            **data[i],
            "loan_status": int(pred),
            "status_text": "одобрено" if int(pred) == 1 else "отказ"
        })
    return result

@app.post("/predict-from-csv")
async def predict_csv_api(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть CSV")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        roc_auc = None
        if "loan_status" in df.columns:
            y_true = df["loan_status"]
            X = df.drop(columns=["loan_status"])
        else:
            X = df
        predictions = predict_dataframe(X)
        df["predicted_loan_status"] = predictions
        df["predicted_status_text"] = df["predicted_loan_status"].apply(lambda x: "одобрено" if int(x) == 1 else "отказ")
        if "loan_status" in df.columns and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(y_true, probabilities)
        return {
            "roc_auc": roc_auc,
            "rows": df.to_dict(orient="records")
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки CSV: {error}")

# ============================================
# ЗАПУСК
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)