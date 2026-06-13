from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import joblib
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
    {"name": "person_income", "label": "Годовой доход (после налогов)", "type": "number", "placeholder": "50000"},
    {"name": "person_emp_exp", "label": "Официальный стаж (лет)", "type": "number", "placeholder": "5"},
    {"name": "loan_amnt", "label": "Сумма кредита", "type": "number", "placeholder": "200000"},
    {"name": "loan_int_rate", "label": "Процентная ставка", "type": "number", "step": "0.1", "placeholder": "10.5"},
    {"name": "cb_person_cred_hist_length", "label": "Кредитная история (лет)", "type": "number", "placeholder": "5"},
    {"name": "credit_score", "label": "Кредитный рейтинг", "type": "number", "placeholder": "700"},
    {"name": "previous_loan_defaults_on_file", "label": "Были ли просрочки по кредитам?", "type": "select", "options": ["Yes", "No"]},
    {"name": "person_gender_male", "label": "Пол мужской", "type": "select", "options": ["1", "0"]},
    {"name": "person_education", "label": "Образование", "type": "select", "options": ["Bachelor", "Doctorate", "High School", "Master"]}
]

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================
def predict_dataframe(df: pd.DataFrame):
    if model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    return model.predict(df)

def render_main_page(request: Request,
                     prediction_result=None,
                     model_status_msg=None,
                     csv_result_html=None,
                     form_data=None):
    if model_status_msg is None:
        model_status_msg = "✅ Модель загружена" if model is not None else "❌ Модель не загружена"
    if form_data is None:
        form_data = {}
    return render_template(
        "index.html",
        features=FEATURES,
        model_status=model_status_msg,
        prediction_result=prediction_result,
        csv_result=csv_result_html,
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
        return render_main_page(request, model_status_msg="❌ Модель не загружена")

    form_data_raw = await request.form()
    form_data = dict(form_data_raw)

    # 1. Собираем сырые данные
    row = {}
    for f in FEATURES:
        val = form_data.get(f["name"], "")
        if f["type"] == "number" and val != "":
            row[f["name"]] = float(val)
        else:
            row[f["name"]] = val

    # 2. Рассчитываем loan_percent_income
    row["loan_percent_income"] = row["loan_amnt"] / row["person_income"]

    # 3. One-hot encoding для образования
    education = row.pop("person_education")
    row["person_education_Bachelor"] = 1 if education == "Bachelor" else 0
    row["person_education_Doctorate"] = 1 if education == "Doctorate" else 0
    row["person_education_High School"] = 1 if education == "High School" else 0
    row["person_education_Master"] = 1 if education == "Master" else 0

    # 4. Пол (преобразуем в int)
    row["person_gender_male"] = int(row["person_gender_male"])

    # 5. Просрочки: Yes -> 1, No -> 0
    row["previous_loan_defaults_on_file"] = 1 if row["previous_loan_defaults_on_file"] == "Yes" else 0

    df = pd.DataFrame([row])

    try:
        prediction = int(model.predict(df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[0][1])
        result = {
            "loan_status": prediction,
            "probability": probability
        }
        return render_main_page(request, prediction_result=result, form_data=form_data)
    except Exception as e:
        return render_main_page(request, model_status_msg=f"❌ Ошибка предсказания: {e}", form_data=form_data)

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