from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from pathlib import Path
from typing import List
import io

app = FastAPI(title="Mortgage ML Service")

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
            print("Модель загружена из диска")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            model = None
    return model


@app.on_event("startup")
def startup_event():
    load_model_from_disk()


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
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


def predict_dataframe(df: pd.DataFrame):
    if model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    return model.predict(df)


# ============================================
# ФРОНТЕНД - ГЛАВНАЯ СТРАНИЦА (без Jinja2)
# ============================================
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ипотечный консультант</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        h1 {
            color: white;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .status-bar {
            background: white;
            padding: 12px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status-bar.success { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
        .status-bar.warning { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
        .status-bar.error { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
        .block {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .block h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .hint {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 15px;
        }
        .form-field {
            margin-bottom: 12px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
        }
        .form-field label {
            width: 220px;
            font-weight: 600;
            color: #333;
        }
        .form-field input, .form-field select {
            flex: 1;
            min-width: 200px;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            margin-top: 15px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: scale(1.02);
        }
        .result {
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 13px;
        }
        th {
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }
        .approved { color: #28a745; font-weight: bold; }
        .rejected { color: #dc3545; font-weight: bold; }
        .file-input {
            display: block;
            margin: 10px 0;
        }
        .two-columns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .two-columns { grid-template-columns: 1fr; }
            .form-field { flex-direction: column; align-items: flex-start; }
            .form-field label { margin-bottom: 5px; }
        }
    </style>
</head>
<body>
    <h1>🏠 Предсказание одобрения ипотеки</h1>

    <div class="status-bar" id="statusBar">🔍 Проверка сервера...</div>

    <div class="two-columns">
        <!-- Левая колонка: Загрузка модели и ручной ввод -->
        <div>
            <div class="block">
                <h3>📦 1. Загрузка модели</h3>
                <p class="hint">Загрузите обученную модель в формате .pkl</p>
                <input type="file" id="modelFile" accept=".pkl" class="file-input">
                <button onclick="uploadModel()">Загрузить модель</button>
                <div id="uploadStatus" class="result"></div>
            </div>

            <div class="block">
                <h3>✍️ 2. Проверка одного клиента</h3>
                <p class="hint">Заполните все поля о клиенте</p>
                <div id="formFields"></div>
                <button onclick="predict()">Предсказать</button>
                <div id="singleResult" class="result"></div>
            </div>
        </div>

        <!-- Правая колонка: CSV -->
        <div class="block">
            <h3>📁 3. Массовая проверка (CSV)</h3>
            <p class="hint">Загрузите CSV-файл с данными клиентов</p>
            <input type="file" id="csvFile" accept=".csv" class="file-input">
            <button onclick="predictCSV()">Обработать CSV</button>
            <div id="csvResult" class="result"></div>
        </div>
    </div>

    <script>
        let modelLoaded = false;

        // Список признаков (из schemas.py)
        const FEATURES = [
            { name: "person_age", label: "Возраст", type: "number", placeholder: "Например: 30" },
            { name: "person_gender", label: "Пол", type: "select", options: ["male", "female"] },
            { name: "person_education", label: "Образование", type: "select", options: ["Bachelor", "Master", "PhD", "High School"] },
            { name: "person_income", label: "Доход", type: "number", placeholder: "Например: 50000" },
            { name: "person_emp_exp", label: "Опыт работы (лет)", type: "number", placeholder: "Например: 5" },
            { name: "person_home_ownership", label: "Владение жильём", type: "select", options: ["RENT", "OWN", "MORTGAGE", "OTHER"] },
            { name: "loan_amnt", label: "Сумма кредита", type: "number", placeholder: "Например: 200000" },
            { name: "loan_intent", label: "Цель кредита", type: "select", options: ["EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"] },
            { name: "loan_int_rate", label: "Процентная ставка", type: "number", step: "0.1", placeholder: "Например: 10.5" },
            { name: "loan_percent_income", label: "% дохода на кредит", type: "number", step: "0.01", placeholder: "Например: 0.3" },
            { name: "cb_person_cred_hist_length", label: "Кредитная история (лет)", type: "number", placeholder: "Например: 5" },
            { name: "credit_score", label: "Кредитный рейтинг", type: "number", placeholder: "Например: 700" },
            { name: "previous_loan_defaults_on_file", label: "Были просрочки", type: "select", options: ["Yes", "No"] }
        ];

        // Построение формы
        function buildForm() {
            const container = document.getElementById("formFields");
            container.innerHTML = "";
            FEATURES.forEach(f => {
                const div = document.createElement("div");
                div.className = "form-field";

                const label = document.createElement("label");
                label.textContent = f.label + ":";

                let input;
                if (f.type === "select") {
                    input = document.createElement("select");
                    input.id = f.name;
                    f.options.forEach(opt => {
                        const option = document.createElement("option");
                        option.value = opt;
                        option.textContent = opt;
                        input.appendChild(option);
                    });
                } else {
                    input = document.createElement("input");
                    input.type = f.type;
                    input.id = f.name;
                    input.placeholder = f.placeholder || "";
                    if (f.step) input.step = f.step;
                }

                div.appendChild(label);
                div.appendChild(input);
                container.appendChild(div);
            });
        }

        // Сбор данных из формы
        function getFormData() {
            const data = {};
            FEATURES.forEach(f => {
                const el = document.getElementById(f.name);
                if (el) {
                    let value = el.value;
                    if (f.type === "number" && value !== "") {
                        value = parseFloat(value);
                    }
                    data[f.name] = value;
                }
            });
            return data;
        }

        // Проверка статуса
        async function checkStatus() {
            try {
                const resp = await fetch('/health');
                const data = await resp.json();
                modelLoaded = data.model_loaded;
                const statusBar = document.getElementById('statusBar');
                if (modelLoaded) {
                    statusBar.innerHTML = '✅ Модель загружена. Сервер готов к работе.';
                    statusBar.className = 'status-bar success';
                } else {
                    statusBar.innerHTML = '⚠️ Модель не загружена. Загрузите .pkl файл.';
                    statusBar.className = 'status-bar warning';
                }
            } catch(e) {
                document.getElementById('statusBar').innerHTML = '❌ Нет связи с сервером';
                document.getElementById('statusBar').className = 'status-bar error';
            }
        }

        // Загрузка модели
        async function uploadModel() {
            const fileInput = document.getElementById('modelFile');
            const file = fileInput.files[0];
            if (!file) {
                alert('Выберите .pkl файл');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerHTML = '⏳ Загрузка...';

            try {
                const resp = await fetch('/upload-model', { method: 'POST', body: formData });
                const data = await resp.json();
                if (resp.ok) {
                    statusDiv.innerHTML = '✅ ' + data.message;
                    await checkStatus();
                } else {
                    statusDiv.innerHTML = '❌ ' + (data.detail || 'Ошибка');
                }
            } catch(e) {
                statusDiv.innerHTML = '❌ Ошибка: ' + e.message;
            }
        }

        // Ручное предсказание
        async function predict() {
            if (!modelLoaded) {
                alert('Сначала загрузите модель');
                return;
            }

            const record = getFormData();
            const resultDiv = document.getElementById('singleResult');
            resultDiv.innerHTML = '⏳ Предсказание...';

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify([record])
                });
                const data = await resp.json();

                if (resp.ok && data.length > 0) {
                    const pred = data[0];
                    if (pred.loan_status === 1) {
                        resultDiv.innerHTML = '<span class="approved">✅ ОДОБРЕНО</span><br><small>' + pred.status_text + '</small>';
                    } else {
                        resultDiv.innerHTML = '<span class="rejected">❌ ОТКАЗАНО</span><br><small>' + pred.status_text + '</small>';
                    }
                } else {
                    resultDiv.innerHTML = '❌ Ошибка: ' + JSON.stringify(data);
                }
            } catch(e) {
                resultDiv.innerHTML = '❌ Ошибка: ' + e.message;
            }
        }

        // CSV предсказание
        async function predictCSV() {
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            if (!file) {
                alert('Выберите CSV файл');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            const resultDiv = document.getElementById('csvResult');
            resultDiv.innerHTML = '⏳ Обработка...';

            try {
                const resp = await fetch('/predict-from-csv', { method: 'POST', body: formData });
                const data = await resp.json();

                if (resp.ok) {
                    let html = '';
                    if (data.roc_auc) {
                        html += `<p><strong>📊 ROC-AUC:</strong> ${data.roc_auc.toFixed(4)}</p>`;
                    }
                    html += `<p><strong>Всего строк:</strong> ${data.rows.length}</p>`;

                    if (data.rows && data.rows.length) {
                        html += '<div style="overflow-x: auto;"><table>';
                        const firstRow = data.rows[0];
                        html += '<thead><tr>';
                        for (let key in firstRow) {
                            html += `<th>${key}</th>`;
                        }
                        html += '</tr></thead><tbody>';

                        data.rows.slice(0, 20).forEach(row => {
                            html += '<tr>';
                            for (let value of Object.values(row)) {
                                let display = value;
                                if (typeof value === 'number') display = value.toFixed(2);
                                html += `<td>${display}</td>`;
                            }
                            html += '</tr>';
                        });
                        html += '</tbody></table></div>';
                    }
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = '❌ Ошибка: ' + (data.detail || 'Неизвестная ошибка');
                }
            } catch(e) {
                resultDiv.innerHTML = '❌ Ошибка: ' + e.message;
            }
        }

        // Добавляем эндпоинт health (если нет)
        async function init() {
            buildForm();
            checkStatus();
            setInterval(checkStatus, 10000);
        }

        init();
    </script>
</body>
</html>
    """


# ============================================
# ЭНДПОИНТЫ API
# ============================================
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict_api(data: List[dict]):
    df = pd.DataFrame(data)
    predictions = predict_dataframe(df)

    result = []
    for i, prediction in enumerate(predictions):
        result.append({
            **data[i],
            "loan_status": int(prediction),
            "status_text": "одобрено" if int(prediction) == 1 else "отказ"
        })
    return result


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)):
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
        df["predicted_status_text"] = df["predicted_loan_status"].apply(
            lambda x: "одобрено" if int(x) == 1 else "отказ"
        )

        if "loan_status" in df.columns and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(y_true, probabilities)

        return {
            "roc_auc": roc_auc,
            "rows": df.to_dict(orient="records")
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Ошибка CSV: {error}")


# ============================================
# ЗАПУСК
# ============================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)