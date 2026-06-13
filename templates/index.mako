<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Ипотечный консультант</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            padding: 2rem;
            color: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #a0c0ff, #c084fc);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        /* Стили вкладок */
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid rgba(160, 192, 255, 0.3);
            padding-bottom: 0.5rem;
        }
        .tab-btn {
            background: transparent;
            border: none;
            padding: 0.8rem 1.8rem;
            font-size: 1rem;
            font-weight: bold;
            color: #b9c8ff;
            cursor: pointer;
            border-radius: 40px;
            transition: all 0.2s;
        }
        .tab-btn:hover {
            background: rgba(139, 92, 246, 0.3);
            color: white;
        }
        .tab-btn.active {
            background: linear-gradient(90deg, #5b4caf, #8b5cf6);
            color: white;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
        }
        .tab-content {
            display: none;
            animation: fadeIn 0.3s ease;
        }
        .tab-content.active {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px);}
            to { opacity: 1; transform: translateY(0);}
        }
        /* Карточки (общие) */
        .card {
            background: rgba(30, 30, 60, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 28px;
            padding: 1.5rem 2rem;
            border: 1px solid rgba(160, 192, 255, 0.3);
        }
        .card h3 {
            margin-bottom: 1.2rem;
            color: #c084fc;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 1rem;
        }
        .form-field {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
        .form-field label {
            font-weight: 500;
            font-size: 0.85rem;
            color: #b9c8ff;
        }
        .form-field input, .form-field select {
            background: rgba(20, 20, 45, 0.8);
            border: 1px solid #5a4c8a;
            border-radius: 20px;
            padding: 10px 14px;
            color: white;
            font-size: 0.9rem;
            outline: none;
        }
        .form-field input:focus, .form-field select:focus {
            border-color: #a855f7;
        }
        button, .button {
            background: linear-gradient(90deg, #5b4caf, #8b5cf6);
            border: none;
            padding: 10px 20px;
            border-radius: 40px;
            font-weight: bold;
            color: white;
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        button:hover {
            transform: scale(1.02);
        }
        .result-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(0,0,0,0.4);
            border-radius: 24px;
            border-left: 4px solid #a855f7;
        }
        .approved { color: #4ade80; font-weight: bold; font-size: 1.3rem; }
        .rejected { color: #f87171; font-weight: bold; font-size: 1.3rem; }
        .probability { margin-top: 0.5rem; background: #2a235a; display: inline-block; padding: 5px 14px; border-radius: 40px; }
        .file-input { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-top: 0.5rem; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.8rem; overflow-x: auto; display: block; }
        th, td { border: 1px solid #5a4c8a; padding: 6px; text-align: left; }
        th { background: #2a235a; color: #cba6ff; }
    </style>
</head>
<body>
<div class="container">
    <h1>🏠 Ипотечный консультант</h1>

    <!-- Вкладки -->
    <div class="tabs">
        <button class="tab-btn active" data-tab="model">📦 Управление моделью</button>
        <button class="tab-btn" data-tab="single">✍️ Проверка клиента</button>
        <button class="tab-btn" data-tab="csv">📁 Массовая проверка</button>
    </div>

    <!-- Вкладка 1: Загрузка модели -->
    <div id="tab-model" class="tab-content active">
        <div class="card">
            <h3>Загрузить обученную модель (.pkl)</h3>
            <form action="/upload-model-ui" method="post" enctype="multipart/form-data">
                <div class="file-input">
                    <input type="file" name="file" accept=".pkl" required>
                    <button type="submit">Загрузить модель</button>
                </div>
            </form>
            <div class="result-box">
                ${model_status if model_status else "⚡ Статус: модель не загружена"}
            </div>
        </div>
    </div>

    <!-- Вкладка 2: Проверка одного клиента -->
    <div id="tab-single" class="tab-content">
        <div class="card">
            <h3>Заполните данные клиента</h3>
            <form action="/predict-ui" method="post">
                <div class="grid-form">
                    % for f in features:
                    <div class="form-field">
                        <label>${f['label']}:</label>
                        % if f['type'] == 'select':
                            <select name="${f['name']}">
                                % for opt in f.get('options', []):
                                    <option value="${opt}" ${'selected' if form_data.get(f['name']) == opt else ''}>${opt}</option>
                                % endfor
                            </select>
                        % else:
                            <input type="${f['type']}" name="${f['name']}"
                                   value="${form_data.get(f['name'], '')}"
                                   placeholder="${f.get('placeholder', '')}" step="${f.get('step', '')}">
                        % endif
                    </div>
                    % endfor
                </div>
                <button type="submit">🔮 Предсказать</button>
            </form>

            % if prediction_result:
            <div class="result-box">
                % if prediction_result['loan_status'] == 1:
                    <div class="approved">✅ РЕШЕНИЕ: ОДОБРЕНО</div>
                % else:
                    <div class="rejected">❌ РЕШЕНИЕ: ОТКАЗАНО</div>
                % endif
                % if prediction_result.get('probability') is not None:
                    <div class="probability">📊 Вероятность одобрения: ${"%.1f" % (prediction_result['probability'] * 100)}%</div>
                % endif
            </div>
            % endif
        </div>
    </div>

    <!-- Вкладка 3: Массовая проверка CSV -->
    <div id="tab-csv" class="tab-content">
        <div class="card">
            <h3>Загрузите CSV-файл с данными</h3>
            <form action="/predict-csv-ui" method="post" enctype="multipart/form-data">
                <div class="file-input">
                    <input type="file" name="file" accept=".csv" required>
                    <button type="submit">Обработать CSV</button>
                </div>
            </form>
            % if csv_result:
            <div class="result-box">
                ${csv_result | n}
            </div>
            % endif
        </div>
    </div>
</div>

<!-- Минимальный JS для переключения вкладок (без конфликта с Mako) -->
<script>
    const tabsBtns = document.querySelectorAll('.tab-btn');
    const tabsContents = document.querySelectorAll('.tab-content');

    function switchTab(tabId) {
        tabsContents.forEach(content => {
            content.classList.remove('active');
        });
        tabsBtns.forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`tab-${tabId}`).classList.add('active');
        const activeBtn = Array.from(tabsBtns).find(btn => btn.dataset.tab === tabId);
        if (activeBtn) activeBtn.classList.add('active');
    }

    tabsBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });

    // Если после отправки формы нужно сохранить активную вкладку – передаём параметр ?tab=
    const urlParams = new URLSearchParams(window.location.search);
    const activeTab = urlParams.get('tab');
    if (activeTab && ['model', 'single', 'csv'].includes(activeTab)) {
        switchTab(activeTab);
    }
</script>
</body>
</html>