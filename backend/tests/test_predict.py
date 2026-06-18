# работает ли API без загруженной модели

from fastapi.testclient import TestClient

from backend.main import app
from backend import model_service

client = TestClient(app)


def test_predict_without_model():
    model_service.model = None

    data = [
        {
            "person_income": 75000,
            "person_emp_exp": 10,
            "loan_amnt": 20000,
            "loan_int_rate": 12.5,
            "cb_person_cred_hist_length": 8,
            "credit_score": 710,
            "previous_loan_defaults_on_file": "Нет",
            "person_gender": "Женский",
            "person_education": "Бакалавр"
        }
    ]

    response = client.post("/predict", json=data)

    assert response.status_code == 400
    assert response.json()["detail"] == "Модель не загружена"