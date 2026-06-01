# работает ли API
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_predict_without_model():
    data = [
        {
            "person_age": 35,
            "person_gender": "male",
            "person_education": "Bachelor",
            "person_income": 75000,
            "person_emp_exp": 10,
            "person_home_ownership": "RENT",
            "loan_amnt": 20000,
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_int_rate": 12.5,
            "loan_percent_income": 0.26,
            "cb_person_cred_hist_length": 8,
            "credit_score": 710,
            "previous_loan_defaults_on_file": "No"
        }
    ]

    response = client.post("/predict", json=data)

    assert response.status_code == 400
    assert response.json()["detail"] == "Модель не загружена"