# проверка ошибки, если модель не загружена

from io import BytesIO

from fastapi.testclient import TestClient

from backend.main import app
from backend import model_service


client = TestClient(app)


def test_predict_from_csv_without_model():
    model_service.model = None

    csv_content = (
        "person_income,person_emp_exp,loan_amnt,loan_int_rate,"
        "cb_person_cred_hist_length,credit_score,"
        "previous_loan_defaults_on_file,person_gender,"
        "person_education,person_home_ownership,loan_intent\n"
        "75000,10,20000,12.5,8,710,"
        "Нет,Женский,Бакалавр,Аренда,Ремонт\n"
    )

    files = {
        "file": (
            "test.csv",
            BytesIO(csv_content.encode("utf-8")),
            "text/csv"
        )
    }

    response = client.post("/predict-from-csv", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Модель не загружена"