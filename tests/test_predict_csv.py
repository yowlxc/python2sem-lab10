# возвращаются ли ошибки
from io import BytesIO

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_predict_from_csv_without_model():
    csv_content = (
        "person_age,person_gender,person_education,person_income,"
        "person_emp_exp,person_home_ownership,loan_amnt,loan_intent,"
        "loan_int_rate,loan_percent_income,cb_person_cred_hist_length,"
        "credit_score,previous_loan_defaults_on_file\n"
        "35,male,Bachelor,75000,10,RENT,20000,HOMEIMPROVEMENT,"
        "12.5,0.26,8,710,No\n"
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