# возвращаются ли ошибки

from io import BytesIO

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_predict_from_csv_without_model():
    csv_content = (
        "person_income,person_emp_exp,loan_amnt,loan_int_rate,"
        "cb_person_cred_hist_length,credit_score,"
        "previous_loan_defaults_on_file,person_gender_male,"
        "person_education\n"
        "75000,10,20000,12.5,8,710,No,True,Bachelor\n"
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