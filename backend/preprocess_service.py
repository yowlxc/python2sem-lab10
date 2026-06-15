import pandas as pd
from fastapi import HTTPException


MODEL_FEATURES = [
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
    "person_gender_male",
    "person_education_Bachelor",
    "person_education_Doctorate",
    "person_education_High School",
    "person_education_Master",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
]


def prepare_client_row(row: dict) -> dict:
    try:
        person_income = float(row["person_income"])
        loan_amnt = float(row["loan_amnt"])

        loan_percent_income = 0
        if person_income != 0:
            loan_percent_income = loan_amnt / person_income

        education = row.pop("person_education")
        gender = row.pop("person_gender")
        defaults = row["previous_loan_defaults_on_file"]

        home_ownership = row.pop("person_home_ownership", "Ипотека")
        loan_intent = row.pop("loan_intent", "DEBTCONSOLIDATION")

        row["person_income"] = person_income
        row["person_emp_exp"] = int(row["person_emp_exp"])
        row["loan_amnt"] = loan_amnt
        row["loan_int_rate"] = float(row["loan_int_rate"])
        row["loan_percent_income"] = loan_percent_income
        row["cb_person_cred_hist_length"] = float(
            row["cb_person_cred_hist_length"]
        )
        row["credit_score"] = int(row["credit_score"])

        row["previous_loan_defaults_on_file"] = int(
            str(defaults).lower() in ["да", "yes", "1", "true"]
        )

        row["person_gender_male"] = int(
            str(gender).lower() in ["мужской", "male", "man", "мужчина"]
        )

        row["person_education_Bachelor"] = int(
            education in ["Бакалавр", "Bachelor"]
        )

        row["person_education_Doctorate"] = int(
            education in ["Доктор наук", "Doctorate"]
        )

        row["person_education_High School"] = int(
            education in ["Среднее", "High School"]
        )

        row["person_education_Master"] = int(
            education in ["Магистр", "Master"]
        )

        row["person_home_ownership_OTHER"] = int(
            home_ownership in ["Другое", "OTHER"]
        )

        row["person_home_ownership_OWN"] = int(
            home_ownership in ["Собственное жильё", "OWN"]
        )

        row["person_home_ownership_RENT"] = int(
            home_ownership in ["Аренда", "RENT"]
        )

        row["loan_intent_EDUCATION"] = int(
            loan_intent in ["Образование", "EDUCATION"]
        )

        row["loan_intent_HOMEIMPROVEMENT"] = int(
            loan_intent in ["Ремонт", "HOMEIMPROVEMENT"]
        )

        row["loan_intent_MEDICAL"] = int(
            loan_intent in ["Медицина", "MEDICAL"]
        )

        row["loan_intent_PERSONAL"] = int(
            loan_intent in ["Личные нужды", "PERSONAL"]
        )

        row["loan_intent_VENTURE"] = int(
            loan_intent in ["Бизнес", "VENTURE"]
        )

        return {
            feature: row[feature]
            for feature in MODEL_FEATURES
        }

    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Не хватает признака: {error}"
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка предобработки данных: {error}"
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared_rows = []

    for _, row in df.iterrows():
        prepared_rows.append(
            prepare_client_row(row.to_dict())
        )

    return pd.DataFrame(
        prepared_rows,
        columns=MODEL_FEATURES
    )