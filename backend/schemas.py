# Здесь описывается структура данных клиента.

from pydantic import BaseModel


class ClientData(BaseModel):
    # годовой доход клиента
    person_income: float

    # общий трудовой стаж
    person_emp_exp: int

    # сумма кредита
    loan_amnt: float

    # процентная ставка
    loan_int_rate: float

    # длина кредитной истории
    cb_person_cred_hist_length: float

    # кредитный рейтинг
    credit_score: int

    # были ли просрочки: Yes / No
    previous_loan_defaults_on_file: str

    # пол: male / female
    person_gender: str

    # образование:
    # Bachelor / Master / High School / Doctorate
    person_education: str