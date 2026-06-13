import joblib
import pandas as pd

# загружаем данные
df = pd.read_csv("new_loan_data.csv")

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# загружаем лучшую модель
model = joblib.load("our-model.pkl")

# дообучаем на всём датасете
model.fit(X, y)

# сохраняем обученную модель
joblib.dump(
    model,
    "our-model.pkl"
)

print("Модель обучена и сохранена.")