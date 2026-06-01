#!/usr/bin/env python
# coding: utf-8

# # Обработка данных обучающего датасета

# Импортируем библиотеки и загружаем данные

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import jupyter

data = pd.read_csv('train_data/loan_data.csv')
df = pd.DataFrame(data)
df.head(5)


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# Заметим, что есть невозможные значения: возраст 144 года, стаж работы 125 лет

# In[ ]:


df = df[df["person_age"] <= 100]
df = df[df["person_emp_exp"] <= df["person_age"] - 14]


# In[ ]:


df.describe().T


# Для всех числовых признаков посмотрим распредления

# In[ ]:


numeric_features = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]


# In[ ]:


import matplotlib.pyplot as plt

for column in numeric_features:
    plt.figure(figsize=(8, 4))

    plt.hist(
        df[column],
        bins=30
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")

    plt.show()


# Сильная ассиметрия дохода

# In[ ]:


df["person_income"] = np.log1p(df["person_income"])


# Для категориальных признаков используем One-hot-encoding

# In[ ]:


categorical_features = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
]

df = pd.get_dummies(
    df,
    columns=categorical_features,
    drop_first=True
)


# In[ ]:


df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})


# In[ ]:


df.info()


# In[ ]:


df.to_csv("new_loan_data.csv", index=False)

