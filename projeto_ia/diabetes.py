#Caroline Begiato Cabral; Guilherme Ponciano; Renata Ardito

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#verificando valores
df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.info())
print(df.describe())

#preparando dados
cols = ['Glucose', 'BloodPressure', 'BMI', 'Insulin']
for col in cols:
    df[col] = df[col].replace(0, df[col].mean())
print((df[cols] == 0).sum())

#análise de correlação
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()