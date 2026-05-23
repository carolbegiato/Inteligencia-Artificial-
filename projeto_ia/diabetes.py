# ==============================================================================
# PROJETO: [Nome do seu Projeto - ex: Previsão de Diabetes e Análise com LLM]
# 
# INTEGRANTES:
# - Carol Begiato - 10419181
# - Guilherme Ponciano - 10373466
# - Renata Ardito - 10417520
# 
# SÍNTESE DO ARQUIVO:
# Este notebook contém a análise exploratória do dataset de diabetes, 
# o treinamento de um modelo preditivo (Random Forest) e a integração com 
# um LLM para interpretação de resultados.
# ==============================================================================

# ==============================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# CARREGAMENTO DO DATASET
# ==============================
# Exemplo: dataset de diabetes (troque pelo seu arquivo se necessário)
df = pd.read_csv("projeto_ia/diabetes.csv")

# Visualizar dados
print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())

print("\nEstatísticas:")
print(df.describe())

# ==============================
# ANÁLISE EXPLORATÓRIA (EDA)
# ==============================

# Verificar valores nulos
print("\nValores nulos:")
print(df.isnull().sum())

# Distribuição da variável alvo
sns.countplot(x="Outcome", data=df)
plt.title("Distribuição do Resultado (Diabetes)")
plt.show()

# Correlação entre variáveis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

# ==============================
# PREPARAÇÃO DOS DADOS
# ==============================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TREINAMENTO DO MODELO
# ==============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# ==============================
# PREDIÇÃO
# ==============================

y_pred = model.predict(X_test)

# ==============================
# AVALIAÇÃO DO MODELO
# ==============================

print("\nAcurácia:", accuracy_score(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# ==============================
# IMPORTÂNCIA DAS VARIÁVEIS
# ==============================

coeficientes = pd.DataFrame({
    "Variável": X.columns,
    "Peso": model.coef_[0]
})

coeficientes = coeficientes.sort_values(by="Peso", ascending=False)

print("\nImportância das Variáveis:")
print(coeficientes)

# ==============================
# TESTE COM NOVO DADO (EXEMPLO)
# ==============================

novo_paciente = np.array([[6,148,72,35,0,33.6,0.627,50]])
previsao = model.predict(novo_paciente)

print("\nPrevisão para novo paciente:", previsao)