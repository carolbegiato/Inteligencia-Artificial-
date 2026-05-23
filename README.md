# Previsão de Diabetes utilizando Regressão Logística
> Projeto de Inteligência Artificial
---

## Integrantes:
 - Carol Begiato - 10419181
 - Guilherme Ponciano - 10373466
 - Renata Ardito - 10417520

---

## Resumo
Este projeto tem como objetivo desenvolver um modelo de Inteligência Artificial capaz de prever a ocorrência de diabetes em pacientes com base em dados clínicos. Utilizando regressão logística, foi possível identificar padrões e realizar previsões com boa acurácia.

---

## Introdução

### a. Contextualização
A diabetes é uma doença crônica que afeta milhões de pessoas no mundo.

### b. Justificativa
A IA pode auxiliar na tomada de decisões médicas mais rápidas e precisas.

### c. Objetivo
Desenvolver um modelo de machine learning para prever diabetes.

### d. Opção do projeto
Uso de Python + Scikit-learn com regressão logística.

---

## Descrição do Problema
Prever a variável "Outcome" (0 ou 1) com base em dados clínicos.

---

## Aspectos Éticos
- Privacidade dos dados
- Possíveis vieses
- Uso responsável da IA
- Transparência

---

## Dataset
Dataset contendo:
- Glicose
- Pressão arterial
- IMC
- Idade
- Outcome

Preparação:
- Separação X e y
- Divisão treino/teste

---

## Metodologia

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
```

## Resultados

### Análise do Dataset

O dataset possui *768 registros* e *9 atributos*, todos numéricos, sem valores nulos.

Principais características:
- Média de glicose: *120.89*
- Idade média: *33 anos*
- Aproximadamente *35% dos pacientes possuem diabetes*

Observação importante:
Alguns atributos possuem valores iguais a *0 (ex: glicose, pressão)*, o que pode indicar dados ausentes ou inconsistentes.

---

### Desempenho do Modelo

O modelo de Regressão Logística apresentou:

- *Acurácia:* 74.67%

#### Relatório de Classificação:

| Classe | Precisão | Recall | F1-Score |
|--------|---------|--------|----------|
| 0 (Não diabético) | 0.81 | 0.79 | 0.80 |
| 1 (Diabético)     | 0.64 | 0.67 | 0.65 |

📌 Interpretação:
- O modelo é melhor em identificar pacientes *sem diabetes*
- Há mais dificuldade em prever corretamente pacientes *com diabetes*

---

### Importância das Variáveis

As variáveis mais relevantes para a previsão foram:

1. *DiabetesPedigreeFunction* → maior influência
2. *BMI (Índice de Massa Corporal)*
3. *Pregnancies*
4. *Age*

Variáveis com pouca influência:
- Insulin
- SkinThickness
- BloodPressure

---

### Exemplo de Previsão

Para um novo paciente, o modelo retornou:

👉 *Resultado: 1 (Possui diabetes)*