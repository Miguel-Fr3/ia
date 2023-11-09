
# Divide os dados em conjunto de treinamento e teste
from sklearn.model_selection import train_test_split
# Calcula a assertividade do modelo
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

gender_class = pd.read_csv('./dataset/gender_classification_v7.csv')
# Pré processamento: mapenado valores numérico para as classificações
categories = { 'Male': 0, 'Female': 1}

# Redimensionando com reshape para obter uma matriz 2D [[]] -1 diemnsiona para o tamanho total de elementos e 1 define 1 unico elemento
x = np.array(gender_class['BMI']).reshape(-1, 1)
# Fazendo o parse da label para seu representante numérico
y = gender_class["Label"].map(categories).tolist()

# test_size define 20% dos registros para teste e 80% para treino
# random_state garante que a aleatoriedade dos dados seja reproduzivel retornando sempre a mesma ordem quando o valor fo 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
classificationRL = classification_report(y_test, y_pred)

print(accuracy)
print(classificationRL)