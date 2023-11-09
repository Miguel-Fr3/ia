
# Pré processamento: mapenado valores numérico para as classificações
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
# Carregue seus dados em um DataFrame, por exemplo:
data = pd.read_csv('./dataset/gender_classification_v7.csv')

# Divida os dados em recursos (X) e rótulos (y)
X = data[['long_hair', 'forehead_width_cm', 'forehead_height_cm', 'nose_wide', 'nose_long', 'lips_thin', 'distance_nose_to_lip_long']]
y = data['gender']

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o modelo de Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Faça previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avalie o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Acurácia do modelo:", accuracy)
print("\nRelatório de classificação:\n", report)


joblib.dump(model, 'modelo_gender_classification.pkl')
