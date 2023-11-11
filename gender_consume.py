from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('modelo_gender_classification.pkl')

def realizar_previsao(data):
    long_hair = data['long_hair']
    forehead_width_cm = data['forehead_width_cm']
    forehead_height_cm = data['forehead_height_cm']
    nose_wide = data['nose_wide']
    nose_long = data['nose_long']
    lips_thin = data['lips_thin']
    distance_nose_to_lip_long = data['distance_nose_to_lip_long']

    prediction = model.predict([[long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]])[0]

    gender = "Male" if prediction == 0 else "Female"
    return gender

@app.route("/", methods=['GET'])
def main():
    content = """
    <h1> Modelo classificador de Gênero </h1>
    <ul>
        <li> 0: Male</li>
        <li> 1: Female</li>
    </ul>

    <p> Para classificar, faça uma solicitação POST para http://127.0.0.1:5000/predict com os seguintes parâmetros:</p>
    <ul>
        <li> long_hair</li>
        <li> forehead_width_cm</li>
        <li> forehead_height_cm</li>
        <li> nose_wide</li>
        <li> nose_long</li>
        <li> lips_thin</li>
        <li> distance_nose_to_lip_long</li>
    </ul>
    """
    return content

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        gender = realizar_previsao(data)

        response = {'gender_prediction': gender}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Erro na previsão: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)


import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "long_hair": 1,
    "forehead_width_cm": 15.0,
    "forehead_height_cm": 10.0,
    "nose_wide": 1,
    "nose_long": 0,
    "lips_thin": 1,
    "distance_nose_to_lip_long": 5.0
}

headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())