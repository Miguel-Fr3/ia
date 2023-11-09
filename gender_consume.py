# from flask import Flask, request
# import traceback

# app = Flask(_name_)

# @app.route("/gender", methods=['POST'])
# def gender():
#     book = request.jsontry
#     try:

#         return book, 200
#     except Exception:
#         traceback.print_exc()
#         error = {
#             "erro": "A classificação falhou!"
#         }
#         return info, 400

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregue o modelo treinado
model = joblib.load('modelo_gender_classification.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        long_hair = data['long_hair']
        forehead_width_cm = data['forehead_width_cm']
        forehead_height_cm = data['forehead_height_cm']
        nose_wide = data['nose_wide']
        nose_long = data['nose_long']
        lips_thin = data['lips_thin']
        distance_nose_to_lip_long = data['distance_nose_to_lip_long']

        # Faça previsões com base nos dados de entrada
        prediction = model.predict([[long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]])[0]

        # Converta a previsão em texto
        gender = "Male" if prediction == 0 else "Female"

        response = {'gender': gender}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
