import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_path = 'model1.bin'
dv_path = 'dv.bin'


# loading the model
with open(model_path, 'rb') as model_loading:
    model = pickle.load(model_loading)

# loading the vectorizer
with open(dv_path, 'rb') as dv_loading:
    dv = pickle.load(dv_loading)

app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    card = y_pred >= 0.5

    result = {
        "card_probability": float(y_pred),
        "card": bool(card)
    }
    return jsonify(result)




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    