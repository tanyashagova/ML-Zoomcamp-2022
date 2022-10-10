import pickle


model_path = 'model1.bin'
dv_path = 'dv.bin'


# loading the model
with open(model_path, 'rb') as model_loading:
    model = pickle.load(model_loading)

# loading the vectorizer
with open(dv_path, 'rb') as dv_loading:
    dv = pickle.load(dv_loading)


client = {"reports": 0, 
            "share": 0.001694, 
            "expenditure": 0.12,
            "owner": "yes"}

X = dv.transform([client])

y_pred = model.predict_proba(X)[0, 1]

print(y_pred.round(3))