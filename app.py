from flask import Flask, request
import numpy as np  
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

def prediction(model, scaler, data):
  o = data["Organic"]
  m = data["Moisture"]
  f = data["Fertiliser"]
  pest = data["Pesticide"]
  q = data["q"]
  n = data["N"]
  p = data["P"]
  k = data["K"]
  a = [[o,m,f,pest,q,n,p,k]]
  b = scaler.transform(a)
  pred = model.predict(b)
  a = pred[0]
  print("***************************************")
  print(pd.Series(a).to_json(orient='values'))
  print("***************************************")
  return pd.Series(a).to_json(orient='values')

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


agri_model = load_model("major_project_model.h5")
agri_scaler = joblib.load("major_project_agri.pkl")

@app.route('/api/agri', methods=['POST'])
def predict_yield():

    content = request.json
    
    results = prediction(model=agri_model,scaler=agri_scaler,data=content)
    
    return results

if __name__ == '__main__':
    app.run(debug=True)