# from flask import Flask, jsonify, request
# import joblib
# import pickle

# app = Flask(__name__)

# arima_model = joblib.load('arima_model.sav')

# # with open('arima_model.pkl', 'rb') as file:
# #     arima_model = pickle.load(file)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     passenger_count = data['passenger_count']
#     prediction = arima_model.predict(n_periods=1, X=passenger_count)
#     return jsonify({'prediction': prediction.tostr()})

# if __name__ == '__main__':
#     app.run(port="8080", debug=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import List
import uvicorn

app = FastAPI()

arima_model = joblib.load('arima_model.sav')

class PredictionInput(BaseModel):
    passenger_count: List[float]

class PredictionOutput(BaseModel):
    prediction: List[float]

@app.get("/")
async def check():
    return {"message": "ARIMA predicition passangers flights"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Make prediction
        prediction = arima_model.predict(n_periods=1, X=input_data.passenger_count)
        
        prediction_list = prediction.tolist()
        
        return PredictionOutput(prediction=prediction_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, port=8080)