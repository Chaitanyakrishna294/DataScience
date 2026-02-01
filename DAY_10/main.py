from  fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
#1 intilializ
app=FastAPI(title="AQI Prediction")
#2 loading model and encoder
try:
    artifacts=joblib.load("aqi_model.pkl")
    model=artifacts['model']
    le=artifacts['encoder']
    print("model and encoder loaded")
except Exception as e:
    print(f"Error loading: {e}")

#3 defining input data

class AQIdata(BaseModel):
    pm25:float
    pm10:float
    no2:float   
    co:float
    so2:float
    o3:float
    city:str
#4 prediction
@app.post("/predict_aqi/")
def predict(data:AQIdata):
    try:
        clean_city=data.city.title()
        if clean_city not in le.classes_:
            return{
                "error"
            }
        city_code=le.transform([clean_city])[0]
        features=[[data.pm25,data.pm10,data.no2,data.co,data.so2,data.o3,city_code]]
        aqi=float(model.predict(features)[0])
        if aqi<=50: status="Good"
        elif aqi<=100: status="Moderate"
        elif aqi<=200: status="Poor"
        elif aqi<=300: status="Very Poor"
        else: status="Severe"
        
        return {
            "City":clean_city,
            "Predicted_AQI":round(aqi,2),
            "Status":status
        }
    except Exception as e:
        return{"error":str(e)}

@app.get("/")
def home():
    return {"message":"AQI API is running"}
