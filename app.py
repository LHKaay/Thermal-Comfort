from modules import model_inference
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/thermal_sensation")
def thermal_sensation(Height: float, Weight: float, Clothing: float, Metabolic: float, 
                      PMV: float, PPD: float, 
                      Indoor_Temperature: float, Indoor_Humidity: float, Indoor_Velocity: float, 
                      Summer_Season: bool, Transition_Season: bool, Winter_Season: bool, 
                      Cold_zone: bool, Hot_summer_and_cold_winter_zone: bool, Hot_summer_and_warm_winter_zone: bool,
                      Mild_zone: bool, Severe_cold_zone: bool,
                      Female: bool, Male: bool):
    
    thermal_features =1*[Height, Weight, Clothing, Metabolic, 
                      PMV, PPD, 
                      Indoor_Temperature, Indoor_Humidity, Indoor_Velocity, 
                      Summer_Season, Transition_Season, Winter_Season, 
                      Cold_zone, Hot_summer_and_cold_winter_zone, Hot_summer_and_cold_winter_zone_, Hot_summer_and_warm_winter_zone,
                      Mild_zone, Sever_cold_zone, Severe_cold_zone,
                      Female, Male]
    
    features = thermal_features
    thermal_sensation = model_inference.model_pipeline(features)
    
    return thermal_sensation