# Put the code for your API here.
import os
from typing import Union 
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
#from starter.ml.data import process_data
#from starter.ml.model import inference, load_model_and_encoder
import sys
sys.path.insert(0, "starter/starter")
from ml.data import process_data
from ml.model import inference, load_model_and_encoder

# Declare the data object with its components and their type.
class InferenceData(BaseModel):
    age: int = 39
    workclass: str = "State-gov"
    fnlgt: int = 77516
    education: str = "Bachelors"
    education_num: int = Field(example=13, alias="education-num")
    marital_status: str = Field(example="Never-married", alias="marital-status")
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = Field(example=2174, alias="capital-gain")
    capital_loss: int = Field(example=0, alias="capital-loss")
    hours_per_week: int = Field(example=40, alias="hours-per-week")
    native_country: str = Field(example="United-States",
                                alias="native-country")

app = FastAPI()

# Define a POST
@app.post("/inference/")
async def predict(data: InferenceData):
    
    path = os.path.dirname(__file__)
    model_path = os.path.join(path, "./model/lr_model.joblib") 
    encoder_path = os.path.join(path, "./model/encoder.joblib") 
    lb_path = os.path.join(path, "./model/label_binarizer.joblib")
    lr_model, encoder, lb = load_model_and_encoder(model_path, encoder_path, lb_path)
    
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
    ]

    X =pd.DataFrame(data = [data.dict(by_alias=True)], index = [0])
    
    X,_,_,_ = process_data(X, cat_features, label=None, training=False, encoder=encoder, lb=lb)
    
    prediction = inference(lr_model, X)

    prediction = prediction.tolist()

    return prediction


# Define a GET on the specified endpoint.
@app.get("/")
async def greetings():
    return "Hello Visitor!"
