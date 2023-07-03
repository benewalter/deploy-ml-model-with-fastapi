import pytest
import os

print(os.getcwd())

#from starter.ml.data import process_data
#from starter.ml.model import train_model
from ml.data import *
from ml.model import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture
def data():
    #df = pd.read_csv("../data/census.csv", skipinitialspace=True)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/census.csv"), skipinitialspace=True) 
    return df

def test_process_data(data):
    train, test = train_test_split(data, test_size=0.20)
    
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape[0] == y_train.shape[0]
    
    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", 
    encoder= encoder, lb = lb,training=False
    )
        
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
        
def test_train_model(data):
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", 
    encoder= encoder, lb = lb,training=False
    )
    
    lr_model = train_model(X_train, y_train)
    
    assert "sklearn.linear_model._logistic.LogisticRegression" in str(type(lr_model))

def test_inference(data):
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", 
    encoder= encoder, lb = lb,training=False
    )
    
    lr_model = train_model(X_train, y_train)
    
    predictions = lr_model.predict(X_test)
    
    assert np.max(predictions) == 1
    assert np.min(predictions) == 0
    
if __name__ == "__main__":
    data = data()
    test_process_data(data)
    test_train_model(data)
    test_inference(data)
