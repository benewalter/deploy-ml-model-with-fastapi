from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
#from starter.ml.data import process_data
import joblib
import os
import pandas as pd

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Logistic Regression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds

def save_model_and_encoder(model, encoder, lb, path):
    """ Saves trained machine learning model and encoder and label binarizer.

    Inputs
    ------
    model : Logistic Regression
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer
    path : string
        Path for saving the model, encoder, and label binarizer
    Returns
    -------
    None
    """
    
    model_path = os.path.join(path, "../model/lr_model.joblib") 
    joblib.dump(model, model_path)
    
    encoder_path = os.path.join(path, "../model/encoder.joblib") 
    joblib.dump(encoder, encoder_path)
    
    lb_path = os.path.join(path, "../model/label_binarizer.joblib") 
    joblib.dump(lb, lb_path)    
    

def load_model_and_encoder(model_path, encoder_path, lb_path):
    """ Loads trained machine learning model and encoder and label binarizer.

    Inputs
    ------
    model_path : string
        Path to trained machine learning model.
    encoder_path : string
        Path to trained OneHotEncoder
    lb_path : string
        Path to trained LabelBinarizer
    Returns
    -------
    model : model object
        Trained machine learning model
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer
    """
    
    model = joblib.load(model_path)
    
    encoder = joblib.load(encoder_path)
    
    lb = joblib.load(lb_path)  
    
    return model, encoder, lb

def compute_model_metrics_on_slices(df, cat_features, encoder, lb, model):
    """
    Validates the trained machine learning model on data slices using precision, recall, and F1.

    Inputs
    ------
    df : Pandas dataframe
        Test data based on which to do the evaluation.
    cat_features : list
        List of categorical features.
    model : Logistic Regression
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer
    Returns
    -------
    None
    """
    
    performance_on_slices = {}
    
    for feature in cat_features:
        for cls in df[feature].unique():
            data_slice = df.loc[df[feature] == cls]
            
            X_test, y_test, _, _ = process_data(
                                    data_slice, categorical_features=cat_features, label="salary", 
                                        encoder= encoder, lb = lb,training=False
                                    )

            predictions = inference(model, X_test)
            
            precision, recall, fbeta = compute_model_metrics(y_test, predictions)
            print(str(feature) + " " + str(cls))
            print(precision)
            print(recall)
            print(fbeta)
            
            performance_on_slices[feature + "__" + cls] = compute_model_metrics(y_test, predictions)
            
    performance_on_slices = pd.DataFrame(performance_on_slices).T
    performance_on_slices.columns = ['Precision', 'Recall', 'Fbeta']
    performance_on_slices.to_csv(r'slice_output.txt', header=True, index=True, sep=' ', mode='a')
