# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, save_model_and_encoder, load_model_and_encoder, compute_model_metrics_on_slices
import joblib
import os
import logging


# Add the necessary imports for the starter code.

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Add code to load in the data.
logging.info('Loading data')
#data = pd.read_csv(os.getcwd()[:-8] + "/data/census.csv", skipinitialspace=True)
#data = pd.read_csv("../data/census.csv", skipinitialspace=True)

#print(data.head())
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/census.csv"), skipinitialspace=True) 
#print(str(os.path.dirname(__file__)))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logging.info('Splitting data into train and test set')
train, test = train_test_split(data, test_size=0.20)

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

# Process the train data with the process_data function.
logging.info('Processing train data')
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
logging.info('Processing test data')
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", 
    encoder= encoder, lb = lb,training=False
)

# Train and save a model.

# Train logistic regression
logging.info('Training machine learning model')
lr_model = train_model(X_train, y_train)

# Apply logistic regression to test set
logging.info('Applying machine learning model')
predictions = inference(lr_model, X_test)

# Check results
logging.info('Checking inference results')
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("Fbeta: " + str(fbeta))

# Check results on slices of data
logging.info('Checking inference results on slices of data')
compute_model_metrics_on_slices(test, cat_features, encoder, lb, lr_model)

# Save model, encoder, and label binarizer
logging.info('Saving machine learning model, encoder, and label binarizer')
#model_path = os.getcwd()[:-8] + "/model/lr_model.joblib"
#model_path = os.path.join(os.path.dirname(__file__), "../model/lr_model.joblib") 
#model_path = "../model/lr_model.joblib"
#joblib.dump(lr_model, model_path)
path = os.path.dirname(__file__)
save_model_and_encoder(lr_model, encoder, lb, path)


# Load model, encoder, and label binarizer
logging.info('Loading machine learning model, encoder, and label binarizer')

model_path = os.path.join(path, "../model/lr_model.joblib") 
encoder_path = os.path.join(path, "../model/encoder.joblib") 
lb_path = os.path.join(path, "../model/label_binarizer.joblib")
lr_model, encoder, lb = load_model_and_encoder(model_path, encoder_path, lb_path)
