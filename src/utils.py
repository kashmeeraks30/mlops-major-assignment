from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dataset():
    housingdata=fetch_california_housing()
    X,y=housingdata.data,housingdata.target
    return X,y

def train_model(X,y):
    model=LinearRegression()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_normalize_train = scaler.fit_transform(X_train)
    X_normalize_test = scaler.transform(X_test)
    model.fit(X_normalize_train,y_train)
    return model,X_normalize_test,y_test

def model_metrics(model,X_test,y_test):
    y_pred = model.predict(X_test)
    R2_Score = r2_score(y_test,y_pred)
    loss = mean_squared_error(y_test,y_pred)
    print(f"Original R2 score: {R2_Score:.4f}")
    print(f"Original Loss:{loss:.4f}")

def save_model(model):
    modelname='models/california_housing.joblib'
    os.makedirs('models/',exist_ok=True)
    joblib.dump(model,modelname)
    print("Model saved successfully")

def load_model(modelpath="models/california_housing.joblib"):
    model=joblib.load(modelpath)
    return(model)

def extract_parameters(model):
    coef=model.coef_
    intercept=model.intercept_
    return coef,intercept

def parameter_quantization(parameters, nbits):
    level = 2 ** nbits - 1
    value = np.round(parameters * level).astype(np.uint8)
    return value

def parameter_dequantization(quant_parameters, min_value, max_value, nbits):
    level = 2 ** nbits - 1
    # Convert back to float and normalize to [0, 1]
    normalized_params = quant_parameters.astype(np.float32) / level
    # Scale back to original range
    dequantized_params = normalized_params * (max_value - min_value) + min_value
    return dequantized_params

