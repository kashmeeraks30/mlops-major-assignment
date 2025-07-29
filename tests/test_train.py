#Step 3: Testing Pipeline
import pytest
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from src.utils import train_model,load_dataset,model_metrics,save_model
from sklearn.metrics import r2_score

#Test if dataset is loaded successfully
def test_dataset_loading():

    #Test if the dimensions of the loaded dataset matches the original dimensions    
    housingdata=fetch_california_housing()
    X,y = load_dataset()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    assert X.shape[0]==20640, "Dataset should have 20640 samples"
    assert X.shape[1]==8, "Dataset should have 8 features"
    assert len(y)==20640, "Target should have 20640 samples"

    #Test if all the expected features are present in the loaded dataset    
    expected_features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
    
    for feature in expected_features:
        assert feature in housingdata.feature_names, f"Feature '{feature}' not found in dataset"

#Validate model creation (LinearRegression instance)
def test_validate_model():
    
    X,y = load_dataset()
    model,X_test,y_test=train_model(X,y)

    #Test if the train_model function returns a LogisticRegression object
    assert isinstance(model,LinearRegression),"Model should be LinearRegression"

    #Test if the model attributes are present and model was trained
    assert hasattr(model,'coef_'),"Missing coef_ attribute, object must be fitted"
    assert hasattr(model,'intercept_'), "Missing classes_ attribute, object must be fitted"

#Test if R2 score exceeds minimum threshold 
def test_r2_score():
    
    X,y = load_dataset()

    model,X_test,y_test = train_model(X,y)

    y_pred = model.predict(X_test)
    R2_Score = r2_score(y_test,y_pred)

    assert R2_Score>0.5, f"R2 score doesnt exceed the minimal threshold"

  

        
