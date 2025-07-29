#Step 2: Model Training

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from utils import train_model,load_dataset,model_metrics,save_model

def main():

    #Load dataset
    X,y = load_dataset()

    #Train LinearRegression model
    model,X_test,y_test = train_model(X,y)

    #Print R2 score and loss
    model_metrics(model,X_test,y_test)

    #Save model using joblib
    save_model(model)

if __name__ == "__main__":
    main()