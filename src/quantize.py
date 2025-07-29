#Step 4: Manual Quantization
import numpy as np
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from utils import load_model,extract_parameters,parameter_quantization,parameter_dequantization,model_metrics


def main():
    
    housingdata = fetch_california_housing()
    X, y = housingdata.data, housingdata.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_normalize_train = scaler.fit_transform(X_train)
    X_normalize_test = scaler.transform(X_test)

    #Loading the saved scikit-learn model from .joblib file
    model=load_model()


    #Extracting the parameters 
    coef,intercept=extract_parameters(model)
    print(f"Coef:{coef}")
    print(f"Intercept:{intercept}")

    # Raw parameters stored in a dictionary and saved as unquant_params.joblib
    unquantized_param = {'coef': coef, 'intercept': intercept}
    joblib.dump(unquantized_param, 'models/unquant_params.joblib')
    print(f"\nRaw parameters saved as unquant_params.joblib!")
    print(f"Size of original sklearn model: {(os.path.getsize('models/unquant_params.joblib')/1024):.2f} KB")

    # Normalizing the parameter values between 0 and 1
    parameters = np.asarray(coef)
    parameters = np.append(parameters, intercept)

    min_value = parameters.min()
    max_value = parameters.max()
    
    # Normalize to [0, 1] range
    parameters_normalized = (parameters - min_value) / (max_value - min_value)
    
    # Performing manual quantization of parameters to unsigned 8 bit integer
    quantized_parameters = parameter_quantization(parameters_normalized, 8)
    
    # Quantized parameters stored in a dictionary and saved as manual_quant_params.joblib
    quantized_coef = quantized_parameters[:-1]
    quantized_intercept = quantized_parameters[-1]

    quantized_param = {'coef': quantized_coef, 'intercept': quantized_intercept}
    joblib.dump(quantized_param, 'models/manual_quant_params.joblib')
    print(f"Size of manually quantized model: {(os.path.getsize('models/manual_quant_params.joblib')/1024):.2f} KB")

    #De-quantization
    quantized_params = np.append(quantized_coef, quantized_intercept)
    dequantized_params = parameter_dequantization(quantized_params, min_value, max_value, 8)
    
    # Split back into coefficients and intercept
    dequant_coef = dequantized_params[:-1]
    dequant_intercept = dequantized_params[-1]
    
    # Use scaled test data for inference
    y_pred_dequant = np.dot(X_normalize_test, dequant_coef) + dequant_intercept

    r2_dequantized_model = r2_score(y_test, y_pred_dequant)
    mse_dequantized = mean_squared_error(y_test, y_pred_dequant)

    print(f"\nInference results with de-quantized weights:")
    model_metrics(model,X_normalize_test,y_test)
    print(f"De-Quantized model R2 score: {r2_dequantized_model:.4f}")
    print(f"De-Quantized MSE: {mse_dequantized:.4f}")

if __name__ =="__main__":
    main()