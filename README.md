List of files:
Step 1: Repository Setup
.gitignore
requirements.txt 

Step 2: Model Training (src/train.py)
src/train.py 

Step 3: Testing Pipeline
tests/test_train.py

Step 4: Manual Quantization
src/quantize.py

Step 5: Dockerization
Dockerfile
src/predict.py

Step 6: CI/CD Workflow 
.github/workflows/ci.yml

Modularization:
src/utils.py


Output & Comparison:
Step 2: Model Training Output
python src/train.py
Original R2 score: 0.5758
Original Loss:0.5559
Model saved successfully

Step 3: Testing Pipeline
python -m pytest tests/test_train.py 
=================================================================== test session starts ===================================================================
platform linux -- Python 3.11.13, pytest-7.4.0, pluggy-1.6.0
rootdir: /home/kashmeera/mlops-major-assignment
collected 3 items                                                                                                                                         

tests/test_train.py ...                                                                                                                             [100%]

==================================================================== 3 passed in 4.48s ====================================================================

Step 4: Manual Quantization

python src/quantize.py
Coef:[ 0.85438303  0.12254624 -0.29441013  0.33925949 -0.00230772 -0.0408291
 -0.89692888 -0.86984178]
Intercept:2.071946937378619

Raw parameters saved as unquant_params.joblib!
Size of original sklearn model: 0.40 KB
Size of manually quantized model: 0.31 KB

Inference results with de-quantized weights:
Original R2 score: 0.5758
Original Loss:0.5559
De-Quantized model R2 score: 0.5758
De-Quantized MSE: 0.5559

Step 5: Dockerization
docker build -t majorassignment .
[+] Building 4.0s (13/13) FINISHED                                                                                                          docker:default

docker run majorassignment python predict.py
Original R2 score: 0.5758
Original Loss:0.5559