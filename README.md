Creation of Conda environment:
------------------------------
source ~/miniconda3/bin/activate
conda create --name majorassignment python=3.11
conda activate majorassignment


Cloning the remote github repository:
--------------------------------------
git clone git@github.com:kashmeeraks30/mlops-major-assignment.git


Creating and installing requirements.txt file:
---------------------------------------------
pip install -r requirements.txt


List of files:
-----------------
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


Output:
-------

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

Docker command:
docker run majorassignment python predict.py
Original R2 score: 0.5758
Original Loss:0.5559

Output Comparison Report:
-------------------------
Size of original sklearn model: 0.40 KB
Size of manually quantized model: 0.31 KB

Inference results with de-quantized weights:
Original R2 score: 0.5758
Original Loss:0.5559
De-Quantized model R2 score: 0.5758
De-Quantized MSE: 0.5559

From the above output, we can infer that the manually quantized model has occupied less space (0.31KB) compared to the original sklearn model 0.40KB. This shows that quantization has offered significant space savings by quantizing the parameters to unsigned 8-bit integers.

Regarding the R2 score and loss, the original R2 score was 0.5758 whereas the model after dequantization showed the same R2 score which is 0.5758. This implies that the model performance is perfectly maintained even after de-quantization.
Similarly, the loss remains the same for both original and dequantized models which is 0.5559. This suggests that no significant loss occurred during quantization-dequantization process. 

Thus, the results after dequantization has showed similar performance in terms of R2 score and loss compared to the original sklearn model. 
