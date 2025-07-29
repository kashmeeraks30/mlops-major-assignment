#Step 5: Dockerization
FROM python:3.9
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/utils.py .
COPY src/predict.py .
COPY models/ ./models/

CMD ["python","src/predict.py"]