from prefect import flow, task
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

@task
def fetch_and_predict():
    logging.info("Starting email fetch and prediction...")
    subprocess.run(["python", "src/ingestion/fetch_and_predict_imap.py"], check=True)
    logging.info("Email fetch and prediction complete.")

@task
def retrain():
    logging.info("Starting model retraining...")
    subprocess.run(["python", "src/models/retrain.py"], check=True)
    logging.info("Model retraining complete.")

@flow(name="Email Phishing MLOps Pipeline")
def mlops_pipeline():
    fetch_and_predict()
    retrain()

if __name__ == "__main__":
    mlops_pipeline()
