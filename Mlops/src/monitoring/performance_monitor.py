import logging
import os
import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f"performance_{datetime.date.today()}.log")

logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_prediction(email_id, prediction, score):
    logging.info(f"EmailID={email_id} Prediction={prediction} Score={score}")

if __name__ == "__main__":
    # Example usage
    log_prediction("1234", "phishing", 0.85)
