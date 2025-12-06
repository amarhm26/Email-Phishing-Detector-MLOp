import logging
from logging.handlers import RotatingFileHandler
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def _make_logger(name, filename, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    path = os.path.join(LOG_DIR, filename)
    handler = RotatingFileHandler(path, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    # Also add a console handler for convenience
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

training_logger = _make_logger("training", "training.log")
prediction_logger = _make_logger("prediction", "predictions.log")
drift_logger = _make_logger("drift", "drift.log")
