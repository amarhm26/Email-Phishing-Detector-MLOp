import os
import sys
import imaplib
import email
from email.header import decode_header
import joblib
from dotenv import load_dotenv
from datetime import datetime

# PATH fix
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.preprocess import preprocess_text
from src.utils.logger import prediction_logger, drift_logger
from src.monitoring.drift import monitor_drift   # we'll create this next

# load env
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
MAILBOX = os.getenv("MAILBOX", "INBOX")
MARK_AS_SEEN = False

# model paths
MODEL_PATH = os.path.join(ROOT, "artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join(ROOT, "artifacts", "vectorizer.pkl")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def decode_mime_words(s):
    parts = decode_header(s)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="ignore"))
        else:
            decoded.append(part)
    return "".join(decoded)


def get_first_text_part(msg):
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if ctype == "text/plain" and "attachment" not in disp:
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    return payload.decode(charset, errors="ignore")
        return ""
    else:
        payload = msg.get_payload(decode=True)
        if not payload:
            return ""
        return payload.decode(msg.get_content_charset() or "utf-8", errors="ignore")


def predict_text(subject, body):
    raw_text = f"{subject} {body}"
    cleaned = preprocess_text(raw_text)
    X = vectorizer.transform([cleaned])
    score = model.predict_proba(X)[:, 1][0]
    label = "phishing" if score > 0.495 else "legitimate"
    return label, float(score)


def fetch_and_predict():
    prediction_logger.info("Starting IMAP fetch_and_predict run")
    m = imaplib.IMAP4_SSL(IMAP_HOST)
    m.login(EMAIL_USER, EMAIL_PASS)
    m.select(MAILBOX)

    status, message_nums = m.search(None, "ALL")
    ids = message_nums[0].split()[-100:]  # last 100 emails

    results = []
    for num in ids:
        status, msg_data = m.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        subject = decode_mime_words(msg.get("Subject", "") or "")
        sender = decode_mime_words(msg.get("From", "") or "")
        body = get_first_text_part(msg) or ""

        label, score = predict_text(subject, body)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "id": num.decode() if isinstance(num, bytes) else str(num),
            "from": sender,
            "subject": subject,
            "label": label,
            "score": round(score, 4)
        }
        results.append(entry)

        # log each prediction line
        prediction_logger.info(f"{entry['id']} | {entry['from']} | {entry['label']} | {entry['score']} | {entry['subject'][:120]}")

        if MARK_AS_SEEN:
            m.store(num, "+FLAGS", "\\Seen")

    m.close()
    m.logout()

    # Run drift monitor with the latest batch of predictions
    try:
        drift_alert = monitor_drift(results)
        if drift_alert:
            drift_logger.warning(f"Drift detected: {drift_alert}")
        else:
            drift_logger.info("No drift detected.")
    except Exception as e:
        drift_logger.error(f"Drift monitor failed: {e}")

    prediction_logger.info("Completed IMAP fetch_and_predict run")
    return results


def health_check():
    try:
        _ = model
        _ = vectorizer
        return True
    except Exception:
        return False


if __name__ == "__main__":
    res = fetch_and_predict()
    for r in res:
        print(r)
