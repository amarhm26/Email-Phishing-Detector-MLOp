# Email-Phishing-Detector-MLOp
A fully automated machine learning system with ingestion, training, deployment, monitoring, and retraining.

 Project Overview

This project implements a complete MLOps pipeline for detecting phishing emails using machine learning.
The system continuously fetches emails from an IMAP inbox, preprocesses and transforms the text using TF-IDF, classifies emails using a Logistic Regression model, and serves predictions through a FastAPI backend and a Streamlit dashboard.

Beyond basic ML, the project includes:

Automated data ingestion

Preprocessing and feature engineering

Model training and evaluation

Drift detection

Automated retraining

Model registry and versioning

CI/CD through GitHub Actions

Dockerized deployment

Logging and monitoring

Streamlit UI for visualization

This makes it a full production-style MLOps project, not just an ML script.

Features
✔ 1. Automated Email Ingestion

Connects to IMAP server

Fetches latest emails

Extracts subject, sender, and body

Stores logs for every batch

2. Text Preprocessing

MIME decoding

HTML/tag removal

Stopword removal

Lowercasing

Signature/noise cleaning

Subject + body merged

3. Feature Engineering

TF-IDF vectorization

Bi-grams

max_features = 5000

Saved vectorizer for reuse

4. Machine Learning Model

Logistic Regression (Scikit-learn)

High accuracy on phishing classification

Balanced precision/recall

Model stored as model.pkl

5. Drift Detection

Compares baseline vs current phishing rate

Detects significant distribution change

Logs drift alerts

Triggers retraining when threshold is exceeded

6. Automated Retraining Pipeline

Loads updated dataset

Re-trains model

Saves new version in model registry

Updates artifacts folder

7. Deployment

FastAPI (backend)

Streamlit dashboard (frontend)

Docker for full environment consistency

Works locally and is cloud-ready

8. CI/CD

GitHub Actions pipeline includes:

Linting

Build checks

Pipeline validation

Test execution

9. Model Registry

Stores versioned models

Enables rollback

Tracks improvement history

10. Logging

Email ingestion logs

Prediction logs

Drift logs

Training logs

Daily rotation

Project Structure
Mlops/
│
├── artifacts/                # Saved model, vectorizer, metrics
├── data/                     # Raw and processed datasets
├── logs/                     # Logs for predictions, drift, training
├── model_registry/           # Versioned model storage
├── notebooks/                # Jupyter experiments
├── pages/                    # Streamlit dashboard pages
├── docker/                   # Docker config files
│
├── src/
│   ├── features/             # TF-IDF vectorizer code
│   ├── ingestion/            # IMAP email ingestion
│   ├── monitoring/           # Drift detection module
│   ├── models/               # Training + retraining scripts
│   ├── preprocessing/        # Text cleaning utilities
│   ├── serving/              # FastAPI serving code
│   ├── utils/                # Logging + helpers
│   └── workflows/            # End-to-end orchestration
│
├── launch_full_model.bat     # Runs full workflow
├── deploy_local.bat          # Local deployment script
├── run_all.bat               # Combined pipeline execution
│
├── Dockerfile                # Docker build file
├── requirements.txt          # Python dependencies
└── README.md

Installation & Setup
1. Clone the repository
git clone https://github.com/yourusername/mlops-email-phishing.git
cd mlops-email-phishing

2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Setup environment variables

Create a .env file:

EMAIL_USER = your_email
EMAIL_PASS = your_password
IMAP_HOST  = imap.gmail.com
MAILBOX    = INBOX

Running the Pipeline
1. Fetch and Predict Emails
python src/ingestion/imap_fetch_predict.py

2. Train Model
python src/models/train.py

3. Retrain Model (Manual Trigger)
python src/models/retrain.py

4. Run Full Workflow
./run_all.bat

5. Local Deployment
./deploy_local.bat

Serving the Model
FastAPI Server
uvicorn src.serving.api:app --reload


Endpoint:

POST /predict

Streamlit Dashboard
streamlit run app.py

Drift Monitoring

The system logs:

baseline phishing rate

current phishing rate

drift magnitude

alerts when drift > threshold

Logs stored in:

logs/drift/


When drift is detected → retraining pipeline is triggered.

Docker Deployment
Build image:
docker build -t phishing-mlops .

Run container:
docker run -p 8000:8000 phishing-mlops

CI/CD Pipeline
GitHub Actions includes

Code checks

Dependency validation

Linting

Automated testing

Build pipeline

Ensures every commit is production-ready.

Model Performance

High accuracy

Strong precision/recall

Effective TF-IDF features for email text

Reliable predictions in real-time

Evaluation reports stored under:

artifacts/metrics.json

Visualization Dashboard

Streamlit UI shows:

Prediction results

Logs and trends

Model comparison

Phishing vs legitimate distribution

Drift status

 Architecture Diagram (Text Version)
IMAP EMAILS → Preprocessing → TF-IDF → Model Prediction → Logs
                          ↓
                       Drift Monitor → Retraining → Model Registry
                          ↓
                      Deployment (API + Streamlit)
                          ↓
                     CI/CD + Dockerized Execution

Conclusion

This project delivers a complete, production-ready MLOps pipeline capable of automatically detecting phishing emails, monitoring drift, retraining models, and deploying services with reliability and scalability. It offers a strong foundation for real-world applications and advanced extensions using LSTMs or transformer models.

Author

Amar
B.Tech Student — ML/MLOps Developer
Passionate about building end-to-end intelligent systems.

If you want, I can also create:

Project Report (PDF)
Architecture diagram (image)
MLOps workflow figure
Abstract + Introduction + Literature Review
Viva questions & answers

Just tell me!

You said:
dont add icon or imgs 
ChatGPT said:

Understood, Amar.
Here is your clean, professional README with zero icons or images, formatted in a traditional and straightforward style.
You can copy–paste this directly into your GitHub repository.

README
End-to-End MLOps Pipeline for Email Phishing Detection

This project implements a complete MLOps workflow for detecting phishing emails using machine learning. The system is designed to automate the entire lifecycle of the model, from data ingestion and preprocessing to deployment, monitoring, drift detection, and automated retraining. The solution is built to reflect real-world MLOps practices and includes modular components, versioning, CI/CD, and a dashboard for monitoring system performance.

Project Overview

The goal of this project is to classify emails as phishing or legitimate. The system continuously fetches emails from an IMAP inbox, processes the text, applies TF-IDF feature engineering, and classifies them using a Logistic Regression model. The entire pipeline is automated and supports deployment through FastAPI and Streamlit.

This project contains all major components expected in a full MLOps system: ingestion, preprocessing, feature engineering, training, monitoring, logging, versioning, and automated retraining.

Features

Automated Email Ingestion

Text Preprocessing

TF-IDF Feature Engineering

Logistic Regression Model

Drift Detection

Automated Retraining

FastAPI Model Serving

Streamlit Dashboard

Docker Deployment

CI/CD with GitHub Actions

Model Registry and Versioning

Logging and Monitoring

Project Structure
Mlops/
|
├── artifacts/                Saved model, vectorizer, metrics
├── data/                     Raw and processed datasets
├── logs/                     Training, prediction, drift logs
├── model_registry/           Versioned models
├── notebooks/                Experiments
├── pages/                    Streamlit pages
├── docker/                   Docker configuration
|
├── src/
│   ├── features/             TF-IDF vectorizer module
│   ├── ingestion/            IMAP email ingestion
│   ├── monitoring/           Drift detection code
│   ├── models/               Training and retraining scripts
│   ├── preprocessing/        Text cleaning functions
│   ├── serving/              FastAPI service
│   ├── utils/                Logging, helpers
│   └── workflows/            End-to-end orchestration
|
├── launch_full_model.bat     Full pipeline executor
├── deploy_local.bat          Local deployment script
├── run_all.bat               Combined workflow runner
|
├── Dockerfile                Docker build configuration
├── requirements.txt          Python dependencies
└── README.md

Setup Instructions
1. Clone the repository
git clone https://github.com/yourrepo/mlops-email-phishing.git
cd mlops-email-phishing

2. Create virtual environment
python -m venv venv
venv\Scripts\activate  (Windows)

3. Install dependencies
pip install -r requirements.txt

4. Set environment variables

Create a .env file with:

EMAIL_USER=your_email
EMAIL_PASS=your_password
IMAP_HOST=imap.gmail.com
MAILBOX=INBOX

Running the Pipeline
Fetch and Predict Emails
python src/ingestion/imap_fetch_predict.py

Train the Model
python src/models/train.py

Retrain the Model
python src/models/retrain.py

Execute Full Workflow
run_all.bat

Local Deployment
deploy_local.bat

Model Serving
FastAPI

Start the server:

uvicorn src.serving.api:app --reload


Endpoints:

POST /predict

GET /healthcheck

Streamlit Dashboard
streamlit run app.py

Drift Monitoring

The drift detection module compares the current phishing prediction rate with the baseline. If the difference exceeds a threshold, drift is logged and triggers retraining.

Logs are stored in:

logs/drift/

Docker Deployment
Build Docker Image
docker build -t phishing-mlops .

Run Container
docker run -p 8000:8000 phishing-mlops

Continuous Integration

The GitHub Actions pipeline performs:

Linting

Dependency checks

Basic tests

Build validation

This ensures every commit is stable and production-ready.

Model Performance

The model gives strong performance on the dataset, with stable accuracy, precision, recall, and F1-score. TF-IDF features capture important language characteristics used in phishing.

Evaluation reports are stored under:

artifacts/metrics.json

Architecture Summary (Text Only)

Data ingestion through IMAP
→ Preprocessing
→ TF-IDF vectorization
→ Logistic Regression prediction
→ Logging
→ Drift monitoring
→ Retraining
→ Model Registry
→ Deployment through API and Streamlit
→ CI/CD automation

Conclusion

This project demonstrates a fully functioning MLOps system for phishing detection. It automates data ingestion, preprocessing, model training, deployment, monitoring, drift detection, and retraining. The pipeline is structured for long-term reliability and can be easily scaled or transitioned to cloud environments. The work reflects a balanced combination of traditional machine learning and modern MLOps practices.

Author

Amar
B.Tech Student
Specialized in Machine Learning and MLOps
