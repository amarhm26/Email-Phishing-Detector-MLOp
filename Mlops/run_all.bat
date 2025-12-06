@echo off
echo Starting Full MLOps Pipeline...

REM Train / Retrain model
echo Training model...
python src/training/train.py

REM Start FastAPI in background
echo Starting FastAPI API...
start cmd /k "python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000"

REM Wait a moment
timeout /t 15

REM Start Streamlit app
echo Opening Streamlit...
start cmd /k "python -m streamlit run streamlit_app.py"

pause
