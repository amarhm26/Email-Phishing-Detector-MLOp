@echo off
echo ========================================
echo      LOCAL DEPLOYMENT STARTED
echo ========================================

echo.
echo Fetching latest artifacts from GitHub...

gh run download --repo YOUR_GITHUB_USERNAME/MLOps --name model-artifacts --dir artifacts

echo.
echo Starting FastAPI server...
start cmd /k "uvicorn src.serving.api:app --host 0.0.0.0 --port 8000"

echo.
echo Starting Streamlit dashboard...
start cmd /k "streamlit run streamlit_app.py"

echo.
echo Deployment complete!
pause
