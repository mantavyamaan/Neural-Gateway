@echo off
echo Starting Neural Gateway Backend...
start cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"

echo Starting Neural Gateway Frontend...
start cmd /k ".venv\Scripts\streamlit run frontend.py --server.port 8505"

echo Both services have been started in separate windows!
