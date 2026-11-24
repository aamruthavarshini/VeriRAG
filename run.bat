
@echo off

title VeriRAG - AI Answer Verification

cd /d "%~dp0"

if not exist .venv (
    py -3.11 -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

echo.
echo Starting VeriRAG...
echo.

streamlit run app.py

pause
