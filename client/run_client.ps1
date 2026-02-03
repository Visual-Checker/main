# 간단 실행 스크립트 (PowerShell)
# 사용: Open PowerShell, then: .\run_client.ps1

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python client_ui.py
