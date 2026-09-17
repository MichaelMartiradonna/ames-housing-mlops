$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the Python environment and install requirements as described in README.md.'
}
Set-Location -LiteralPath $projectRoot
& $pythonPath -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
