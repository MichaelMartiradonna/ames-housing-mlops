$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$runtimePath = Join-Path $projectRoot '.tools\ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $runtimePath)) {
    throw 'Run python scripts/setup_local_llm.py first, or install Ollama from ollama.com.'
}
$env:OLLAMA_MODELS = Join-Path $projectRoot '.tools\models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$logDir = Join-Path $projectRoot 'artifacts'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Start-Process -FilePath $runtimePath -ArgumentList 'serve' -WindowStyle Hidden -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $logDir 'ollama-output.log') -RedirectStandardError (Join-Path $logDir 'ollama-error.log')
Write-Output 'Local Ollama server started on 127.0.0.1:11434.'
