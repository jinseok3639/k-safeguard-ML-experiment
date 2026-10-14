# ML 샌드박스 환경 활성화.
# venv·HF 캐시·torch 캐시를 전부 이 폴더 안으로 고정한다 — 기본값을 쓰면 사용자 홈에
# 수 GB가 조용히 쌓이는데, 이 실험실은 폴더 밖에 아무것도 남기지 않는 것이 규칙이다.
#
# 사용법:  . .\env.ps1

$Root = $PSScriptRoot
$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $Activate)) {
    throw "가상환경이 없습니다. 먼저 'python -m venv .venv' 를 실행하세요."
}

$env:KSLAB_ROOT         = $Root
$env:HF_HOME            = Join-Path $Root "data\hf-home"
$env:HF_HUB_CACHE       = Join-Path $Root "data\hf-home\hub"
$env:HF_DATASETS_CACHE  = Join-Path $Root "data\hf-home\datasets"
$env:TORCH_HOME         = Join-Path $Root "data\torch-home"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONHASHSEED     = "0"
# MLflow 기록도 폴더 안에만. artifact 위치는 kslab/tracking.py 가 data\mlflow\artifacts 로 잡는다.
$MlflowDir = Join-Path $Root "data\mlflow"
$env:MLFLOW_TRACKING_URI = "sqlite:///" + ((Join-Path $MlflowDir "mlflow.db") -replace "\\", "/")
# `mlflow ui` 서버는 TRACKING_URI 가 아니라 이 값을 본다. 없으면 cwd에 빈 mlflow.db 를 새로 만든다.
$env:MLFLOW_BACKEND_STORE_URI = $env:MLFLOW_TRACKING_URI
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

foreach ($d in @($env:HF_HOME, $env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE, $env:TORCH_HOME, $MlflowDir)) {
    if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

. $Activate

Write-Output "kslab environment activated"
Write-Output "Root:     $Root"
Write-Output "Python:   $((Get-Command python).Source)"
Write-Output "HF_HOME:  $env:HF_HOME"
