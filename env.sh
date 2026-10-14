# env.ps1 의 bash 판. 같은 경로 규칙을 쓴다.
#   source ./env.sh
KSLAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KSLAB_ROOT
export HF_HOME="$KSLAB_ROOT/data/hf-home"
export HF_HUB_CACHE="$KSLAB_ROOT/data/hf-home/hub"
export HF_DATASETS_CACHE="$KSLAB_ROOT/data/hf-home/datasets"
export TORCH_HOME="$KSLAB_ROOT/data/torch-home"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONHASHSEED=0
# MLflow 기록도 폴더 안에만. artifact 위치는 kslab/tracking.py 가 data/mlflow/artifacts 로 잡는다.
export MLFLOW_TRACKING_URI="sqlite:///$KSLAB_ROOT/data/mlflow/mlflow.db"
# `mlflow ui` 서버는 TRACKING_URI 가 아니라 이 값을 본다. 없으면 cwd에 빈 mlflow.db 를 새로 만든다.
export MLFLOW_BACKEND_STORE_URI="$MLFLOW_TRACKING_URI"
export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$KSLAB_ROOT/data/mlflow"
if [ -f "$KSLAB_ROOT/.venv/Scripts/activate" ]; then
    . "$KSLAB_ROOT/.venv/Scripts/activate"
elif [ -f "$KSLAB_ROOT/.venv/bin/activate" ]; then
    . "$KSLAB_ROOT/.venv/bin/activate"
else
    echo "가상환경이 없습니다. 먼저 'python -m venv .venv' 를 실행하세요." >&2
    return 1
fi
echo "kslab environment activated: $(python --version) @ $KSLAB_ROOT"
