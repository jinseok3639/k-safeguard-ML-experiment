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
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"
if [ -f "$KSLAB_ROOT/.venv/Scripts/activate" ]; then
    . "$KSLAB_ROOT/.venv/Scripts/activate"
elif [ -f "$KSLAB_ROOT/.venv/bin/activate" ]; then
    . "$KSLAB_ROOT/.venv/bin/activate"
else
    echo "가상환경이 없습니다. 먼저 'python -m venv .venv' 를 실행하세요." >&2
    return 1
fi
echo "kslab environment activated: $(python --version) @ $KSLAB_ROOT"
