#!/usr/bin/env bash

# Source this file after connecting over SSH:
#   source ~/cuda-lab/scripts/setup_cuda_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced so its environment persists:" >&2
    echo "  source $0" >&2
    exit 1
fi

export CUDA_LAB_ROOT
CUDA_LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
# TORCH_CUDA_ARCH_LIST is deliberately not set: PyTorch then detects the real
# compute capability. Pinning it (this used to say 8.6) builds a cubin for the
# wrong architecture, and without a "+PTX" suffix it will not load at all.

if [[ ! -d "$CUDA_HOME" ]]; then
    echo "CUDA toolkit not found at $CUDA_HOME" >&2
    return 1
fi

case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
    *":$CUDA_HOME/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

for command in nvidia-smi nvcc ncu compute-sanitizer uv; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command is missing: $command" >&2
        return 1
    fi
done

cd "$CUDA_LAB_ROOT" || return 1
uv sync --locked || return 1

# The NVIDIA driver on this host restricts performance counters to privileged
# processes, so only the Nsight commands need sudo. `cuda-lab bench` does not.
cuda_lab_profile() {
    local kernel="${1:-01_vecadd}"
    local status
    shift 2>/dev/null || true

    sudo -E env \
        HOME="$HOME" \
        PATH="$CUDA_LAB_ROOT/.venv/bin:$PATH" \
        CUDA_HOME="$CUDA_HOME" \
        CUDA_PATH="$CUDA_PATH" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$CUDA_LAB_ROOT/.venv/bin/cuda-lab" profile "$kernel" "$@"
    status=$?
    sudo chown -R "$(id -u):$(id -g)" "$CUDA_LAB_ROOT/results"
    return "$status"
}

echo "CUDA lab ready: $CUDA_LAB_ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader
echo "nvcc: $(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p')"
echo "ncu:  $(ncu --version | tail -n 1)"
echo "uv:   $(uv --version)"
python -c "import torch; print('arch:', 'sm_%d%d' % torch.cuda.get_device_capability())" 2>/dev/null
echo
echo "  uv run cuda-lab test  02_matmul          # correctness"
echo "  uv run cuda-lab bench 02_matmul --plot   # latency, seconds"
echo "  cuda_lab_profile      02_matmul          # Nsight counters, minutes"
