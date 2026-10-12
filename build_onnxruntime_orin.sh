#!/usr/bin/env bash
# Build for the Orin SM87 GPU, CUDA 13.2 and TensorRT 10.16 on this Jetson.
set -euo pipefail

ORT_SOURCE=/home/tofu/.cache/onnxruntime-orin-v1.23.0
ORT_ENV=/home/tofu/.venvs/lane-onnx
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
swap_kb=$(awk '/SwapFree:/ {print $2}' /proc/meminfo)
if (( available_kb + swap_kb < 12000000 )); then
    echo 'Build requires at least 12 GB available RAM + swap. Enable 16 GB swap first.' >&2
    exit 1
fi
test -f "$ORT_SOURCE/tools/ci_build/build.py"
test -x "$ORT_ENV/bin/python"
export PATH="$ORT_ENV/bin:/usr/local/cuda/bin:$PATH"
cd "$ORT_SOURCE"
python tools/ci_build/build.py \
    --build_dir "$ORT_SOURCE/build/Linux" \
    --config Release --update --build --parallel 3 --nvcc_threads 1 \
    --build_wheel --skip_tests --use_tensorrt \
    --disable_ml_ops --disable_cuda_nhwc_ops \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF \
        onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
    --compile_no_warning_as_error \
    2>&1 | tee "$ORT_SOURCE/build-orin.log"

shopt -s nullglob
wheels=("$ORT_SOURCE"/build/Linux/Release/dist/onnxruntime_gpu-*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo 'Expected exactly one built GPU wheel.' >&2
    exit 1
fi
python -m pip install --force-reinstall --no-deps "${wheels[0]}"
cd "$PROJECT_DIR"
python -u - <<'PY'
import time
import numpy as np
import onnxruntime as ort
from camera_lane_follow_dl_jetson_v3 import Segmenter

print('ORT:', ort.__version__, 'providers:', ort.get_available_providers())
frame = np.zeros((480, 640, 3), dtype=np.uint8)
for backend in ('cuda', 'tensorrt'):
    segmenter = Segmenter('lane_model.onnx', backend=backend, crop_top=0.0)
    elapsed = []
    for i in range(12):
        start = time.perf_counter()
        white, yellow = segmenter.predict(frame)
        assert white.shape == yellow.shape == frame.shape[:2]
        if i >= 2:
            elapsed.append((time.perf_counter() - start) * 1000)
    print(backend, 'predict median/max ms:', np.median(elapsed), max(elapsed), flush=True)
    del segmenter
print('GPU MODEL TEST PASSED. Motor outputs were not enabled.')
PY
