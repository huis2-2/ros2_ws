#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source /home/tofu/.venvs/lane-onnx/bin/activate
exec python3 -u camera_lane_follow_dl_jetson_v3.py \
    --backend tensorrt --crop-top 0.0 --show --debug-every 2 "$@"
