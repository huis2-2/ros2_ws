#!/usr/bin/env bash

set -o pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIDAR_BY_ID="/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_12f6e4544f6eef1189d8e7c2c169b110-if00-port0"
LIDAR_PORT="${LIDAR_PORT:-${LIDAR_BY_ID}}"

if [[ ! -e "${LIDAR_PORT}" ]]; then
    LIDAR_PORT="/dev/ttyUSB0"
fi

if [[ ! -e "${LIDAR_PORT}" ]]; then
    echo "오류: LiDAR 장치를 찾을 수 없습니다."
    echo "확인한 경로: ${LIDAR_BY_ID}, /dev/ttyUSB0"
    exit 1
fi

if [[ ! -e /dev/video0 ]]; then
    echo "오류: 카메라 /dev/video0을 찾을 수 없습니다."
    exit 1
fi

source /opt/ros/jazzy/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"
set -u

lidar_pid=""
camera_pid=""
follower_pid=""
cleanup_started=0

cleanup() {
    if (( cleanup_started )); then
        return
    fi
    cleanup_started=1

    echo
    echo "카메라, LiDAR, 통합 주행 코드를 종료합니다..."

    for pid in "${follower_pid}" "${camera_pid}" "${lidar_pid}"; do
        if [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        fi
    done

    for _ in {1..30}; do
        running=0
        for pid in "${follower_pid}" "${camera_pid}" "${lidar_pid}"; do
            if [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
                running=1
            fi
        done
        (( running )) || break
        sleep 0.1
    done

    for pid in "${follower_pid}" "${camera_pid}" "${lidar_pid}"; do
        if [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
            kill -KILL -- "-${pid}" 2>/dev/null || true
        fi
    done

    for pid in "${follower_pid}" "${camera_pid}" "${lidar_pid}"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
}

trap cleanup EXIT INT TERM

echo "LiDAR 연결: ${LIDAR_PORT}"
echo "RViz에서 /scan을 표시합니다."
setsid ros2 launch sllidar_ros2 view_sllidar_c1_launch.py \
    serial_port:="${LIDAR_PORT}" &
lidar_pid=$!

setsid ros2 run camera_node camera_node &
camera_pid=$!

echo "카메라+LiDAR 통합 노드를 시작합니다."
setsid "${WORKSPACE_DIR}/camera_lidar_follow.py" "$@" &
follower_pid=$!

wait -n "${lidar_pid}" "${camera_pid}" "${follower_pid}"
