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

source /opt/ros/jazzy/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"
set -u

lidar_pid=""
avoidance_pid=""
cleanup_started=0

cleanup() {
    if (( cleanup_started )); then
        return
    fi
    cleanup_started=1

    echo
    echo "LiDAR, RViz, 장애물 회피 코드를 종료합니다..."

    if [[ -n "${avoidance_pid}" ]] && kill -0 "${avoidance_pid}" 2>/dev/null; then
        kill -INT "${avoidance_pid}" 2>/dev/null || true
    fi
    if [[ -n "${lidar_pid}" ]] && kill -0 "${lidar_pid}" 2>/dev/null; then
        kill -INT "${lidar_pid}" 2>/dev/null || true
    fi

    [[ -z "${avoidance_pid}" ]] || wait "${avoidance_pid}" 2>/dev/null || true
    [[ -z "${lidar_pid}" ]] || wait "${lidar_pid}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "LiDAR 연결: ${LIDAR_PORT}"
echo "RViz와 Slamtec C1 노드를 시작합니다."
ros2 launch sllidar_ros2 view_sllidar_c1_launch.py \
    serial_port:="${LIDAR_PORT}" &
lidar_pid=$!

echo "LiDAR 장애물 회피 코드를 시작합니다."
"${WORKSPACE_DIR}/lidar_obstacle_avoidance.py" &
avoidance_pid=$!

wait -n "${lidar_pid}" "${avoidance_pid}"
