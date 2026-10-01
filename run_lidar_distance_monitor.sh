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
monitor_pid=""
cleanup_started=0

cleanup() {
    if (( cleanup_started )); then
        return
    fi
    cleanup_started=1

    echo
    echo "LiDAR 거리 모니터를 종료합니다..."

    if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
        kill -INT "${monitor_pid}" 2>/dev/null || true
    fi
    if [[ -n "${lidar_pid}" ]] && kill -0 "${lidar_pid}" 2>/dev/null; then
        kill -INT "${lidar_pid}" 2>/dev/null || true
    fi

    [[ -z "${monitor_pid}" ]] || wait "${monitor_pid}" 2>/dev/null || true
    [[ -z "${lidar_pid}" ]] || wait "${lidar_pid}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "LiDAR 연결: ${LIDAR_PORT}"
echo "주행 없이 정면 장애물 거리를 1초마다 출력합니다."

ros2 launch sllidar_ros2 sllidar_c1_launch.py \
    serial_port:="${LIDAR_PORT}" &
lidar_pid=$!

"${WORKSPACE_DIR}/lidar_distance_monitor.py" &
monitor_pid=$!

wait -n "${lidar_pid}" "${monitor_pid}"
