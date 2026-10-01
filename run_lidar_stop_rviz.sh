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

# ROS setup 스크립트는 값이 없을 수 있는 환경 변수를 참조하므로,
# 환경을 모두 불러온 뒤 미정의 변수 검사를 활성화한다.
set -u

lidar_pid=""
drive_pid=""
cleanup_started=0

cleanup() {
    if (( cleanup_started )); then
        return
    fi
    cleanup_started=1

    echo
    echo "LiDAR, RViz, 주행 코드를 종료합니다..."

    if [[ -n "${drive_pid}" ]] && kill -0 "${drive_pid}" 2>/dev/null; then
        kill -INT "${drive_pid}" 2>/dev/null || true
    fi
    if [[ -n "${lidar_pid}" ]] && kill -0 "${lidar_pid}" 2>/dev/null; then
        kill -INT "${lidar_pid}" 2>/dev/null || true
    fi

    [[ -z "${drive_pid}" ]] || wait "${drive_pid}" 2>/dev/null || true
    [[ -z "${lidar_pid}" ]] || wait "${lidar_pid}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "LiDAR 연결: ${LIDAR_PORT}"
echo "RViz와 Slamtec C1 노드를 시작합니다."
ros2 launch sllidar_ros2 view_sllidar_c1_launch.py \
    serial_port:="${LIDAR_PORT}" &
lidar_pid=$!

echo "장애물 정지 주행 코드를 시작합니다."
"${WORKSPACE_DIR}/lidar_stop_drive.py" &
drive_pid=$!

# 둘 중 하나라도 끝나면 다른 프로세스도 cleanup에서 종료한다.
wait -n "${lidar_pid}" "${drive_pid}"
