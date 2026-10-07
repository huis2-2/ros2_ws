#!/usr/bin/env python3
"""바퀴를 띄운 상태에서만 사용하는 ROS 2 라이다 회피 시험 코드.

앞 40cm 이하 -> 넓은 쪽으로 조향 -> 반대 조향 -> 중앙으로 계속 전진.
장애물/측면/센서 오류에 의한 자동 정지 없음. 지면 주행에 사용하지 말 것.
센서가 끊겨도 시간에 따라 회피를 마치고 중앙 조향+전진을 유지한다.
좌1800/중앙1640/우1400us, 속도1565us, 속도CH8/조향CH9.
초기 ESC 안정화 및 Ctrl+C/정상 종료 시에만 중립1500us를 출력한다.
기본은 로그 시험. 실제 출력은 --drive.
"""
import argparse
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

I2C_BUS, PCA_ADDR, PCA_FREQ = 7, 0x40, 50
THROTTLE_CH, STEERING_CH = 8, 9
MUX_PIN, MUX_FREQ, MUX_DUTY = 15, 50, 10.0
NEUTRAL_US, DRIVE_SPEED_US, STEERING_CENTER_US = 1500, 1565, 1640

SCAN_TOPIC = '/scan'
FRONT_CENTER_DEG = 180.0
SCAN_TIMEOUT_SEC = 0.5
DISTANCE_LOG_INTERVAL_SEC = 0.5
RVIZ_CONFIG = (
    Path(__file__).resolve().parent
    / 'src'
    / 'sllidar_ros2'
    / 'rviz'
    / 'sllidar_ros2.rviz'
)

# LiDAR 원점 기준 거리
AVOID_DISTANCE_M = 0.40
CLEAR_DISTANCE_M = 0.50

MIN_TURN_SEC, MAX_TURN_SEC = 0.40, 0.80
CLEAR_SCANS = 3

MODE1, PRESCALE, LED0_ON_L = 0x00, 0xFE, 0x06


def init_pca9685(bus):
    prescale = int(round(25_000_000 / (4096 * PCA_FREQ) - 1))
    old = bus.read_byte_data(PCA_ADDR, MODE1)
    awake = old & ~0x10

    bus.write_byte_data(PCA_ADDR, MODE1, (awake & 0x7F) | 0x10)
    bus.write_byte_data(PCA_ADDR, PRESCALE, prescale)
    bus.write_byte_data(PCA_ADDR, MODE1, awake)
    time.sleep(0.005)
    bus.write_byte_data(PCA_ADDR, MODE1, awake | 0xA1)
    time.sleep(0.01)


def set_pwm_us(bus, channel, pulse_us):
    ticks = max(
        0,
        min(4095, int(pulse_us * PCA_FREQ * 4096 / 1_000_000))
    )
    reg = LED0_ON_L + 4 * channel

    for offset, value in enumerate(
        (0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F)
    ):
        bus.write_byte_data(PCA_ADDR, reg + offset, value)


class DriveHardware:
    def __init__(self):
        self.bus = self.mux_pwm = self.gpio = None
        self.current_speed_us = self.current_steering_us = None

        try:
            from smbus2 import SMBus
            import Jetson.GPIO as GPIO

            self.bus, self.gpio = SMBus(I2C_BUS), GPIO
            init_pca9685(self.bus)

            self.stop()
            self.set_steering(STEERING_CENTER_US)

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(MUX_PIN, GPIO.OUT)

            self.mux_pwm = GPIO.PWM(MUX_PIN, MUX_FREQ)
            self.mux_pwm.start(MUX_DUTY)

        except Exception:
            self.close()
            raise

    def set_speed(self, us):
        if self.bus is not None and us != self.current_speed_us:
            set_pwm_us(self.bus, THROTTLE_CH, us)
            self.current_speed_us = us

    def set_steering(self, us):
        if self.bus is not None and us != self.current_steering_us:
            set_pwm_us(self.bus, STEERING_CH, us)
            self.current_steering_us = us

    def stop(self):
        self.set_speed(NEUTRAL_US)

    def close(self):
        actions = []

        if self.bus is not None:
            actions += [
                lambda: set_pwm_us(
                    self.bus, THROTTLE_CH, NEUTRAL_US
                ),
                lambda: set_pwm_us(
                    self.bus, STEERING_CH, STEERING_CENTER_US
                ),
                lambda: time.sleep(0.2),
            ]

        if self.mux_pwm is not None:
            actions.append(self.mux_pwm.stop)

        if self.gpio is not None:
            actions.append(self.gpio.cleanup)

        if self.bus is not None:
            actions.append(self.bus.close)

        for action in actions:
            try:
                action()
            except Exception as exc:
                print(f'하드웨어 정리 실패: {exc}')

        self.bus = self.mux_pwm = self.gpio = None


def sector_distance(scan, low_deg, high_deg):
    """차량 앞 기준 왼쪽이 양수. 구간의 80% 이상 유효해야 사용."""
    if (
        not math.isfinite(scan.angle_min)
        or not math.isfinite(scan.angle_increment)
        or scan.angle_increment == 0
        or not math.isfinite(scan.range_min)
        or scan.range_min < 0
        or not math.isfinite(scan.range_max)
        or scan.range_max < CLEAR_DISTANCE_M
        or scan.range_min >= scan.range_max
    ):
        return None

    total, valid, nearest = 0, 0, math.inf
    span = abs(math.degrees(scan.angle_increment))
    expected = (high_deg - low_deg) / span

    for i, distance in enumerate(scan.ranges):
        angle = (
            scan.angle_min
            + i * scan.angle_increment
            - math.radians(FRONT_CENTER_DEG)
        )
        relative = math.degrees(
            math.atan2(math.sin(angle), math.cos(angle))
        )

        if not low_deg - 1e-6 <= relative <= high_deg + 1e-6:
            continue

        total += 1

        if distance == math.inf:
            valid += 1

        elif (
            math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max
        ):
            valid += 1
            nearest = min(nearest, distance)

        elif math.isfinite(distance) and 0 < distance < scan.range_min:
            valid += 1
            nearest = 0.0

    if total < 3 or total < 0.8 * expected or valid < 0.8 * total:
        return None

    return nearest


@dataclass
class Distances:
    front: float
    left: float
    right: float


def format_distance(distance):
    """Format one LiDAR distance for the terminal."""
    if distance is None:
        return '없음'
    if math.isinf(distance):
        return '감지 없음'
    return f'{distance:.2f} m'


def start_rviz():
    """Start RViz with the workspace LiDAR display configuration."""
    if not os.environ.get('DISPLAY'):
        print('RViz 시작 건너뜀: DISPLAY 환경 변수가 없습니다')
        return None
    if not RVIZ_CONFIG.is_file():
        print(f'RViz 설정 파일을 찾을 수 없습니다: {RVIZ_CONFIG}')
        return None

    try:
        process = subprocess.Popen(
            ['rviz2', '-d', str(RVIZ_CONFIG)],
            start_new_session=True,
        )
    except (FileNotFoundError, OSError) as exc:
        print(f'RViz 시작 실패: {exc}')
        return None

    print(f'RViz 시작: {RVIZ_CONFIG}')
    return process


def stop_rviz(process):
    """Stop an RViz process started by this program."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


class AvoidController:
    """공중 시험용: 회피 중에도 전진. 같은 장애물로 연속 반복하지 않음."""

    def __init__(self):
        self.state = 'CRUISE'
        self.direction = None
        self.phase_time = 0.0
        self.turn_duration = 0.0
        self.clear_count = 0
        self.armed = True
        self.reason = ''

    def update(self, d, now):
        self.reason = (
            '' if d is not None
            else '거리 데이터 없음: 공중 시험 전진 유지'
        )

        if self.state == 'CRUISE':
            if d is None:
                self.clear_count = 0
                return 'STRAIGHT'

            if not self.armed:
                self.clear_count = (
                    self.clear_count + 1
                    if d.front > CLEAR_DISTANCE_M
                    else 0
                )

                if self.clear_count >= CLEAR_SCANS:
                    self.armed = True
                    self.clear_count = 0

                return 'STRAIGHT'

            if d.front > AVOID_DISTANCE_M + 1e-6:
                return 'STRAIGHT'

            self.direction = (
                'LEFT' if d.left >= d.right else 'RIGHT'
            )
            self.state, self.phase_time = 'TURN_OUT', now
            self.armed, self.clear_count = False, 0
            return self.direction

        if self.state == 'TURN_OUT':
            elapsed = now - self.phase_time

            self.clear_count = (
                self.clear_count + 1
                if d is not None and d.front > CLEAR_DISTANCE_M
                else 0
            )

            if (
                elapsed >= MAX_TURN_SEC
                or (
                    elapsed >= MIN_TURN_SEC
                    and self.clear_count >= CLEAR_SCANS
                )
            ):
                self.turn_duration = min(elapsed, MAX_TURN_SEC)
                self.state, self.phase_time = 'ALIGN', now
                self.clear_count = 0
                return self.opposite()

            return self.direction

        if self.state == 'ALIGN':
            if now - self.phase_time >= self.turn_duration:
                self.state = 'CRUISE'
                self.clear_count = 0
                return 'STRAIGHT'

            return self.opposite()

        self.state = 'CRUISE'
        return 'STRAIGHT'

    def opposite(self):
        return 'RIGHT' if self.direction == 'LEFT' else 'LEFT'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--drive',
        action='store_true',
        help='실제 PWM 출력',
    )
    parser.add_argument(
        '--left-us',
        type=int,
        default=1800,
        help='좌회전 펄스(us), 기본1800',
    )
    parser.add_argument(
        '--right-us',
        type=int,
        default=1400,
        help='우회전 펄스(us), 기본1400',
    )
    parser.add_argument(
        '--no-rviz',
        action='store_true',
        help='RViz를 자동으로 실행하지 않음',
    )
    opts, ros_args = parser.parse_known_args()

    if opts.drive:
        if opts.left_us is None or opts.right_us is None:
            parser.error(
                '--drive에는 --left-us와 --right-us가 필요합니다'
            )

        if (
            not 1000 <= opts.left_us <= 2000
            or not 1000 <= opts.right_us <= 2000
            or (
                (opts.left_us - STEERING_CENTER_US)
                * (opts.right_us - STEERING_CENTER_US)
                >= 0
            )
        ):
            parser.error(
                '좌우는 1000~2000us 안에서 '
                '중앙1640us의 서로 반대편 값이어야 합니다'
            )

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan

    class LidarAvoidDrive(Node):
        def __init__(self, hardware):
            super().__init__('lidar_avoid_drive')

            self.hardware = hardware
            self.control = AvoidController()
            self.last_scan_time = None
            self.last_status = None
            self.last_log_time = 0.0

            self.sub = self.create_subscription(
                LaserScan,
                SCAN_TOPIC,
                self.scan_callback,
                qos_profile_sensor_data,
            )
            self.timer = self.create_timer(0.05, self.watchdog)

            self.get_logger().warning(
                '공중 시험용: 자동 정지 없음. 지면 주행 금지'
            )
            self.get_logger().info(
                '실제 PWM 출력'
                if hardware
                else '판단 로그 시험: PWM 출력 없음'
            )

        def apply(self, command):
            if self.hardware is None:
                return

            steering = {
                'LEFT': opts.left_us,
                'RIGHT': opts.right_us,
            }.get(command, STEERING_CENTER_US)

            self.hardware.set_steering(steering)
            self.hardware.set_speed(DRIVE_SPEED_US)

        def report(self, command, d=None):
            now = time.monotonic()
            status = (
                self.control.state,
                command,
                self.control.reason,
            )

            if (
                status != self.last_status
                or now - self.last_log_time >= DISTANCE_LOG_INTERVAL_SEC
            ):
                front = format_distance(None if d is None else d.front)
                left = format_distance(None if d is None else d.left)
                right = format_distance(None if d is None else d.right)
                reason = (
                    ''
                    if not self.control.reason
                    else f' | {self.control.reason}'
                )

                self.get_logger().info(
                    f'거리: 전방={front}, 왼쪽={left}, 오른쪽={right}'
                    f' | 상태={status[0]}/{command}{reason}'
                )
                self.last_status = status
                self.last_log_time = now

        def scan_callback(self, scan):
            now = time.monotonic()

            stamp = (
                scan.header.stamp.sec
                + scan.header.stamp.nanosec / 1e9
            )
            ros_now = self.get_clock().now().nanoseconds / 1e9
            age = ros_now - stamp

            if (
                stamp <= 0
                or age > SCAN_TIMEOUT_SEC
                or age < -0.1
            ):
                command = self.control.update(None, now)
                self.apply(command)
                self.report(command)
                return

            self.last_scan_time = now

            values = [
                sector_distance(scan, -45, 45),
                sector_distance(scan, 45, 100),
                sector_distance(scan, -100, -45),
            ]

            d = (
                None
                if any(x is None for x in values)
                else Distances(*values)
            )

            command = self.control.update(d, now)
            self.apply(command)
            self.report(command, d)

        def watchdog(self):
            now = time.monotonic()

            if (
                self.last_scan_time is None
                or now - self.last_scan_time > SCAN_TIMEOUT_SEC
            ):
                command = self.control.update(None, now)
                self.apply(command)
                self.report(command)

    hardware = node = rviz_process = None
    initialized = False

    try:
        if opts.drive:
            hardware = DriveHardware()
            print('ESC 중립 안정화 3초')
            time.sleep(3.0)

        rclpy.init(args=ros_args)
        initialized = True
        node = LidarAvoidDrive(hardware)
        if not opts.no_rviz:
            rviz_process = start_rviz()
        rclpy.spin(node)

    except KeyboardInterrupt:
        print('정지 요청')

    except Exception as exc:
        print(f'실행 오류: {exc}')
        return 1

    finally:
        stop_rviz(rviz_process)

        if hardware is not None:
            hardware.close()

        if node is not None:
            node.destroy_node()

        if initialized and rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
