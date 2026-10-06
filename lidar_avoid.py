#!/usr/bin/env python3
"""ROS 2 /scan 기반 저속 RC카 회피 시험 코드.

기존 설정: I2C7, PCA9685 0x40/50Hz, 속도 CH8, 조향 CH9,
ESC 중립1500/전진1565us, 조향 중앙1640us, MUX BOARD15.
기본 실행은 판단 로그만 출력한다. --drive와 검증된 좌우 펄스가 있어야 출력한다.
시간 기반 S자 기동: 원래 차선 복귀/정확한 평행 자세/충돌 회피를 보장하지 않는다.
차량 폭, 회전반경, 센서 위치, 속도에 맞춰 아래 거리와 시간을 조정해야 한다.
"""
import argparse
from dataclasses import dataclass
import math
import time

I2C_BUS, PCA_ADDR, PCA_FREQ = 7, 0x40, 50
THROTTLE_CH, STEERING_CH = 8, 9
MUX_PIN, MUX_FREQ, MUX_DUTY = 15, 50, 10.0
NEUTRAL_US, DRIVE_SPEED_US, STEERING_CENTER_US = 1500, 1565, 1640
SCAN_TOPIC = '/scan'
FRONT_CENTER_DEG = 180.0  # 제공 코드: 센서 0도가 차량 뒤쪽
SCAN_TIMEOUT_SEC = 0.5
# 모두 LiDAR 원점에서 잰 거리. 차체 앞끝 기준 거리가 아님.
AVOID_DISTANCE_M = 0.80
CLEAR_DISTANCE_M = 0.95
EMERGENCY_DISTANCE_M = 0.30
BODY_CLEARANCE_M = 0.25
DIRECTION_SWITCH_MARGIN_M = 0.15
MIN_TURN_SEC, MAX_ALIGN_SEC = 0.40, 1.50
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
    ticks = max(0, min(4095, int(pulse_us * PCA_FREQ * 4096 / 1_000_000)))
    reg = LED0_ON_L + 4 * channel
    for offset, value in enumerate((0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F)):
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
            print(f'PCA9685 조향 출력: CH{STEERING_CH}, {us} us', flush=True)

    def stop(self):
        self.set_speed(NEUTRAL_US)

    def close(self):
        # 각 정리 작업이 실패하더라도 나머지 정리는 시도한다.
        actions = []
        if self.bus is not None:
            actions += [lambda: set_pwm_us(self.bus, THROTTLE_CH, NEUTRAL_US),
                        lambda: set_pwm_us(self.bus, STEERING_CH, STEERING_CENTER_US),
                        lambda: time.sleep(0.2)]
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
    """차량 앞 기준 반시계(왼쪽) 양수 거리값을 반환한다.

    범위 내 80% 이상 유효해야 사용한다.
    +inf는 제공 드라이버와 마찬가지로 측정범위 내 반사 없음으로 해석한다.
    실제 드라이버에서 inf가 고장을 뜻한다면 이 처리를 변경해야 한다.
    """
    if (not math.isfinite(scan.angle_min) or
            not math.isfinite(scan.angle_increment) or scan.angle_increment == 0 or
            not math.isfinite(scan.range_min) or scan.range_min < 0 or
            not math.isfinite(scan.range_max) or scan.range_max < CLEAR_DISTANCE_M or
            scan.range_min >= scan.range_max):
        return None
    total, valid, nearest = 0, 0, math.inf
    span = abs(math.degrees(scan.angle_increment))
    expected = (high_deg - low_deg) / span
    for i, distance in enumerate(scan.ranges):
        angle = scan.angle_min + i * scan.angle_increment - math.radians(FRONT_CENTER_DEG)
        relative = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
        if not low_deg - 1e-6 <= relative <= high_deg + 1e-6:
            continue
        total += 1
        if distance == math.inf:
            valid += 1
        elif math.isfinite(distance) and scan.range_min <= distance <= scan.range_max:
            valid += 1
            nearest = min(nearest, distance)
        elif math.isfinite(distance) and 0 < distance < scan.range_min:
            # 양수 최소거리 미만 값은 가까운 물체일 수 있으므로 정지 판단.
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


class AvoidController:
    """ROS/하드웨어와 분리한 자동 복구 회피 상태기계."""

    def __init__(self):
        self.state = 'WAIT'
        self.direction = None
        self.phase_time = 0.0
        self.turn_duration = 0.0
        self.clear_count = 0
        self.reason = ''

    def pause(self, reason):
        self.state, self.reason = 'PAUSED', reason
        return 'STOP'

    def update(self, d, now):
        if d is None:
            return self.pause('정면/좌/우 구간의 유효 측정 부족')
        if d.front <= EMERGENCY_DISTANCE_M + 1e-6:
            return self.pause('정면 30cm 이하: 장애물이 멀어지면 자동 재개')
        if min(d.left, d.right) <= BODY_CLEARANCE_M:
            return self.pause('차량 측면 25cm 이하: 공간 확보 시 자동 재개')
        if self.state in ('WAIT', 'PAUSED'):
            self.state = 'CRUISE'
            self.reason = ''
        if self.state == 'CRUISE':
            if d.front > AVOID_DISTANCE_M:
                return 'STRAIGHT'
            self.direction = 'LEFT' if d.left >= d.right else 'RIGHT'
            self.state, self.phase_time = 'TURN_OUT', now
            self.clear_count = 0
            return self.direction
        chosen = d.left if self.direction == 'LEFT' else d.right
        other = d.right if self.direction == 'LEFT' else d.left
        if other >= chosen + DIRECTION_SWITCH_MARGIN_M:
            self.direction = self.opposite()
            self.state, self.phase_time = 'TURN_OUT', now
            self.clear_count = 0
            return self.direction
        if self.state == 'TURN_OUT':
            elapsed = now - self.phase_time
            self.clear_count = self.clear_count + 1 if d.front >= CLEAR_DISTANCE_M else 0
            if elapsed >= MIN_TURN_SEC and self.clear_count >= CLEAR_SCANS:
                self.turn_duration = min(elapsed, MAX_ALIGN_SEC)
                self.state, self.phase_time = 'ALIGN', now
                return self.opposite()
            # Keep avoiding while the chosen corridor remains open. There is
            # intentionally no time limit; distance checks above remain active.
            return self.direction
        if d.front <= AVOID_DISTANCE_M:
            # A new obstacle appeared while aligning. Choose a fresh avoidance
            # direction rather than treating this as a permanent failure.
            self.state, self.direction = 'CRUISE', None
            return self.update(d, now)
        if self.state == 'ALIGN':
            if now - self.phase_time >= self.turn_duration:
                self.state = 'CRUISE'
                return 'STRAIGHT'
            return self.opposite()
        return self.pause('알 수 없는 상태')

    def opposite(self):
        return 'RIGHT' if self.direction == 'LEFT' else 'LEFT'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive', action='store_true', help='실제 PWM 출력')
    parser.add_argument('--left-us', type=int, default=1880, help='좌회전 펄스(us), 기본1880')
    parser.add_argument('--right-us', type=int, default=1400, help='우회전 펄스(us), 기본1400')
    parser.add_argument(
        '--steering-test',
        action='store_true',
        help='ESC 중립 상태에서 좌/우/중앙 조향만 시험하고 종료',
    )
    opts, ros_args = parser.parse_known_args()
    if opts.steering_test and not opts.drive:
        parser.error('--steering-test에는 실제 PWM 출력을 위한 --drive가 필요합니다')
    if opts.drive:
        if opts.left_us is None or opts.right_us is None:
            parser.error('--drive에는 --left-us와 --right-us가 필요합니다')
        if (not 1000 <= opts.left_us <= 2000 or not 1000 <= opts.right_us <= 2000 or
                (opts.left_us - STEERING_CENTER_US) *
                (opts.right_us - STEERING_CENTER_US) >= 0):
            parser.error('좌우는 1000~2000us 안에서 중앙1640us의 서로 반대편 값이어야 합니다')
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
    except ModuleNotFoundError as exc:
        print(f'ROS 2 Python 모듈을 불러올 수 없습니다: {exc}')
        print('먼저 다음 명령을 실행하세요:')
        print('  source /opt/ros/jazzy/setup.bash')
        print('  source ~/ros2_ws/install/setup.bash')
        return 2

    class LidarAvoidDrive(Node):
        def __init__(self, hardware):
            super().__init__('lidar_avoid_drive')
            self.hardware, self.control = hardware, AvoidController()
            self.last_scan_time = None
            self.last_status = None
            self.last_log_time = 0.0
            self.sub = self.create_subscription(LaserScan, SCAN_TOPIC,
                                                self.scan_callback, qos_profile_sensor_data)
            self.timer = self.create_timer(0.05, self.watchdog)
            self.get_logger().info('실제 주행' if hardware else '판단 로그 시험: PWM 출력 없음')
            self.get_logger().info(
                '감지 기준: 전방 0.80m 회피, 0.30m 급정지, '
                '좌우 중 더 넓은 통로 선택'
            )
            self.get_logger().info(
                '상태 표시: STRAIGHT=직진, LEFT/RIGHT=회피, '
                'PAUSED=장애물이 멀어질 때까지 일시 정지'
            )

        def apply(self, command):
            if self.hardware is None:
                return
            # 정지 명령은 조향 변경보다 ESC 중립을 먼저 출력한다.
            if command.startswith('STOP'):
                self.hardware.stop()
            direction = command.replace('STOP_', '')
            steering = {
                'LEFT': opts.left_us,
                'RIGHT': opts.right_us,
            }.get(direction, STEERING_CENTER_US)
            self.hardware.set_steering(steering)
            if not command.startswith('STOP'):
                self.hardware.set_speed(DRIVE_SPEED_US)

        def report(self, command, d=None):
            now = time.monotonic()
            status = (self.control.state, command, self.control.reason)
            if status != self.last_status or now - self.last_log_time >= 1.0:
                distances = (
                    ''
                    if d is None
                    else f' 앞={d.front:.2f} 좌={d.left:.2f} '
                    f'우={d.right:.2f}m'
                )
                self.get_logger().info(f'{status[0]}: {command}{distances} {self.control.reason}')
                self.last_status, self.last_log_time = status, now

        def scan_callback(self, scan):
            now = time.monotonic()
            # 수신시각뿐 아니라 ROS 시각상 오래된 스캔도 배제한다.
            stamp = scan.header.stamp.sec + scan.header.stamp.nanosec / 1e9
            ros_now = self.get_clock().now().nanoseconds / 1e9
            age = ros_now - stamp
            if stamp <= 0 or age > SCAN_TIMEOUT_SEC or age < -0.1:
                command = self.control.pause(
                    '스캔 타임스탬프가 없거나 오래됨/시계 불일치'
                )
                self.apply(command)
                self.report(command)
                return
            self.last_scan_time = now
            values = [sector_distance(scan, -45, 45),
                      sector_distance(scan, 45, 100), sector_distance(scan, -100, -45)]
            d = None if any(x is None for x in values) else Distances(*values)
            command = self.control.update(d, now)
            self.apply(command)
            self.report(command, d)

        def watchdog(self):
            if self.last_scan_time is None:
                self.apply('STOP')
                now = time.monotonic()
                if now - self.last_log_time >= 1.0:
                    self.get_logger().warning('/scan 대기 중: LiDAR 드라이버를 확인하세요')
                    self.last_log_time = now
                return
            if time.monotonic() - self.last_scan_time > SCAN_TIMEOUT_SEC:
                command = self.control.pause('LiDAR 데이터 0.5초 이상 끊김')
                self.apply(command)
                self.report(command)

    hardware = node = None
    initialized = False
    try:
        if opts.drive:
            hardware = DriveHardware()
            print('ESC 중립 안정화 3초')
            time.sleep(3.0)
            if opts.steering_test:
                for label, pulse_us in (
                    ('왼쪽', opts.left_us),
                    ('오른쪽', opts.right_us),
                    ('중앙', STEERING_CENTER_US),
                ):
                    print(f'조향 시험 {label}: {pulse_us} us')
                    hardware.set_steering(pulse_us)
                    time.sleep(1.0)
                return 0
        rclpy.init(args=ros_args)
        initialized = True
        node = LidarAvoidDrive(hardware)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print('정지 요청')
    except Exception as exc:
        print(f'실행 오류: {exc}')
        return 1
    finally:
        if hardware is not None:
            hardware.close()
        if node is not None:
            node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
