#!/usr/bin/env python3
"""Drive continuously and latch-stop in a front distance band."""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from smbus2 import SMBus


# test3.py와 동일한 구동 설정
I2C_BUS = 7
PCA_ADDR = 0x40
PCA_FREQ = 50
THROTTLE_CH = 8

MUX_PIN = 15
MUX_FREQ = 50
MUX_DUTY = 10.0

NEUTRAL_US = 1500
DRIVE_SPEED_US = 1565

# LiDAR 안전 설정
SCAN_TOPIC = "/scan"
STOP_DISTANCE_MIN_M = 0.20
STOP_DISTANCE_MAX_M = 0.30
DISTANCE_COMPARISON_TOLERANCE_M = 1e-6
SCAN_TIMEOUT_SEC = 0.5
# LiDAR is mounted with its 0° direction facing the rear of the vehicle.
FRONT_CENTER_DEG = 180.0
FRONT_HALF_ANGLE_DEG = 30.0

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


def init_pca9685(bus, frequency_hz=PCA_FREQ):
    prescale = int(round(25_000_000.0 / (4096.0 * frequency_hz) - 1.0))
    old_mode = bus.read_byte_data(PCA_ADDR, MODE1)
    awake_mode = old_mode & ~0x10

    bus.write_byte_data(PCA_ADDR, MODE1, (awake_mode & 0x7F) | 0x10)
    bus.write_byte_data(PCA_ADDR, PRESCALE, prescale)
    bus.write_byte_data(PCA_ADDR, MODE1, awake_mode)
    time.sleep(0.005)
    bus.write_byte_data(PCA_ADDR, MODE1, awake_mode | 0xA1)
    time.sleep(0.01)


def set_pwm_us(bus, channel, pulse_us):
    ticks = int(pulse_us * PCA_FREQ * 4096 / 1_000_000)
    ticks = max(0, min(4095, ticks))
    register = LED0_ON_L + (4 * channel)

    bus.write_byte_data(PCA_ADDR, register, 0)
    bus.write_byte_data(PCA_ADDR, register + 1, 0)
    bus.write_byte_data(PCA_ADDR, register + 2, ticks & 0xFF)
    bus.write_byte_data(PCA_ADDR, register + 3, (ticks >> 8) & 0x0F)


class DriveHardware:
    """Own the PCA9685 ESC output and MUX selection signal."""

    def __init__(self):
        self.bus = None
        self.mux_pwm = None
        self.gpio = None
        self.current_speed_us = None

        try:
            import Jetson.GPIO as GPIO

            self.gpio = GPIO
            self.bus = SMBus(I2C_BUS)
            init_pca9685(self.bus)

            # MUX 전환 전에 ESC 입력을 먼저 중립으로 만든다.
            self.set_speed(NEUTRAL_US)

            self.gpio.setwarnings(False)
            self.gpio.setmode(self.gpio.BOARD)
            self.gpio.setup(MUX_PIN, self.gpio.OUT)
            self.mux_pwm = self.gpio.PWM(MUX_PIN, MUX_FREQ)
            self.mux_pwm.start(MUX_DUTY)
        except Exception:
            self.close()
            raise

    def set_speed(self, pulse_us):
        if self.bus is None or self.current_speed_us == pulse_us:
            return
        set_pwm_us(self.bus, THROTTLE_CH, pulse_us)
        self.current_speed_us = pulse_us

    def stop(self):
        self.set_speed(NEUTRAL_US)

    def close(self):
        if self.bus is not None:
            try:
                set_pwm_us(self.bus, THROTTLE_CH, NEUTRAL_US)
                self.current_speed_us = NEUTRAL_US
                time.sleep(0.2)
            except OSError as exc:
                print(f"ESC 중립 출력 실패: {exc}")

        if self.mux_pwm is not None:
            self.mux_pwm.stop()
            self.mux_pwm = None

        if self.gpio is not None:
            self.gpio.cleanup()

        if self.bus is not None:
            self.bus.close()
            self.bus = None


def minimum_front_distance(scan):
    """Return the nearest valid distance in the forward detection sector."""
    front_center = math.radians(FRONT_CENTER_DEG)
    half_angle = math.radians(FRONT_HALF_ANGLE_DEG)
    nearest = math.inf
    has_measurement = False

    for index, distance in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        angle_from_front = math.atan2(
            math.sin(angle - front_center),
            math.cos(angle - front_center),
        )
        if abs(angle_from_front) > half_angle:
            continue

        # Positive infinity means no return inside the sensor's maximum range,
        # so it is a valid "clear" reading. NaN and negative infinity are invalid.
        if math.isinf(distance) and distance > 0:
            has_measurement = True
        elif math.isfinite(distance) and scan.range_min <= distance <= scan.range_max:
            has_measurement = True
            nearest = min(nearest, distance)

    if not has_measurement:
        return None
    return nearest


def is_in_stop_distance_band(distance):
    """Include the configured boundaries despite LaserScan float32 rounding."""
    return (
        distance > STOP_DISTANCE_MIN_M
        or math.isclose(
            distance,
            STOP_DISTANCE_MIN_M,
            rel_tol=0.0,
            abs_tol=DISTANCE_COMPARISON_TOLERANCE_M,
        )
    ) and (
        distance < STOP_DISTANCE_MAX_M
        or math.isclose(
            distance,
            STOP_DISTANCE_MAX_M,
            rel_tol=0.0,
            abs_tol=DISTANCE_COMPARISON_TOLERANCE_M,
        )
    )


class LidarStopDrive(Node):
    def __init__(self, hardware):
        super().__init__("lidar_stop_drive")
        self.hardware = hardware
        self.last_scan_time = None
        self.start_time = time.monotonic()
        self.last_wait_warning = 0.0
        self.driving = False
        self.stop_latched = False

        self.subscription = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.watchdog = self.create_timer(0.1, self.check_scan_timeout)

        self.get_logger().info(
            f"{SCAN_TOPIC} 대기 중: LiDAR {FRONT_CENTER_DEG:.0f}° 중심 "
            f"±{FRONT_HALF_ANGLE_DEG:.0f}°에서 "
            f"{STOP_DISTANCE_MIN_M:.2f}~{STOP_DISTANCE_MAX_M:.2f} m 감지 시 정지"
        )

    def latch_stop(self, reason):
        if self.stop_latched:
            return
        self.hardware.stop()
        self.driving = False
        self.stop_latched = True
        self.get_logger().error(f"정지: {reason}")
        self.get_logger().error("안전을 위해 자동 재출발하지 않습니다. 다시 실행하세요.")

    def scan_callback(self, scan):
        self.last_scan_time = time.monotonic()
        if self.stop_latched:
            return

        nearest = minimum_front_distance(scan)
        if nearest is None:
            self.latch_stop("정면 구간에 유효한 LiDAR 거리값이 없습니다")
            return

        if is_in_stop_distance_band(nearest):
            self.latch_stop(
                f"가장 가까운 장애물 {nearest:.3f} m "
                f"(정지 구간 {STOP_DISTANCE_MIN_M:.2f}~"
                f"{STOP_DISTANCE_MAX_M:.2f} m)"
            )
            return

        if not self.driving:
            self.hardware.set_speed(DRIVE_SPEED_US)
            self.driving = True
            nearest_text = (
                "측정 범위 내 장애물 없음"
                if math.isinf(nearest)
                else f"최근접 거리 {nearest:.3f} m"
            )
            self.get_logger().info(
                f"주행 시작: {DRIVE_SPEED_US} us, {nearest_text}, "
                "장애물 감지까지 계속 주행"
            )

    def check_scan_timeout(self):
        now = time.monotonic()

        if self.last_scan_time is None:
            if now - self.start_time >= 1.0 and now - self.last_wait_warning >= 2.0:
                self.get_logger().warning("/scan 데이터가 없어 중립 상태를 유지합니다")
                self.last_wait_warning = now
            return

        if not self.stop_latched and now - self.last_scan_time > SCAN_TIMEOUT_SEC:
            self.latch_stop(
                f"LiDAR 데이터가 {SCAN_TIMEOUT_SEC:.1f}초 이상 끊겼습니다"
            )


def main(args=None):
    hardware = None
    node = None
    ros_initialized = False

    try:
        # ESC를 먼저 안전하게 초기화하고 그 뒤 ROS 구독을 시작한다.
        hardware = DriveHardware()
        print("ESC 중립 신호 안정화 중 (3초)")
        time.sleep(3.0)

        rclpy.init(args=args)
        ros_initialized = True
        node = LidarStopDrive(hardware)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n정지 요청")
    except Exception as exc:
        print(f"실행 오류: {exc}")
        return 1
    finally:
        if hardware is not None:
            hardware.close()
            print("ESC 중립: 1500 us")
        if node is not None:
            node.destroy_node()
        if ros_initialized and rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
