#!/usr/bin/env python3
"""Drive a PCA9685 vehicle while avoiding obstacles from ROS 2 LaserScan."""

import time

from lidar_avoidance_logic import analyze_scan, AvoidanceController
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from smbus2 import SMBus


# Vehicle hardware settings verified by the existing steering/drive tests.
I2C_BUS = 7
PCA_ADDR = 0x40
PCA_FREQ = 50
THROTTLE_CH = 8
STEERING_CH = 9

MUX_PIN = 15
MUX_FREQ = 50
MUX_DUTY = 10.0

NEUTRAL_US = 1500
DRIVE_SPEED_US = 1565
AVOID_SPEED_US = 1545
STEERING_CENTER_US = 1640
STEERING_LEFT_US = 1880
STEERING_RIGHT_US = 1400
STEERING_STEP_US = 10

SCAN_TOPIC = '/scan'
SCAN_TIMEOUT_SEC = 0.5
FRONT_CENTER_DEG = 180.0
FRONT_HALF_WIDTH_DEG = 25.0
SIDE_CENTER_OFFSET_DEG = 50.0
SIDE_HALF_WIDTH_DEG = 25.0

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
    """Own throttle, steering, and automatic-input MUX outputs."""

    def __init__(self):
        self.bus = None
        self.gpio = None
        self.mux_pwm = None
        self.current_speed_us = None
        self.current_steering_us = None
        self.target_steering_us = STEERING_CENTER_US

        try:
            import Jetson.GPIO as GPIO

            self.gpio = GPIO
            self.bus = SMBus(I2C_BUS)
            init_pca9685(self.bus)

            # Establish safe outputs before switching the MUX to automation.
            self.set_speed(NEUTRAL_US)
            self.set_steering_immediate(STEERING_CENTER_US)

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

    def set_steering_target(self, pulse_us):
        self.target_steering_us = pulse_us

    def steering_is_at_target(self):
        return self.current_steering_us == self.target_steering_us

    def set_steering_immediate(self, pulse_us):
        if self.bus is None or self.current_steering_us == pulse_us:
            return
        set_pwm_us(self.bus, STEERING_CH, pulse_us)
        self.current_steering_us = pulse_us
        self.target_steering_us = pulse_us

    def update_steering(self):
        if self.bus is None or self.current_steering_us == self.target_steering_us:
            return
        difference = self.target_steering_us - self.current_steering_us
        step = max(-STEERING_STEP_US, min(STEERING_STEP_US, difference))
        new_value = self.current_steering_us + step
        set_pwm_us(self.bus, STEERING_CH, new_value)
        self.current_steering_us = new_value

    def safe_stop(self):
        self.set_speed(NEUTRAL_US)
        self.set_steering_target(STEERING_CENTER_US)

    def close(self):
        if self.bus is not None:
            try:
                set_pwm_us(self.bus, THROTTLE_CH, NEUTRAL_US)
                set_pwm_us(self.bus, STEERING_CH, STEERING_CENTER_US)
                time.sleep(0.2)
            except OSError as exc:
                print(f'안전 출력 실패: {exc}')

        if self.mux_pwm is not None:
            self.mux_pwm.stop()
            self.mux_pwm = None
        if self.gpio is not None:
            self.gpio.cleanup()
        if self.bus is not None:
            self.bus.close()
            self.bus = None


class LidarObstacleAvoidance(Node):
    def __init__(self, hardware):
        super().__init__('lidar_obstacle_avoidance')
        self.hardware = hardware
        self.controller = AvoidanceController()
        self.last_scan_time = None
        self.start_time = time.monotonic()
        self.last_wait_warning = 0.0
        self.last_mode = None
        self.timeout_stopped = False

        self.subscription = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.watchdog = self.create_timer(0.1, self.check_scan_timeout)
        self.steering_timer = self.create_timer(0.02, self.update_steering)

        cfg = self.controller.config
        self.get_logger().info(
            f'{SCAN_TOPIC} 대기 중: 정면 {FRONT_CENTER_DEG:.0f}°, '
            f'{cfg.avoid_distance_m:.2f} m부터 좌우 통로 비교 회피'
        )
        self.get_logger().info(
            f'급정지 {cfg.emergency_distance_m:.2f} m, '
            f'직진 복귀 {cfg.clear_distance_m:.2f} m, '
            f'측면 최소 여유 {cfg.minimum_side_clearance_m:.2f} m'
        )

    def update_steering(self):
        self.hardware.update_steering()

    def apply_command(self, command):
        steering_us = {
            'left': STEERING_LEFT_US,
            'center': STEERING_CENTER_US,
            'right': STEERING_RIGHT_US,
        }[command.steering]

        self.hardware.set_steering_target(steering_us)

        # Do not move while the steering servo is crossing to a new target.
        if not command.moving or not self.hardware.steering_is_at_target():
            self.hardware.set_speed(NEUTRAL_US)
        else:
            speed = (
                AVOID_SPEED_US
                if command.mode.startswith('avoiding_')
                else DRIVE_SPEED_US
            )
            self.hardware.set_speed(speed)

        if command.mode != self.last_mode:
            log = (
                self.get_logger().warning
                if not command.moving
                else self.get_logger().info
            )
            log(f'{command.mode}: {command.reason}')
            self.last_mode = command.mode

    def scan_callback(self, scan):
        self.last_scan_time = time.monotonic()
        self.timeout_stopped = False
        sectors = analyze_scan(
            scan,
            front_center_deg=FRONT_CENTER_DEG,
            front_half_width_deg=FRONT_HALF_WIDTH_DEG,
            side_center_offset_deg=SIDE_CENTER_OFFSET_DEG,
            side_half_width_deg=SIDE_HALF_WIDTH_DEG,
        )
        self.apply_command(self.controller.update(sectors))

    def check_scan_timeout(self):
        now = time.monotonic()
        if self.last_scan_time is None:
            if now - self.start_time >= 1.0 and now - self.last_wait_warning >= 2.0:
                self.get_logger().warning(
                    '/scan 데이터가 없어 중립 상태를 유지합니다'
                )
                self.last_wait_warning = now
            return

        if now - self.last_scan_time > SCAN_TIMEOUT_SEC and not self.timeout_stopped:
            self.hardware.safe_stop()
            self.timeout_stopped = True
            self.last_mode = 'scan_timeout'
            self.get_logger().error(
                f'LiDAR 데이터가 {SCAN_TIMEOUT_SEC:.1f}초 이상 '
                '끊겨 정지합니다'
            )


def main(args=None):
    hardware = None
    node = None
    ros_initialized = False

    try:
        hardware = DriveHardware()
        print('ESC 중립 신호 안정화 중 (3초)')
        time.sleep(3.0)

        rclpy.init(args=args)
        ros_initialized = True
        node = LidarObstacleAvoidance(hardware)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n정지 요청')
    except Exception as exc:
        print(f'실행 오류: {exc}')
        return 1
    finally:
        if hardware is not None:
            hardware.close()
            print('ESC 중립 1500 us, 조향 중앙 1640 us')
        if node is not None:
            node.destroy_node()
        if ros_initialized and rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
