#!/usr/bin/env python3
"""Follow a yellow lane from a ROS 2 camera image at low speed.

The default mode only reports the command it would send. Pass ``--drive``
after checking the camera image and steering direction with the wheels raised.
"""

import argparse
import time
from dataclasses import dataclass

import cv2

import numpy as np


# Hardware values verified by the existing vehicle tests in this workspace.
I2C_BUS = 7
PCA_ADDR = 0x40
PCA_FREQ = 50
THROTTLE_CH = 8
STEERING_CH = 9

MUX_PIN = 15
MUX_FREQ = 50
MUX_DUTY = 10.0

NEUTRAL_US = 1500
SLOW_SPEED_US = 1545
STEERING_CENTER_US = 1640
STEERING_LEFT_US = 1880
STEERING_RIGHT_US = 1400

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

CAMERA_TOPIC = '/camera/image_raw'
IMAGE_TIMEOUT_SEC = 0.5
LANE_CONFIRM_FRAMES = 3


@dataclass
class LaneObservation:
    """Detected lane position and direction in the bird's-eye image."""

    center_x: float
    angle_deg: float
    lines: np.ndarray
    bev: np.ndarray
    mask: np.ndarray


def bird_eye_view(image):
    """Transform the road trapezoid into a front-facing rectangular view."""
    height, width = image.shape[:2]
    source = np.float32([
        (width * 0.30, height * 0.55),
        (width * 0.70, height * 0.55),
        (width * 0.96, height * 0.96),
        (width * 0.04, height * 0.96),
    ])
    bev_width = width * 0.60
    x_min = (width - bev_width) / 2.0
    destination = np.float32([
        (x_min, 0),
        (x_min + bev_width, 0),
        (x_min + bev_width, height),
        (x_min, height),
    ])
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def yellow_lane_mask(image):
    """Return a cleaned mask for the yellow lane in the lower road area."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([18, 70, 70], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8),
    )
    height, width = mask.shape
    roi = np.zeros_like(mask)
    polygon = np.array([[
        (int(width * 0.08), int(height * 0.58)),
        (int(width * 0.92), int(height * 0.58)),
        (int(width * 0.98), int(height * 0.98)),
        (int(width * 0.02), int(height * 0.98)),
    ]], dtype=np.int32)
    cv2.fillPoly(roi, polygon, 255)
    mask = cv2.bitwise_and(mask, roi)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def calculate_lane_info(lines, image_height):
    """Calculate a lower-image-weighted lane x position and line angle."""
    if lines is None:
        return None, None

    x_values = []
    angles = []
    min_y = int(image_height * 0.58)
    max_y = int(image_height * 0.96)

    for x1, y1, x2, y2 in lines[:, 0]:
        if not (min_y <= y1 <= max_y or min_y <= y2 <= max_y):
            continue
        dx = int(x2) - int(x1)
        dy = int(y2) - int(y1)
        if dx == 0 and dy == 0:
            continue

        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle < 0:
            angle += 180.0
        if angle < 8.0 or angle > 172.0:
            continue

        x_values.extend([int(x1)] * int(1 + y1 / image_height * 3))
        x_values.extend([int(x2)] * int(1 + y2 / image_height * 3))
        angles.append(angle)

    if not x_values or not angles:
        return None, None
    return float(np.mean(x_values)), float(np.mean(angles))


def detect_yellow_lane(image):
    """Detect the lane using the supplied code's HSV and Hough method."""
    bev = bird_eye_view(image)
    mask = yellow_lane_mask(bev)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=18,
        maxLineGap=28,
    )
    center_x, angle_deg = calculate_lane_info(lines, image.shape[0])
    if center_x is None:
        return None, bev, mask
    return LaneObservation(center_x, angle_deg, lines, bev, mask), bev, mask


class LaneController:
    """Convert lane errors to calibrated steering pulses."""

    def __init__(self):
        """Start with centered steering and no previous correction."""
        self.previous_command = 0.0
        self.previous_pulse = STEERING_CENTER_US

    def reset(self):
        """Reset smoothing state when the lane is lost."""
        self.previous_command = 0.0
        self.previous_pulse = STEERING_CENTER_US

    def steering_pulse(self, center_x, angle_deg, image_width):
        """Return the next rate-limited steering pulse in microseconds."""
        scale = image_width / 640.0
        target_x = image_width / 2.0 - 56.0 * scale
        lateral_error = center_x - target_x
        heading_error = angle_deg - 90.0

        if abs(lateral_error) < 34.0 * scale:
            lateral_error = 0.0
        if abs(heading_error) < 2.0:
            heading_error = 0.0

        command = -0.0033 / scale * lateral_error - 0.20 * heading_error
        command = float(np.clip(command, -8.0, 8.0))
        command = 0.75 * command + 0.25 * self.previous_command

        if command >= 0.0:
            pulse = STEERING_CENTER_US + (
                command / 8.0 * (STEERING_LEFT_US - STEERING_CENTER_US)
            )
        else:
            pulse = STEERING_CENTER_US + (
                -command / 8.0 * (STEERING_RIGHT_US - STEERING_CENTER_US)
            )

        # Limit the servo movement per camera frame to avoid abrupt steering.
        pulse = float(np.clip(
            pulse,
            self.previous_pulse - 12,
            self.previous_pulse + 12,
        ))
        pulse = int(round(np.clip(
            pulse,
            STEERING_RIGHT_US,
            STEERING_LEFT_US,
        )))
        self.previous_command = command
        self.previous_pulse = pulse
        return pulse


def init_pca9685(bus):
    """Configure the PCA9685 for 50 Hz servo and ESC output."""
    prescale = int(round(25_000_000.0 / (4096.0 * PCA_FREQ) - 1.0))
    old_mode = bus.read_byte_data(PCA_ADDR, MODE1)
    awake_mode = old_mode & ~0x10
    bus.write_byte_data(PCA_ADDR, MODE1, (awake_mode & 0x7F) | 0x10)
    bus.write_byte_data(PCA_ADDR, PRESCALE, prescale)
    bus.write_byte_data(PCA_ADDR, MODE1, awake_mode)
    time.sleep(0.005)
    bus.write_byte_data(PCA_ADDR, MODE1, awake_mode | 0xA1)
    time.sleep(0.01)


def set_pwm_us(bus, channel, pulse_us):
    """Write one PCA9685 channel pulse width in microseconds."""
    ticks = int(pulse_us * PCA_FREQ * 4096 / 1_000_000)
    ticks = max(0, min(4095, ticks))
    register = LED0_ON_L + 4 * channel
    values = (0, 0, ticks & 0xFF, (ticks >> 8) & 0x0F)
    for offset, value in enumerate(values):
        bus.write_byte_data(PCA_ADDR, register + offset, value)


class DriveHardware:
    """Own the PCA9685 drive outputs and automatic-input MUX."""

    def __init__(self):
        """Initialize safe outputs before selecting the automatic input."""
        from smbus2 import SMBus
        import Jetson.GPIO as GPIO

        self.bus = None
        self.gpio = GPIO
        self.mux_pwm = None
        self.speed_us = None
        self.steering_us = None
        try:
            self.bus = SMBus(I2C_BUS)
            init_pca9685(self.bus)
            self.set_speed(NEUTRAL_US)
            self.set_steering(STEERING_CENTER_US)

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(MUX_PIN, GPIO.OUT)
            self.mux_pwm = GPIO.PWM(MUX_PIN, MUX_FREQ)
            self.mux_pwm.start(MUX_DUTY)
        except Exception:
            self.close()
            raise

    def set_speed(self, pulse_us):
        """Set the ESC pulse when its value has changed."""
        if self.bus is not None and pulse_us != self.speed_us:
            set_pwm_us(self.bus, THROTTLE_CH, pulse_us)
            self.speed_us = pulse_us

    def set_steering(self, pulse_us):
        """Set the steering pulse when its value has changed."""
        if self.bus is not None and pulse_us != self.steering_us:
            set_pwm_us(self.bus, STEERING_CH, pulse_us)
            self.steering_us = pulse_us

    def stop(self):
        """Set neutral throttle and centered steering."""
        self.set_speed(NEUTRAL_US)
        self.set_steering(STEERING_CENTER_US)

    def close(self):
        """Restore safe outputs and release I2C and GPIO resources."""
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


def build_parser():
    """Build the command-line parser without importing ROS 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--drive',
        action='store_true',
        help='PCA9685에 실제 조향 및 저속 주행 PWM을 출력',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='차선 인식 결과 OpenCV 창 표시',
    )
    parser.add_argument('--camera-topic', default=CAMERA_TOPIC)
    parser.add_argument(
        '--speed-us',
        type=int,
        default=SLOW_SPEED_US,
        help=f'차선 감지 시 속도 펄스(기본 {SLOW_SPEED_US} us)',
    )
    return parser


def main(argv=None):
    """Run the ROS 2 lane follower and always restore safe outputs."""
    parser = build_parser()
    options, ros_args = parser.parse_known_args(argv)
    if not NEUTRAL_US <= options.speed_us <= 1565:
        parser.error('--speed-us는 안전상 1500~1565 범위만 허용합니다')

    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
    except ModuleNotFoundError as exc:
        print(f'ROS 2 모듈을 불러올 수 없습니다: {exc}')
        print('source /opt/ros/jazzy/setup.bash 후 다시 실행하세요.')
        return 2

    hardware = None
    node = None
    ros_initialized = False

    class CameraLaneFollower(Node):
        """Subscribe to camera images and apply lane-following commands."""

        def __init__(self, drive_hardware):
            """Set up image subscription, controller, and watchdog."""
            super().__init__('camera_lane_follower')
            self.hardware = drive_hardware
            self.bridge = CvBridge()
            self.controller = LaneController()
            self.started_at = time.monotonic()
            self.last_image_at = None
            self.last_log_at = 0.0
            self.last_log_key = None
            self.timeout_reported = False
            self.lane_frames = 0
            self.subscription = self.create_subscription(
                Image,
                options.camera_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
            self.watchdog = self.create_timer(0.1, self.check_image_timeout)
            mode = (
                '실제 저속 주행'
                if self.hardware is not None
                else '판단 로그 시험'
            )
            self.get_logger().info(
                f'{mode}: {options.camera_topic}의 노란 차선을 기다립니다'
            )
            self.get_logger().info(
                f'차선 감지 시 속도 CH{THROTTLE_CH}={options.speed_us} us, '
                f'조향 CH{STEERING_CH}={STEERING_RIGHT_US}~'
                f'{STEERING_LEFT_US} us'
            )

        def apply(self, steering_us, moving):
            """Apply a drive command, or do nothing in report-only mode."""
            if self.hardware is None:
                return
            if moving:
                self.hardware.set_steering(steering_us)
                self.hardware.set_speed(options.speed_us)
            else:
                self.hardware.stop()

        def report(self, key, text, warning=False):
            """Log state changes immediately and steady state once a second."""
            now = time.monotonic()
            if key != self.last_log_key or now - self.last_log_at >= 1.0:
                log = (
                    self.get_logger().warning
                    if warning
                    else self.get_logger().info
                )
                log(text)
                self.last_log_at = now
                self.last_log_key = key

        def image_callback(self, message):
            """Detect a lane and update steering for one camera image."""
            self.last_image_at = time.monotonic()
            self.timeout_reported = False
            try:
                image = self.bridge.imgmsg_to_cv2(
                    message,
                    desired_encoding='bgr8',
                )
                observation, bev, mask = detect_yellow_lane(image)
            except Exception as exc:
                self.controller.reset()
                self.lane_frames = 0
                self.apply(STEERING_CENTER_US, False)
                self.report(
                    'error',
                    f'영상 처리 오류로 정지: {exc}',
                    warning=True,
                )
                return

            if observation is None:
                self.controller.reset()
                self.lane_frames = 0
                self.apply(STEERING_CENTER_US, False)
                self.report(
                    'lost',
                    '노란 차선을 찾지 못해 중립 정지',
                    warning=True,
                )
                steering_us = STEERING_CENTER_US
                moving = False
            else:
                steering_us = self.controller.steering_pulse(
                    observation.center_x,
                    observation.angle_deg,
                    image.shape[1],
                )
                self.lane_frames += 1
                moving = self.lane_frames >= LANE_CONFIRM_FRAMES
                self.apply(steering_us, moving)
                if moving:
                    self.report(
                        'tracking',
                        f'차선 x={observation.center_x:.1f}, '
                        f'각도={observation.angle_deg:.1f}°, '
                        f'조향={steering_us} us, 속도={options.speed_us} us',
                    )
                else:
                    self.report(
                        'confirming',
                        f'차선 연속 감지 확인 중 '
                        f'({self.lane_frames}/{LANE_CONFIRM_FRAMES})',
                    )

            if options.show:
                display = bev.copy()
                if observation is not None:
                    for x1, y1, x2, y2 in observation.lines[:, 0]:
                        cv2.line(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2,
                        )
                    cv2.circle(
                        display,
                        (
                            int(observation.center_x),
                            int(image.shape[0] * 0.78),
                        ),
                        7,
                        (0, 255, 0),
                        -1,
                    )
                status = 'DRIVE' if moving else 'STOP: LANE LOST'
                cv2.putText(
                    display,
                    f'{status} steer={steering_us} us',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow('yellow lane follow', display)
                cv2.imshow('yellow lane mask', mask)
                cv2.waitKey(1)

        def check_image_timeout(self):
            """Stop if camera images cease arriving."""
            reference = self.last_image_at or self.started_at
            if time.monotonic() - reference <= IMAGE_TIMEOUT_SEC:
                return
            self.controller.reset()
            self.lane_frames = 0
            self.apply(STEERING_CENTER_US, False)
            if not self.timeout_reported:
                self.get_logger().error(
                    f'카메라 영상이 {IMAGE_TIMEOUT_SEC:.1f}초 이상 없어 중립 정지'
                )
                self.timeout_reported = True

    try:
        if options.drive:
            hardware = DriveHardware()
            print('ESC 중립 신호 안정화 중 (3초)')
            time.sleep(3.0)

        rclpy.init(args=ros_args)
        ros_initialized = True
        node = CameraLaneFollower(hardware)
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
        cv2.destroyAllWindows()
        if ros_initialized and rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
