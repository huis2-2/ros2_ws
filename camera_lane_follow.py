#!/usr/bin/env python3
"""Follow a white lane and display separate white/yellow masks.

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
SLOW_SPEED_US = 1600
MAX_SPEED_US = 1600
STEERING_CENTER_US = 1640
STEERING_LEFT_US = 1880
STEERING_RIGHT_US = 1400

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

CAMERA_TOPIC = '/camera/image_raw'
IMAGE_TIMEOUT_SEC = 0.5
LANE_CONFIRM_FRAMES = 3
SLIDING_WINDOW_COUNT = 8
SLIDING_WINDOW_MARGIN_PX = 45
SLIDING_WINDOW_MIN_PIXELS = 20

# Steering behavior from the supplied lane follower.
KP_LATERAL = 0.0042
KP_HEADING = 0.15
DEADBAND_PX = 24.0
MAX_ANGULAR = 3.20
SMOOTH_ALPHA = 0.60
MAX_DELTA_ANGULAR = 2.35

WHITE_LOWER_HSV = (0, 0, 252)
WHITE_UPPER_HSV = (0, 116, 255)
YELLOW_LOWER_HSV = (7, 21, 201)
YELLOW_UPPER_HSV = (179, 255, 255)


@dataclass
class LaneObservation:
    """Detected lane position and direction in the bird's-eye image."""

    center_x: float
    angle_deg: float
    lines: np.ndarray
    bev: np.ndarray
    mask: np.ndarray
    target_y: int
    track: tuple
    curve: np.ndarray
    bottom_x: float


def bird_eye_view(image):
    """Transform the road trapezoid into a front-facing rectangular view."""
    height, width = image.shape[:2]
    source = np.float32([
        (int(width * 0.30), int(height * 0.55)),
        (int(width * 0.70), int(height * 0.55)),
        (int(width * 0.96), int(height * 0.96)),
        (int(width * 0.04), int(height * 0.96)),
    ])
    bev_width = int(width * 0.60)
    x_min = int((width - bev_width) / 2)
    destination = np.float32([
        (x_min, 0),
        (x_min + bev_width, 0),
        (x_min + bev_width, height),
        (x_min, height),
    ])
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def lane_color_mask(image, lower_hsv, upper_hsv):
    """Return a cleaned rectangular lower-road mask for one HSV range."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )
    height, width = mask.shape
    roi = np.zeros_like(mask)
    top_left = (int(width * 0.02), int(height * 0.58))
    bottom_right = (int(width * 0.98), int(height * 0.98))
    cv2.rectangle(roi, top_left, bottom_right, 255, thickness=-1)
    mask = cv2.bitwise_and(mask, roi)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_edges(mask):
    """Blur the white mask and return the supplied Canny edge result."""
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        50,
        150,
        apertureSize=3,
        L2gradient=True,
    )
    return blurred, edges


def detect_hough_lines(edges):
    """Detect lane segments with the supplied probabilistic Hough values."""
    return cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=18,
        maxLineGap=28,
    )


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


def sliding_window_track(mask, initial_x):
    """Track mask pixels upward from the current steering target."""
    height, width = mask.shape
    roi_top = int(height * 0.58)
    roi_bottom = int(height * 0.98)
    boundaries = np.linspace(roi_bottom + 1, roi_top, SLIDING_WINDOW_COUNT + 1).astype(int)
    margin = max(1, int(SLIDING_WINDOW_MARGIN_PX * width / 640.0))
    current_x = int(np.clip(initial_x, 0, width - 1))
    nonzero_y, nonzero_x = mask.nonzero()
    windows = []
    centers = []
    selected_indices = []

    for window_index in range(SLIDING_WINDOW_COUNT):
        y_high = int(boundaries[window_index])
        y_low = int(boundaries[window_index + 1])
        x_low = max(0, current_x - margin)
        x_high = min(width - 1, current_x + margin)
        windows.append((x_low, y_low, x_high, y_high))

        indices = np.flatnonzero(
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= x_low)
            & (nonzero_x <= x_high)
        )
        if indices.size >= SLIDING_WINDOW_MIN_PIXELS:
            current_x = int(np.mean(nonzero_x[indices]))
            centers.append((current_x, int(np.mean(nonzero_y[indices]))))
            selected_indices.append(indices)
        else:
            centers.append(None)

    if selected_indices:
        selected = np.concatenate(selected_indices)
    else:
        selected = np.empty(0, dtype=np.int64)
    return windows, centers, nonzero_x[selected], nonzero_y[selected]


def draw_sliding_windows(image, mask, initial_x, track=None):
    """Overlay the active sliding windows and selected lane pixels."""
    display = image.copy()
    windows, centers, selected_x, selected_y = (
        track if track is not None else sliding_window_track(mask, initial_x)
    )
    display[selected_y, selected_x] = (255, 0, 255)
    for (x_low, y_low, x_high, y_high), center in zip(windows, centers):
        cv2.rectangle(
            display,
            (x_low, y_low),
            (x_high, y_high),
            (255, 255, 0) if center is not None else (0, 0, 255),
            2,
        )
    for center in centers:
        if center is not None:
            cv2.circle(display, center, 3, (0, 255, 255), -1)
    return display


def fit_lane_track(mask, previous_x=None):
    """Choose a bottom-connected stripe and fit its observed window centers."""
    height, width = mask.shape
    histogram = np.count_nonzero(mask[int(height * 0.88):int(height * 0.98) + 1], axis=0)
    candidates = np.flatnonzero(histogram >= max(3, int(height * 0.02)))
    if not candidates.size:
        return None
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    seeds = [float(np.mean(group)) for group in groups]
    reference = width / 2 if previous_x is None else previous_x
    seed = min(seeds, key=lambda x: abs(x - reference))
    if previous_x is not None and abs(seed - previous_x) > width * 0.15:
        return None
    track = sliding_window_track(mask, seed)
    centers = track[1]
    valid = np.array([point for point in centers if point is not None])
    # Require support near the vehicle and around the forward target.
    target_y = int(height * 0.66)
    if (len(valid) < 5 or centers[0] is None
            or valid[:, 1].min() > target_y
            or valid[:, 1].max() < height * 0.90):
        return None
    coefficients = np.polyfit(valid[:, 1] / height, valid[:, 0], 2)
    residual = np.abs(np.polyval(coefficients, valid[:, 1] / height) - valid[:, 0])
    if residual.max() > width * 0.04:
        return None
    target_x = float(np.polyval(coefficients, target_y / height))
    bottom_y = int(valid[:, 1].max())
    bottom_x = float(np.polyval(coefficients, bottom_y / height))
    if not 0 <= target_x < width:
        return None
    angle = 90.0 + float(np.degrees(np.arctan2(target_x - bottom_x, bottom_y - target_y)))
    ys = np.arange(int(valid[:, 1].min()), bottom_y + 1)
    xs = np.clip(np.polyval(coefficients, ys / height), 0, width - 1)
    curve = np.column_stack((xs, ys)).astype(np.int32)
    return target_x, angle, target_y, track, curve, bottom_x


def curve_speed_us(max_speed, angle_deg, steering_us):
    """Reduce throttle toward neutral as path heading or steering grows."""
    severity = min(1.0, max(abs(angle_deg - 90.0) / 30.0,
                            abs(steering_us - STEERING_CENTER_US) / 240.0))
    return int(round(max_speed - (max_speed - NEUTRAL_US) * 0.40 * severity))


def detect_white_lane(image, previous_x=None):
    """Detect only the white lane and also return a yellow preview mask."""
    bev = bird_eye_view(image)
    white_mask = lane_color_mask(bev, WHITE_LOWER_HSV, WHITE_UPPER_HSV)
    yellow_mask = lane_color_mask(bev, YELLOW_LOWER_HSV, YELLOW_UPPER_HSV)
    fit = fit_lane_track(white_mask, previous_x)
    if fit is None:
        return None, bev, white_mask, yellow_mask
    center_x, angle_deg, target_y, track, curve, bottom_x = fit
    observation = LaneObservation(
        center_x,
        angle_deg,
        np.empty((0, 1, 4), dtype=np.int32),
        bev,
        white_mask,
        target_y,
        track,
        curve,
        bottom_x,
    )
    return observation, bev, white_mask, yellow_mask


def write_status_log(logger, text, warning=False):
    """Log each severity from a stable ROS 2 call site."""
    if warning:
        logger.warning(text)
    else:
        logger.info(text)


class LaneController:
    """Convert lane errors to calibrated steering pulses."""

    def __init__(self):
        """Start with centered steering and no previous correction."""
        self.previous_command = 0.0

    def reset(self):
        """Reset smoothing state when the lane is lost."""
        self.previous_command = 0.0

    def steering_pulse(self, center_x, angle_deg, image_width):
        """Return the next rate-limited steering pulse in microseconds."""
        scale = image_width / 640.0
        lateral_error = center_x - image_width / 2.0
        heading_error = angle_deg - 90.0

        if abs(lateral_error) < DEADBAND_PX * scale:
            lateral_error = 0.0
        if abs(heading_error) < 2.0:
            heading_error = 0.0

        command = (
            -KP_LATERAL / scale * lateral_error
            - KP_HEADING * heading_error
        )
        command = float(np.clip(command, -MAX_ANGULAR, MAX_ANGULAR))
        command = (
            SMOOTH_ALPHA * command
            + (1.0 - SMOOTH_ALPHA) * self.previous_command
        )
        delta = command - self.previous_command
        if delta > MAX_DELTA_ANGULAR:
            command = self.previous_command + MAX_DELTA_ANGULAR
        elif delta < -MAX_DELTA_ANGULAR:
            command = self.previous_command - MAX_DELTA_ANGULAR

        if command >= 0.0:
            pulse = STEERING_CENTER_US + (
                command
                / MAX_ANGULAR
                * (STEERING_LEFT_US - STEERING_CENTER_US)
            )
        else:
            pulse = STEERING_CENTER_US + (
                -command
                / MAX_ANGULAR
                * (STEERING_RIGHT_US - STEERING_CENTER_US)
            )

        pulse = int(round(np.clip(
            pulse,
            STEERING_RIGHT_US,
            STEERING_LEFT_US,
        )))
        self.previous_command = command
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
    if not NEUTRAL_US <= options.speed_us <= MAX_SPEED_US:
        parser.error(
            f'--speed-us는 안전상 {NEUTRAL_US}~{MAX_SPEED_US} '
            '범위만 허용합니다'
        )

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
            self.previous_lane_x = None
            self.subscription = self.create_subscription(
                Image,
                options.camera_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
            self.watchdog = self.create_timer(0.1, self.check_image_timeout)
            mode = (
                '실제 차선 주행'
                if self.hardware is not None
                else '판단 로그 시험'
            )
            self.get_logger().info(
                f'{mode}: {options.camera_topic}의 흰 차선을 기다립니다'
            )
            self.get_logger().info(
                f'차선 감지 시 속도 CH{THROTTLE_CH}={options.speed_us} us, '
                f'조향 CH{STEERING_CH}={STEERING_RIGHT_US}~'
                f'{STEERING_LEFT_US} us'
            )

        def apply(self, steering_us, moving, speed_us=NEUTRAL_US):
            """Apply a drive command, or do nothing in report-only mode."""
            if self.hardware is None:
                return
            if moving:
                self.hardware.set_steering(steering_us)
                self.hardware.set_speed(speed_us)
            else:
                self.hardware.stop()

        def report(self, key, text, warning=False):
            """Log state changes immediately and steady state once a second."""
            now = time.monotonic()
            if key != self.last_log_key or now - self.last_log_at >= 1.0:
                write_status_log(self.get_logger(), text, warning)
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
                observation, bev, white_mask, yellow_mask = (
                    detect_white_lane(image, self.previous_lane_x)
                )
            except Exception as exc:
                self.previous_lane_x = None
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
                self.previous_lane_x = None
                self.controller.reset()
                self.lane_frames = 0
                self.apply(STEERING_CENTER_US, False)
                self.report(
                    'lost',
                    '흰 차선을 찾지 못해 중립 정지',
                    warning=True,
                )
                steering_us = STEERING_CENTER_US
                moving = False
                speed_us = NEUTRAL_US
            else:
                self.previous_lane_x = observation.bottom_x
                steering_us = self.controller.steering_pulse(
                    observation.center_x,
                    observation.angle_deg,
                    image.shape[1],
                )
                self.lane_frames += 1
                moving = self.lane_frames >= LANE_CONFIRM_FRAMES
                speed_us = curve_speed_us(options.speed_us, observation.angle_deg, steering_us)
                if not moving:
                    speed_us = NEUTRAL_US
                self.apply(steering_us, moving, speed_us)
                if moving:
                    self.report(
                        'tracking',
                        f'차선 x={observation.center_x:.1f}, '
                        f'각도={observation.angle_deg:.1f}°, '
                        f'조향={steering_us} us, 속도={speed_us} us',
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
                    display = draw_sliding_windows(
                        display,
                        white_mask,
                        observation.center_x,
                        observation.track,
                    )
                    cv2.polylines(display, [observation.curve], False, (0, 255, 255), 2)
                    cv2.circle(
                        display,
                        (
                            int(observation.center_x),
                            observation.target_y,
                        ),
                        7,
                        (0, 255, 0),
                        -1,
                    )
                status = ('DRIVE' if options.drive else 'DRY RUN') if moving else (
                    'STOP: LANE LOST' if observation is None else 'STOP: CONFIRMING'
                )
                cv2.putText(
                    display,
                    f'{status} steer={steering_us} speed={speed_us} us',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    'CYAN valid  RED missing  YELLOW fit  GREEN steering target',
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow('lane follow', display)
                cv2.imshow('white mask - steering input', white_mask)
                cv2.imshow('yellow mask - preview only', yellow_mask)
                cv2.waitKey(1)

        def check_image_timeout(self):
            """Stop if camera images cease arriving."""
            reference = self.last_image_at or self.started_at
            if time.monotonic() - reference <= IMAGE_TIMEOUT_SEC:
                return
            self.controller.reset()
            self.previous_lane_x = None
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
