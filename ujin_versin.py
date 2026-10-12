#!/usr/bin/env python3
"""Single-file white lane follower. Edit the parameter block below.

Uses the uploaded vehicle calibration: CH8 throttle, CH9 steering,
center=1640us, left=1880us, right=1400us. White stripe is the reference
path as in the original; a road boundary needs a calibrated offset instead.

The default mode only reports the command it would send. Pass ``--drive``
after checking the camera image and steering direction with the wheels raised.
"""

import argparse
import time
from dataclasses import dataclass

import cv2

import numpy as np

# ===== 조절할 제어 파라미터: 수정 후 다시 실행 =====
# 1. 속도: 기존 실행 가능한 PWM 범위에서 먼저 조정
STRAIGHT_SPEED_US = 1640  # 직선 기본값. --speed-us 옵션이 있으면 옵션이 우선
CURVE_SPEED_US = 1600     # 커브/부분 검출 시 목표 저속값
MAX_SPEED_US = 1660       # 실행 옵션 상한. 속도를 올릴 때 별도로 확인

# 2. 정상 차선 조향: 한 번에 한 값씩 변경
KP_LATERAL = 0.014        # 앞쪽 목표점(선 위)의 위치 오차 보정. 시험 간격: 0.001
KP_HEADING = 0.03         # 방향 오차(도 단위) 보정. 크면 선 반대로 꺾음. 시험 간격: 0.01
DEADBAND_PX = 8.0         # 640px 폭 기준, 이보다 작은 위치 오차는 무시
HEADING_DEADBAND_DEG = 1.0  # 이보다 작은 방향 오차(도)는 무시
MAX_ANGULAR = 2.40        # 내부 명령 정규화 상한(실제 바퀴 각도 아님)
                         # 올리면 같은 오차에서 PWM 변화가 작아짐. 우선 고정

# 3. 커브 감속: severity=max(|방향오차|/30도, |조향PWM-중앙|/좌우범위)
TURN_SLOW_START = 0.12    # 이 지표부터 감속 시작. 낮출수록 빨리 감속
TURN_SLOW_FULL = 0.50     # 이 지표에서 CURVE_SPEED_US까지 감속

# 4. 부분 검출: 마지막 정상 조향을 기준으로 제한적 보정
PARTIAL_MAX_CHANGE_US = 25  # 마지막 정상 조향에서 허용할 최대 변화 ±us
PARTIAL_NEW_WEIGHT = 0.25   # 부분 검출 새 조향값 반영 비율(0~1)
PARTIAL_MAX_SEC = 0.8       # 마지막 정상 차선 이후 부분 추종 허용 시간
                           # 부분 선만 보이는 상태에서 새로 출발하지 않음
LANE_CONFIRM_FRAMES = 3     # 정상 차선을 연속 확인한 뒤 출발/재출발

# 5. 차선 유실 탐색: 코너에서 선이 사라지면 직전 조향보다 조금 더 선 쪽으로
LOST_SEARCH_SEC = 1.0       # 마지막 추종 후 이 시간 동안 탐색 주행, 이후 정지
LOST_SEARCH_EXTRA_US = 40   # 직전 조향에 더할 양. 크면 선을 지나치고 작으면 못 따라감
LOST_SEARCH_SPEED_US = 1600  # 탐색 중 속도. CURVE_SPEED_US 이하 권장

# 6. 탐색창: 픽셀 값은 영상 폭에 맞춰 640px 기준에서 자동 비례
SLIDING_WINDOW_MARGIN_PX = 65  # 기본 좌우 탐색 반폭
MAX_WINDOW_MARGIN_PX = 130    # 커브/짧은 공백에서 넓힐 최대 반폭


def validate_control_params():
    import math
    numeric = (
        STRAIGHT_SPEED_US, CURVE_SPEED_US, MAX_SPEED_US,
        KP_LATERAL, KP_HEADING, DEADBAND_PX, HEADING_DEADBAND_DEG,
        MAX_ANGULAR, TURN_SLOW_START, TURN_SLOW_FULL,
        PARTIAL_MAX_CHANGE_US, PARTIAL_NEW_WEIGHT, PARTIAL_MAX_SEC,
        LANE_CONFIRM_FRAMES, SLIDING_WINDOW_MARGIN_PX, MAX_WINDOW_MARGIN_PX,
        LOST_SEARCH_SEC, LOST_SEARCH_SPEED_US, LOST_SEARCH_EXTRA_US,
    )
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in numeric):
        raise ValueError('제어 파라미터는 유한한 숫자여야 합니다')
    if not 1500 <= CURVE_SPEED_US <= STRAIGHT_SPEED_US <= MAX_SPEED_US:
        raise ValueError('1500 <= 커브속도 <= 직선속도 <= 속도상한이어야 합니다')
    if KP_LATERAL < 0 or KP_HEADING < 0 or MAX_ANGULAR <= 0:
        raise ValueError('게인은 0 이상, MAX_ANGULAR는 양수여야 합니다')
    if DEADBAND_PX < 0 or HEADING_DEADBAND_DEG < 0:
        raise ValueError('데드밴드는 0 이상이어야 합니다')
    if not 0 <= TURN_SLOW_START < TURN_SLOW_FULL <= 1:
        raise ValueError('0 <= 감속시작 < 최대감속 <= 1이어야 합니다')
    if not 0 <= PARTIAL_NEW_WEIGHT <= 1 or PARTIAL_MAX_CHANGE_US < 0 or PARTIAL_MAX_SEC <= 0:
        raise ValueError('부분 검출 제어값 범위를 확인하세요')
    if not isinstance(LANE_CONFIRM_FRAMES, int) or LANE_CONFIRM_FRAMES < 1:
        raise ValueError('LANE_CONFIRM_FRAMES는 1 이상의 정수여야 합니다')
    if LOST_SEARCH_EXTRA_US < 0:
        raise ValueError('LOST_SEARCH_EXTRA_US는 0 이상이어야 합니다')
    if LOST_SEARCH_SEC < 0 or not 1500 <= LOST_SEARCH_SPEED_US <= MAX_SPEED_US:
        raise ValueError('탐색 시간은 0 이상, 탐색 속도는 1500~속도상한이어야 합니다')
    if not 0 < SLIDING_WINDOW_MARGIN_PX <= MAX_WINDOW_MARGIN_PX:
        raise ValueError('0 < 기본 탐색 반폭 <= 최대 탐색 반폭이어야 합니다')
# ===== 아래는 주행 코드 =====



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
STEERING_CENTER_US = 1640
STEERING_LEFT_US = 1880
STEERING_RIGHT_US = 1400

MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06

CAMERA_TOPIC = '/camera/image_raw'
IMAGE_TIMEOUT_SEC = 0.25
MAX_IMAGE_AGE_SEC = 0.20
SLIDING_WINDOW_COUNT = 10
ROI_TOP_RATIO = 0.50
SLIDING_WINDOW_MIN_PIXELS = 20


WHITE_LOWER_HSV = (0, 0, 140)
WHITE_UPPER_HSV = (179, 45, 255)
# A stripe must also be brighter than its surroundings (top-hat on V), which
# rejects the wide bright floor beside the track but keeps dim, blurred stripes.
WHITE_TOPHAT_MIN = 60
WHITE_TOPHAT_KERNEL_PX = 41
YELLOW_LOWER_HSV = (0, 59, 220)
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
    partial: bool


def bird_eye_view(image):
    """Transform the road trapezoid into a front-facing rectangular view."""
    height, width = image.shape[:2]
    source = np.float32([
        (int(width * 0.30), int(height * 0.55)),
        (int(width * 0.70), int(height * 0.55)),
        (width - 1, height - 1),
        (0, height - 1),
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


def brightness_tophat(image):
    """Return how much brighter each pixel is than its neighbourhood."""
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (WHITE_TOPHAT_KERNEL_PX, WHITE_TOPHAT_KERNEL_PX))
    return cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)


def lane_color_mask(image, lower_hsv, upper_hsv, tophat=None, tophat_min=0):
    """Return a cleaned rectangular lower-road mask for one HSV range."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )
    if tophat is not None:
        mask[tophat < tophat_min] = 0
    height, width = mask.shape
    roi = np.zeros_like(mask)
    top_left = (0, int(height * ROI_TOP_RATIO))
    bottom_right = (width - 1, height - 1)
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
    """Predict the next stripe position; widen on bends and short gaps."""
    height, width = mask.shape
    boundaries = np.linspace(height, int(height * ROI_TOP_RATIO),
                             SLIDING_WINDOW_COUNT + 1).astype(int)
    base = max(1, int(SLIDING_WINDOW_MARGIN_PX * width / 640))
    maximum = max(base, int(MAX_WINDOW_MARGIN_PX * width / 640))
    nonzero_y, nonzero_x = mask.nonzero()
    windows, centers, selected_indices = [], [], []
    history = []
    misses = 0
    for i in range(SLIDING_WINDOW_COUNT):
        y_high, y_low = int(boundaries[i]), int(boundaries[i+1])
        mid_y = (y_high + y_low) / 2
        predicted = initial_x if not history else history[-1][0]
        shift = 0.0
        if len(history) >= 2:
            x0, y0 = history[-2]
            x1, y1 = history[-1]
            slope = (x1-x0) / (y1-y0)
            shift = float(np.clip(slope * (mid_y-y1), -maximum, maximum))
            predicted += shift
        predicted = float(np.clip(predicted, 0, width-1))
        margin = min(maximum, base + int(abs(shift)) + misses * base)
        x_low, x_high = max(0, int(predicted-margin)), min(width-1, int(predicted+margin))
        windows.append((x_low, y_low, x_high, y_high))
        indices = np.flatnonzero((nonzero_y >= y_low) & (nonzero_y < y_high)
                                & (nonzero_x >= x_low) & (nonzero_x <= x_high))
        # Select one connected x band instead of averaging distinct stripes.
        columns = np.unique(nonzero_x[indices])
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 2)+1) if columns.size else []
        candidates = []
        for group in groups:
            if group[-1]-group[0] > width * 0.18:
                continue
            chosen = indices[(nonzero_x[indices] >= group[0]) & (nonzero_x[indices] <= group[-1])]
            if chosen.size >= SLIDING_WINDOW_MIN_PIXELS:
                candidates.append(chosen)
        if candidates:
            chosen = min(candidates, key=lambda c: abs(np.mean(nonzero_x[c])-predicted))
            center = (int(np.mean(nonzero_x[chosen])), int(np.mean(nonzero_y[chosen])))
            centers.append(center)
            history.append(center)
            selected_indices.append(chosen)
            misses = 0
        else:
            centers.append(None)
            if history:
                misses += 1
            if misses >= 2 and history:
                # Do not bridge large unseen regions onto an unrelated marking.
                break
    selected = np.concatenate(selected_indices) if selected_indices else np.empty(0, dtype=np.int64)
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
    """Accept full tracks or short, bottom-supported partial tracks."""
    height, width = mask.shape
    # Seed from the lowest band that has white, so a stripe that only
    # appears higher up or at the side of the image is still followed.
    # While tracking, look through every band and keep the stripe nearest
    # the previous one instead of jumping to a new mark lower in the image.
    seeds = []
    band_top = int(height * 0.85)
    while band_top >= int(height * ROI_TOP_RATIO):
        histogram = np.count_nonzero(mask[band_top:band_top + int(height * 0.15)], axis=0)
        candidates = np.flatnonzero(histogram >= max(3, int(height * 0.015)))
        if candidates.size:
            groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1)+1)
            seeds += [float(np.mean(g)) for g in groups if g[-1]-g[0] < width*0.18]
            if seeds and previous_x is None:
                break
        band_top -= int(height * 0.075)
    if not seeds:
        return None
    reference = width/2 if previous_x is None else previous_x
    seed = min(seeds, key=lambda x: abs(x-reference))
    track = sliding_window_track(mask, seed)
    valid = np.array([point for point in track[1] if point is not None])
    if len(valid) < 2:
        return None
    top_y, bottom_y = int(valid[:,1].min()), int(valid[:,1].max())
    if bottom_y-top_y < height*0.04:
        return None
    partial = len(valid) < 5 or top_y > height*0.66 or bottom_y < height*0.85
    # Partial tracks use a line and an observed target: no blind extrapolation.
    target_y = max(int(height*0.66), top_y)
    if target_y >= bottom_y:
        target_y = top_y
    coefficients = np.polyfit(valid[:,1]/height, valid[:,0], 1 if partial else 2)
    residual = np.abs(np.polyval(coefficients, valid[:,1]/height)-valid[:,0])
    if residual.max() > width*(0.025 if partial else 0.04):
        return None
    target_x = float(np.polyval(coefficients, target_y/height))
    bottom_x = float(np.polyval(coefficients, bottom_y/height))
    if not (0 <= target_x < width and 0 <= bottom_x < width):
        return None
    angle = 90 + float(np.degrees(np.arctan2(target_x-bottom_x, bottom_y-target_y)))
    ys = np.arange(top_y, bottom_y+1)
    xs = np.clip(np.polyval(coefficients, ys/height), 0, width-1)
    curve = np.column_stack((xs,ys)).astype(np.int32)
    return target_x, angle, target_y, track, curve, bottom_x, partial


def curve_speed_us(max_speed, angle_deg, steering_us):
    """Anticipate bends from heading, even if steering terms cancel."""
    if steering_us >= STEERING_CENTER_US:
        span = STEERING_LEFT_US - STEERING_CENTER_US
    else:
        span = STEERING_CENTER_US - STEERING_RIGHT_US
    severity = max(abs(angle_deg - 90.0) / 30.0,
                   abs(steering_us - STEERING_CENTER_US) / span)
    blend = float(np.clip((severity - TURN_SLOW_START)
                          / (TURN_SLOW_FULL - TURN_SLOW_START), 0.0, 1.0))
    curve_speed = min(max_speed, CURVE_SPEED_US)
    return int(round(max_speed + blend * (curve_speed - max_speed)))


def partial_steering(candidate_us, full_steering_us):
    """Bound uncertain estimates around the last FULL command, not last partial."""
    correction = float(np.clip((candidate_us - full_steering_us) * PARTIAL_NEW_WEIGHT,
                               -PARTIAL_MAX_CHANGE_US, PARTIAL_MAX_CHANGE_US))
    return int(round(np.clip(full_steering_us + correction,
                             STEERING_RIGHT_US, STEERING_LEFT_US)))


def detect_white_lane(image, previous_x=None):
    """Use only the white mask for steering and show yellow as a preview."""
    bev = bird_eye_view(image)
    # Top-hat on the camera image: on the warped image the black area outside
    # the road trapezoid made the floor at its border look like a stripe.
    tophat = bird_eye_view(brightness_tophat(image))
    white_mask = lane_color_mask(bev, WHITE_LOWER_HSV, WHITE_UPPER_HSV,
                                 tophat, WHITE_TOPHAT_MIN)
    yellow_mask = lane_color_mask(bev, YELLOW_LOWER_HSV, YELLOW_UPPER_HSV)
    fit = fit_lane_track(white_mask, previous_x)
    if fit is None:
        return None, bev, white_mask, yellow_mask
    center_x, angle_deg, target_y, track, curve, bottom_x, partial = fit
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
        partial,
    )
    return observation, bev, white_mask, yellow_mask


def write_status_log(logger, text, warning=False):
    """Log each severity from a stable ROS 2 call site."""
    if warning:
        logger.warning(text)
    else:
        logger.info(text)


class LaneController:
    """Near-point lateral plus forward heading; direct calibrated output."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.lateral_term = 0.0
        self.heading_term = 0.0
        self.raw_command = 0.0
        self.command = 0.0
        self.saturated = False

    def steering_pulse(self, near_x, angle_deg, image_width):
        """Positive command turns LEFT; image-right errors turn RIGHT."""
        if image_width <= 0 or not np.isfinite([near_x, angle_deg]).all():
            raise ValueError('Invalid steering observation')
        scale = image_width / 640.0
        lateral_error = near_x - image_width / 2.0
        heading_error = angle_deg - 90.0
        if abs(lateral_error) < DEADBAND_PX * scale:
            lateral_error = 0.0
        if abs(heading_error) < HEADING_DEADBAND_DEG:
            heading_error = 0.0
        self.lateral_term = -KP_LATERAL / scale * lateral_error
        self.heading_term = -KP_HEADING * heading_error
        self.raw_command = self.lateral_term + self.heading_term
        self.command = float(np.clip(self.raw_command, -MAX_ANGULAR, MAX_ANGULAR))
        self.saturated = abs(self.raw_command) > MAX_ANGULAR
        endpoint = STEERING_LEFT_US if self.command >= 0 else STEERING_RIGHT_US
        pulse = STEERING_CENTER_US + (
            abs(self.command) / MAX_ANGULAR * (endpoint - STEERING_CENTER_US)
        )
        return int(round(np.clip(pulse, STEERING_RIGHT_US, STEERING_LEFT_US)))


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
        default=STRAIGHT_SPEED_US,
        help=f'차선 감지 시 속도 펄스(기본 {STRAIGHT_SPEED_US} us)',
    )
    return parser


def main(argv=None):
    """Run the ROS 2 lane follower and always restore safe outputs."""
    parser = build_parser()
    try:
        validate_control_params()
    except ValueError as exc:
        parser.error(str(exc))
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
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image
    except ModuleNotFoundError as exc:
        print(f'ROS 2 모듈을 불러올 수 없습니다: {exc}')
        print('설치한 ROS 2 배포판의 setup.bash를 source 후 다시 실행하세요.')
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
            self.full_frames = 0
            self.previous_lane_x = None
            self.partial_since = None
            self.last_full_at = None
            self.last_full_steering = STEERING_CENTER_US
            self.last_tracking_steering = STEERING_CENTER_US
            self.last_tracking_at = None
            self.last_seen_left = None
            self.subscription = self.create_subscription(
                Image,
                options.camera_topic,
                self.image_callback,
                QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                ),
            )
            self.watchdog = self.create_timer(0.05, self.check_image_timeout)
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
                # Neutral throttle does not instantly remove vehicle momentum.
                # Keep the last accepted steering instead of suddenly centering.
                self.hardware.set_speed(NEUTRAL_US)
                self.hardware.set_steering(self.last_tracking_steering)

        def report(self, key, text, warning=False):
            """Log state changes immediately and steady state once a second."""
            now = time.monotonic()
            if key != self.last_log_key or now - self.last_log_at >= 1.0:
                write_status_log(self.get_logger(), text, warning)
                self.last_log_at = now
                self.last_log_key = key

        def image_callback(self, message):
            """Detect a lane and update steering for one camera image."""
            callback_started = time.monotonic()
            stamp = message.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            # ROS header and node clock must share the same time base.
            frame_age = ((self.get_clock().now().nanoseconds - stamp_ns) / 1e9
                         if stamp_ns else None)
            if frame_age is not None and (frame_age < -0.05 or frame_age > MAX_IMAGE_AGE_SEC):
                self.controller.reset()
                self.previous_lane_x = None
                self.lane_frames = 0
                self.full_frames = 0
                self.apply(STEERING_CENTER_US, False)
                self.report('stale', f'영상 지연/시계 불일치로 정지: age={frame_age:.3f}s', True)
                return
            self.last_image_at = callback_started
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
                self.full_frames = 0
                self.apply(STEERING_CENTER_US, False)
                self.report(
                    'error',
                    f'영상 처리 오류로 정지: {exc}',
                    warning=True,
                )
                return

            processing_sec = time.monotonic() - callback_started
            # Re-check age before sending hardware commands.
            age_before_output = processing_sec + (frame_age or 0.0)
            if age_before_output > MAX_IMAGE_AGE_SEC:
                self.controller.reset()
                self.previous_lane_x = None
                self.lane_frames = 0
                self.full_frames = 0
                self.apply(STEERING_CENTER_US, False)
                self.report('slow_processing',
                            f'처리 지연으로 정지: age={age_before_output:.3f}s', True)
                return

            now = time.monotonic()
            recently_tracked = (self.last_tracking_at is not None
                                and now - self.last_tracking_at < LOST_SEARCH_SEC)
            if observation is None:
                if self.last_full_at is None or now - self.last_full_at > PARTIAL_MAX_SEC:
                    self.previous_lane_x = None
                self.controller.reset()
                self.lane_frames = 0
                self.full_frames = 0
                if recently_tracked and self.last_seen_left is not None:
                    # Corner: the stripe left the view. Keep the last turn and
                    # add a little toward the side it was last seen; full lock
                    # overshoots the stripe.
                    extra = LOST_SEARCH_EXTRA_US if self.last_seen_left else -LOST_SEARCH_EXTRA_US
                    steering_us = int(np.clip(self.last_tracking_steering + extra,
                                              STEERING_RIGHT_US, STEERING_LEFT_US))
                    speed_us = min(LOST_SEARCH_SPEED_US, options.speed_us)
                    moving = True
                    self.apply(steering_us, True, speed_us)
                    self.report(
                        'search',
                        f'차선 유실: {"좌" if self.last_seen_left else "우"}측 탐색 '
                        f'조향={steering_us} us, 속도={speed_us} us',
                        warning=True,
                    )
                else:
                    self.apply(STEERING_CENTER_US, False)
                    self.report(
                        'lost',
                        '차선 유실: 스로틀 중립, 마지막 조향 유지',
                        warning=True,
                    )
                    steering_us = self.last_tracking_steering
                    moving = False
                    speed_us = NEUTRAL_US
            else:
                self.previous_lane_x = observation.bottom_x
                # Aim at a point ahead on the stripe. Steering on the near
                # point plus a large heading term turned away from a stripe
                # that had drifted to the side in a bend. A short stripe cut
                # by the image edge gives an unreliable angle and far point,
                # so steer on its near position only.
                steering_us = self.controller.steering_pulse(
                    observation.bottom_x if observation.partial else observation.center_x,
                    90.0 if observation.partial else observation.angle_deg,
                    image.shape[1],
                )
                self.lane_frames += 1
                # Reacquired during a search: keep driving without re-confirming.
                moving = self.lane_frames >= LANE_CONFIRM_FRAMES or recently_tracked
                speed_us = curve_speed_us(options.speed_us, observation.angle_deg, steering_us)
                if observation.partial:
                    # Follow any visible white stripe at curve speed.
                    self.full_frames = 0
                    if self.partial_since is None:
                        self.partial_since = time.monotonic()
                    speed_us = min(speed_us, CURVE_SPEED_US)
                else:
                    self.partial_since = None
                    self.full_frames += 1
                    moving = self.full_frames >= LANE_CONFIRM_FRAMES or recently_tracked
                    if moving:
                        self.last_full_at = time.monotonic()
                        self.last_full_steering = steering_us
                if not moving:
                    speed_us = NEUTRAL_US
                    steering_us = self.last_tracking_steering
                else:
                    self.last_tracking_steering = steering_us
                    self.last_tracking_at = now
                    self.last_seen_left = observation.bottom_x < image.shape[1] / 2.0
                self.apply(steering_us, moving, speed_us)
                if moving:
                    self.report(
                        'tracking',
                        f'{"PARTIAL" if observation.partial else "FULL"} '
                        f'near_x={observation.bottom_x:.1f}, far_x={observation.center_x:.1f}, '
                        f'각도={observation.angle_deg:.1f}°, '
                        f'lat={self.controller.lateral_term:+.2f}, '
                        f'head={self.controller.heading_term:+.2f}, '
                        f'cmd={self.controller.command:+.2f}, SAT={int(self.controller.saturated)}, '
                        f'조향={steering_us} us, 속도={speed_us} us, '
                        f'처리={processing_sec * 1000:.0f}ms, '
                        f'age_ms={round(age_before_output * 1000) if frame_age is not None else "unknown"}',
                    )
                else:
                    self.report(
                        'confirming',
                        ('부분 차선만으로 출발하지 않음/추종 시간 초과: 전체 차선 재검출 대기'
                         if observation.partial and self.lane_frames >= LANE_CONFIRM_FRAMES
                         else f'차선 연속 감지 확인 중 ({self.lane_frames}/{LANE_CONFIRM_FRAMES})'),
                    )

            if options.show:
                display = bev.copy()
                if observation is not None:
                    display = draw_sliding_windows(
                        display,
                        observation.mask,
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
                    'STOP: FIT FAILED' if observation is None else 'STOP: CONFIRMING'
                )
                if observation is not None and observation.partial:
                    status = 'PARTIAL: SLOW' if moving else 'PARTIAL: WAIT'
                if observation is None and moving:
                    status = 'SEARCH LEFT' if self.last_seen_left else 'SEARCH RIGHT'
                cv2.putText(
                    display,
                    f'{status} steer={steering_us} speed={speed_us} us',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                if observation is not None:
                    cv2.circle(display, (int(observation.bottom_x),
                                         int(observation.curve[-1, 1])),
                               7, (255, 0, 0), -1)
                    cv2.putText(display,
                                f'lat={self.controller.lateral_term:+.2f} '
                                f'head={self.controller.heading_term:+.2f} '
                                f'SAT={int(self.controller.saturated)} '
                                f'proc={processing_sec * 1000:.0f}ms',
                                (20, 95), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1)
                cv2.putText(
                    display,
                    'CYAN windows  RED missing  YELLOW white-line fit  GREEN target',
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
            self.full_frames = 0
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