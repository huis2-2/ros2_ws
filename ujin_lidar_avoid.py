#!/usr/bin/env python3
"""White lane follower that changes to the neighbouring stripe for obstacles.

Lane detection, steering control, calibration and hardware come from
``ujin_versin.py`` (edit lane parameters there). This file adds the LiDAR:
an obstacle within AVOID_TRIGGER_M in the corridor ahead makes the car leave
its stripe, steer to the other white stripe and follow that one instead.

The default mode only reports the command it would send. Pass ``--drive``
after checking the camera image, LiDAR front direction and steering direction
with the wheels raised.
"""

import math
import time

import cv2

import numpy as np

import ujin_versin as lane

# ===== 조절할 LiDAR 회피 파라미터: 수정 후 다시 실행 =====
# 1. 장애물 감지: LiDAR 중심 기준 거리(m). 앞 범퍼까지의 거리는 포함되지 않음
SCAN_TOPIC = '/scan'
FRONT_CENTER_DEG = 180.0      # 차량 정면 방향의 LiDAR 각도(기존 LiDAR 코드와 동일)
AVOID_TRIGGER_M = 0.20        # 진행 통로 안 장애물이 이 거리 이내면 옆 선으로 회피
EMERGENCY_STOP_M = 0.10       # 이 거리 이내면 무조건 정지
CORRIDOR_HALF_WIDTH_M = 0.12  # 진행 통로 반폭. 옆 선 위 물체는 무시되도록 선 간격의 절반 이하
LIDAR_IGNORE_BELOW_M = 0.03   # 차체 자기 반사 무시
OBSTACLE_CONFIRM_SCANS = 2    # 연속 감지 스캔 수(오검출 방지)
CLEAR_CONFIRM_SCANS = 3       # 장애물 해제 확인 스캔 수
SCAN_TIMEOUT_SEC = 0.5        # LiDAR가 이 시간 이상 끊기면 정지

# 2. 회피 주행: 옆 선이 보일 때까지 고정 조향
AVOID_STEER_US = 200          # 중앙 대비 회피 조향량(us). 크면 급하게 꺾음
AVOID_SPEED_US = 1590         # 회피/합류 중 속도
AVOID_MAX_SEC = 2.0           # 이 시간 안에 옆 선을 못 찾으면 정지
LANE_GAP_MIN_PX = 90          # 기존 선과 이만큼 떨어진 흰 선만 옆 선으로 인정(640px 기준)
NEIGHBOR_MIN_Y_RATIO = 0.70   # 옆 선은 화면 아래쪽(이 비율 아래)에서만 찾음

# 3. 새 선 합류: 새 선이 차 중심에 올 때까지 조향 제한
MERGE_MAX_STEER_US = 220      # 합류 중 중앙 대비 최대 조향량(us). 크면 선을 지나침
MERGE_DONE_PX = 30            # 새 선 위치 오차가 이 이하면 일반 추종으로 전환
MERGE_MAX_SEC = 3.0           # 합류 최대 시간, 넘으면 일반 추종으로 전환
SWITCH_COOLDOWN_SEC = 1.5     # 선 변경 직후 다시 장애물이면 또 바꾸지 않고 정지


def validate_avoid_params():
    numeric = (
        FRONT_CENTER_DEG, AVOID_TRIGGER_M, EMERGENCY_STOP_M, CORRIDOR_HALF_WIDTH_M,
        LIDAR_IGNORE_BELOW_M, SCAN_TIMEOUT_SEC, AVOID_STEER_US, AVOID_SPEED_US,
        AVOID_MAX_SEC, LANE_GAP_MIN_PX, NEIGHBOR_MIN_Y_RATIO, MERGE_MAX_STEER_US,
        MERGE_DONE_PX, MERGE_MAX_SEC, SWITCH_COOLDOWN_SEC,
    )
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in numeric):
        raise ValueError('회피 파라미터는 유한한 숫자여야 합니다')
    if not 0 <= LIDAR_IGNORE_BELOW_M < EMERGENCY_STOP_M < AVOID_TRIGGER_M:
        raise ValueError('0 <= 자기반사 무시 < 비상정지 거리 < 회피 거리여야 합니다')
    if CORRIDOR_HALF_WIDTH_M <= 0 or SCAN_TIMEOUT_SEC <= 0:
        raise ValueError('통로 반폭과 LiDAR 타임아웃은 양수여야 합니다')
    for name in ('OBSTACLE_CONFIRM_SCANS', 'CLEAR_CONFIRM_SCANS'):
        value = globals()[name]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name}는 1 이상의 정수여야 합니다')
    if not 0 < AVOID_STEER_US <= lane.STEERING_LEFT_US - lane.STEERING_CENTER_US:
        raise ValueError('AVOID_STEER_US가 조향 범위를 벗어납니다')
    if not 0 < AVOID_STEER_US <= lane.STEERING_CENTER_US - lane.STEERING_RIGHT_US:
        raise ValueError('AVOID_STEER_US가 조향 범위를 벗어납니다')
    if not 0 < MERGE_MAX_STEER_US:
        raise ValueError('MERGE_MAX_STEER_US는 양수여야 합니다')
    if not 1500 <= AVOID_SPEED_US <= lane.MAX_SPEED_US:
        raise ValueError('회피 속도는 1500~속도상한이어야 합니다')
    if AVOID_MAX_SEC <= 0 or MERGE_MAX_SEC <= 0 or SWITCH_COOLDOWN_SEC < 0:
        raise ValueError('회피/합류 시간 값을 확인하세요')
    if LANE_GAP_MIN_PX <= 0 or MERGE_DONE_PX <= 0:
        raise ValueError('픽셀 값은 양수여야 합니다')
    if not lane.ROI_TOP_RATIO <= NEIGHBOR_MIN_Y_RATIO < 1:
        raise ValueError('NEIGHBOR_MIN_Y_RATIO는 ROI 시작~1 사이여야 합니다')
# ===== 아래는 주행 코드 =====


def corridor_distance(scan):
    """Nearest forward distance inside the driving corridor.

    Returns None for an unusable scan and math.inf when the corridor is clear.
    """
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    if (ranges.size == 0 or not math.isfinite(scan.angle_min)
            or not math.isfinite(scan.angle_increment) or scan.angle_increment == 0.0
            or not math.isfinite(scan.range_max)):
        return None
    angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
    relative = angles - math.radians(FRONT_CENTER_DEG)
    valid = (np.isfinite(ranges)
             & (ranges >= max(scan.range_min, LIDAR_IGNORE_BELOW_M))
             & (ranges <= scan.range_max))
    ranges = np.where(valid, ranges, 0.0)
    forward = ranges * np.cos(relative)
    lateral = ranges * np.sin(relative)
    hit = valid & (forward > 0.0) & (np.abs(lateral) <= CORRIDOR_HALF_WIDTH_M)
    return float(forward[hit].min()) if hit.any() else math.inf


def lane_masks(image):
    """Bird's-eye image and white/yellow masks, as in ujin_versin."""
    bev = lane.bird_eye_view(image)
    tophat = lane.bird_eye_view(lane.brightness_tophat(image))
    white = lane.lane_color_mask(bev, lane.WHITE_LOWER_HSV, lane.WHITE_UPPER_HSV,
                                 tophat, lane.WHITE_TOPHAT_MIN)
    yellow = lane.lane_color_mask(bev, lane.YELLOW_LOWER_HSV, lane.YELLOW_UPPER_HSV)
    return bev, white, yellow


def stripe_seeds(mask, min_y_ratio, first_band_only=False):
    """White stripe x positions from the bottom band upward."""
    height, width = mask.shape
    seeds = []
    band_top = int(height * 0.85)
    while band_top >= int(height * min_y_ratio):
        histogram = np.count_nonzero(mask[band_top:band_top + int(height * 0.15)], axis=0)
        candidates = np.flatnonzero(histogram >= max(3, int(height * 0.015)))
        if candidates.size:
            groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1)+1)
            seeds += [float(np.mean(g)) for g in groups if g[-1]-g[0] < width*0.18]
            if seeds and first_band_only:
                break
        band_top -= int(height * 0.075)
    return seeds


def visible_neighbour(seeds, current_x, gap):
    """'left'/'right' if exactly one other stripe side is visible, else None."""
    left = any(x < current_x - gap for x in seeds)
    right = any(x > current_x + gap for x in seeds)
    if left != right:
        return 'left' if left else 'right'
    return None


def neighbour_stripe(seeds, old_x, gap, side):
    """Nearest stripe on the avoidance side of the stripe being left."""
    if side == 'left':
        candidates = [x for x in seeds if x < old_x - gap]
    else:
        candidates = [x for x in seeds if x > old_x + gap]
    if not candidates:
        return None
    return max(candidates) if side == 'left' else min(candidates)


def side_steering(side, amount_us):
    """Steering pulse turned toward one side by amount_us."""
    pulse = lane.STEERING_CENTER_US + (amount_us if side == 'left' else -amount_us)
    return int(np.clip(pulse, lane.STEERING_RIGHT_US, lane.STEERING_LEFT_US))


def format_m(distance):
    if distance is None:
        return '없음'
    return 'clear' if math.isinf(distance) else f'{distance:.2f}m'


def main(argv=None):
    """Run the lane follower with LiDAR lane change; always restore outputs."""
    parser = lane.build_parser()
    parser.description = __doc__
    parser.add_argument('--scan-topic', default=SCAN_TOPIC)
    try:
        lane.validate_control_params()
        validate_avoid_params()
    except ValueError as exc:
        parser.error(str(exc))
    options, ros_args = parser.parse_known_args(argv)
    if not lane.NEUTRAL_US <= options.speed_us <= lane.MAX_SPEED_US:
        parser.error(f'--speed-us는 안전상 {lane.NEUTRAL_US}~{lane.MAX_SPEED_US} 범위만 허용합니다')

    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                               DurabilityPolicy, qos_profile_sensor_data)
        from sensor_msgs.msg import Image, LaserScan
    except ModuleNotFoundError as exc:
        print(f'ROS 2 모듈을 불러올 수 없습니다: {exc}')
        print('설치한 ROS 2 배포판의 setup.bash를 source 후 다시 실행하세요.')
        return 2

    hardware = None
    node = None
    ros_initialized = False

    class LidarLaneFollower(Node):
        """Follow one white stripe; switch to the other one for obstacles."""

        def __init__(self, drive_hardware):
            super().__init__('ujin_lidar_avoid')
            self.hardware = drive_hardware
            self.bridge = CvBridge()
            self.controller = lane.LaneController()
            self.started_at = time.monotonic()
            self.last_image_at = None
            self.last_log_at = 0.0
            self.last_log_key = None
            self.timeout_reported = False
            # Lane tracking (same meaning as in ujin_versin).
            self.lane_frames = 0
            self.full_frames = 0
            self.track_x = None
            self.last_full_at = None
            self.last_tracking_steering = lane.STEERING_CENTER_US
            self.last_tracking_at = None
            self.last_seen_left = None
            # Lane change: follow -> avoid -> merge -> follow, or blocked.
            self.state = 'follow'
            self.state_since = self.started_at
            # Which of the two stripes we are on ('left'/'right'), learned from
            # frames that show both; None until then.
            self.current_side = None
            self.target_side = None
            self.old_x = None
            self.last_switch_at = -math.inf
            # LiDAR.
            self.last_scan_at = None
            self.obstacle_m = None
            self.near_scans = 0
            self.clear_scans = 0
            self.obstacle = False
            self.emergency = False

            self.create_subscription(
                Image, options.camera_topic, self.image_callback,
                QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                           reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.VOLATILE))
            self.create_subscription(LaserScan, options.scan_topic, self.scan_callback,
                                     qos_profile_sensor_data)
            self.watchdog = self.create_timer(0.05, self.check_timeouts)
            mode = '실제 차선 주행' if self.hardware is not None else '판단 로그 시험'
            self.get_logger().info(
                f'{mode}: {options.camera_topic} 흰 선 + {options.scan_topic} '
                f'장애물 {AVOID_TRIGGER_M:.2f} m 이내면 옆 선으로 회피')

        # ----- output -----
        def apply(self, steering_us, moving, speed_us=lane.NEUTRAL_US):
            if self.hardware is None:
                return
            if moving:
                self.hardware.set_steering(steering_us)
                self.hardware.set_speed(speed_us)
            else:
                self.hardware.set_speed(lane.NEUTRAL_US)
                self.hardware.set_steering(self.last_tracking_steering)

        def report(self, key, text, warning=False):
            now = time.monotonic()
            if key != self.last_log_key or now - self.last_log_at >= 1.0:
                lane.write_status_log(self.get_logger(), text, warning)
                self.last_log_at = now
                self.last_log_key = key

        def set_state(self, state):
            self.state = state
            self.state_since = time.monotonic()

        def reset_tracking(self):
            self.controller.reset()
            self.track_x = None
            self.lane_frames = 0
            self.full_frames = 0

        def stop(self, key, text):
            """Neutral throttle, hold steering, require lane re-confirmation."""
            self.lane_frames = 0
            self.full_frames = 0
            self.apply(lane.STEERING_CENTER_US, False)
            self.report(key, text, warning=True)
            return self.last_tracking_steering, lane.NEUTRAL_US, False

        def tracking(self, steering_us, observed_x, width):
            self.last_tracking_steering = steering_us
            self.last_tracking_at = time.monotonic()
            if observed_x is not None:
                self.last_seen_left = observed_x < width / 2.0

        # ----- LiDAR -----
        def scan_callback(self, scan):
            distance = corridor_distance(scan)
            if distance is None:
                return  # unusable scan: the timeout stops the car if this persists
            self.last_scan_at = time.monotonic()
            self.obstacle_m = distance
            if distance <= AVOID_TRIGGER_M:
                self.near_scans += 1
                self.clear_scans = 0
            else:
                self.clear_scans += 1
                self.near_scans = 0
            if self.near_scans >= OBSTACLE_CONFIRM_SCANS:
                self.obstacle = True
            elif self.clear_scans >= CLEAR_CONFIRM_SCANS:
                self.obstacle = False
            self.emergency = distance <= EMERGENCY_STOP_M
            if self.emergency:
                # Do not wait for the next camera frame.
                self.apply(lane.STEERING_CENTER_US, False)

        def check_timeouts(self):
            now = time.monotonic()
            image_ref = self.last_image_at or self.started_at
            scan_ref = self.last_scan_at or self.started_at
            image_lost = now - image_ref > lane.IMAGE_TIMEOUT_SEC
            scan_lost = now - scan_ref > SCAN_TIMEOUT_SEC
            if not (image_lost or scan_lost):
                self.timeout_reported = False
                return
            self.reset_tracking()
            self.apply(lane.STEERING_CENTER_US, False)
            if not self.timeout_reported:
                source = '카메라 영상' if image_lost else 'LiDAR 스캔'
                self.get_logger().error(f'{source}이 끊겨 중립 정지')
                self.timeout_reported = True

        # ----- per-state steering -----
        def follow_step(self, white, now):
            """Original ujin_versin lane-follow logic on the current stripe."""
            width = white.shape[1]
            recently_tracked = (self.last_tracking_at is not None
                                and now - self.last_tracking_at < lane.LOST_SEARCH_SEC)
            fit = lane.fit_lane_track(white, self.track_x)
            if fit is None:
                if self.last_full_at is None or now - self.last_full_at > lane.PARTIAL_MAX_SEC:
                    # The next stripe found may be the other one: forget the side.
                    self.track_x = None
                    self.current_side = None
                self.controller.reset()
                self.lane_frames = 0
                self.full_frames = 0
                if recently_tracked and self.last_seen_left is not None:
                    extra = (lane.LOST_SEARCH_EXTRA_US if self.last_seen_left
                             else -lane.LOST_SEARCH_EXTRA_US)
                    steering_us = int(np.clip(self.last_tracking_steering + extra,
                                              lane.STEERING_RIGHT_US, lane.STEERING_LEFT_US))
                    speed_us = min(lane.LOST_SEARCH_SPEED_US, options.speed_us)
                    self.apply(steering_us, True, speed_us)
                    self.report('search', f'차선 유실: {"좌" if self.last_seen_left else "우"}측 '
                                f'탐색 조향={steering_us} us', warning=True)
                    return fit, steering_us, speed_us, True
                return (fit,) + self.stop('lost', '차선 유실: 스로틀 중립, 마지막 조향 유지')

            center_x, angle_deg, _, _, _, bottom_x, partial = fit
            self.track_x = bottom_x
            seen = visible_neighbour(stripe_seeds(white, NEIGHBOR_MIN_Y_RATIO), bottom_x,
                                     LANE_GAP_MIN_PX * width / 640)
            if seen is not None:
                # The other stripe is on the left, so we are on the right one.
                self.current_side = 'right' if seen == 'left' else 'left'
            steering_us = self.controller.steering_pulse(
                bottom_x if partial else center_x, 90.0 if partial else angle_deg, width)
            self.lane_frames += 1
            moving = self.lane_frames >= lane.LANE_CONFIRM_FRAMES or recently_tracked
            speed_us = lane.curve_speed_us(options.speed_us, angle_deg, steering_us)
            if partial:
                self.full_frames = 0
                speed_us = min(speed_us, lane.CURVE_SPEED_US)
            else:
                self.full_frames += 1
                moving = self.full_frames >= lane.LANE_CONFIRM_FRAMES or recently_tracked
                if moving:
                    self.last_full_at = now
            if not moving:
                self.apply(lane.STEERING_CENTER_US, False)
                self.report('confirming', f'차선 연속 감지 확인 중 '
                            f'({self.lane_frames}/{lane.LANE_CONFIRM_FRAMES})')
                return fit, self.last_tracking_steering, lane.NEUTRAL_US, False
            self.tracking(steering_us, bottom_x, width)
            self.apply(steering_us, True, speed_us)
            self.report('tracking', f'{"PARTIAL" if partial else "FULL"} [{self.current_side or "?"}] '
                        f'near_x={bottom_x:.1f} 각도={angle_deg:.1f}° 조향={steering_us} us '
                        f'속도={speed_us} us 장애물={format_m(self.obstacle_m)}')
            return fit, steering_us, speed_us, True

        def start_avoid(self, white):
            width = white.shape[1]
            gap = LANE_GAP_MIN_PX * width / 640
            seeds = stripe_seeds(white, NEIGHBOR_MIN_Y_RATIO)
            side = visible_neighbour(seeds, self.track_x, gap)
            if side is None and self.current_side is not None:
                side = 'left' if self.current_side == 'right' else 'right'
            if side is None:
                self.get_logger().warning(
                    f'장애물 {format_m(self.obstacle_m)}: 옆 선 방향을 알 수 없어 정지')
                return False
            self.target_side = side
            self.old_x = self.track_x
            self.set_state('avoid')
            self.get_logger().warning(
                f'장애물 {format_m(self.obstacle_m)}: {self.current_side or "?"} 선 -> '
                f'{self.target_side} 선으로 회피 시작')
            return True

        def avoid_step(self, white, now):
            """Steer toward the target side until the other stripe appears."""
            width = white.shape[1]
            scale = width / 640
            gap = LANE_GAP_MIN_PX * scale
            seeds = stripe_seeds(white, NEIGHBOR_MIN_Y_RATIO)
            # Follow where the stripe we are leaving has moved to. If it is
            # lost, keep its last position: it only moves away from the target
            # side while we turn, so it can never pass as the new stripe.
            near = [x for x in seeds if abs(x - self.old_x) < gap / 2]
            if near:
                self.old_x = min(near, key=lambda x: abs(x - self.old_x))
            new_x = neighbour_stripe(seeds, self.old_x, gap, self.target_side)
            if new_x is not None:
                self.reset_tracking()
                self.track_x = new_x
                self.current_side = self.target_side
                self.set_state('merge')
                self.get_logger().info(f'{self.target_side} 선 발견 x={new_x:.0f}: 합류 시작')
                return None
            if now - self.state_since > AVOID_MAX_SEC:
                self.set_state('blocked')
                return self.stop('avoid_timeout', f'{AVOID_MAX_SEC:.1f}초 안에 옆 선을 못 찾아 정지')
            steering_us = side_steering(self.target_side, AVOID_STEER_US)
            speed_us = min(AVOID_SPEED_US, options.speed_us)
            self.tracking(steering_us, None, width)
            self.apply(steering_us, True, speed_us)
            self.report('avoid', f'회피 중: {self.target_side} 조향={steering_us} us, '
                        f'장애물={format_m(self.obstacle_m)}', warning=True)
            return steering_us, speed_us, True

        def merge_step(self, white, now):
            """Follow the new stripe with limited steering until centred on it."""
            width = white.shape[1]
            fit = lane.fit_lane_track(white, self.track_x) if self.track_x is not None else None
            if fit is None:
                if now - self.last_tracking_at > lane.LOST_SEARCH_SEC:
                    self.set_state('blocked')
                    return (None,) + self.stop('merge_lost', '합류 중 새 선 유실: 정지')
                # Keep the last command briefly; the stripe is usually only cut once.
                speed_us = min(AVOID_SPEED_US, options.speed_us)
                self.apply(self.last_tracking_steering, True, speed_us)
                return None, self.last_tracking_steering, speed_us, True
            center_x, angle_deg, _, _, _, bottom_x, partial = fit
            self.track_x = bottom_x
            steering_us = self.controller.steering_pulse(
                bottom_x if partial else center_x, 90.0 if partial else angle_deg, width)
            steering_us = int(np.clip(steering_us,
                                      lane.STEERING_CENTER_US - MERGE_MAX_STEER_US,
                                      lane.STEERING_CENTER_US + MERGE_MAX_STEER_US))
            speed_us = min(AVOID_SPEED_US, options.speed_us)
            self.tracking(steering_us, bottom_x, width)
            self.apply(steering_us, True, speed_us)
            error = abs(bottom_x - lane.LINE_CENTER_X_PX * width / 640)
            self.report('merge', f'합류 중: near_x={bottom_x:.1f}, 오차={error:.0f}px, '
                        f'조향={steering_us} us')
            if error <= MERGE_DONE_PX * width / 640 or now - self.state_since > MERGE_MAX_SEC:
                self.last_switch_at = now
                self.lane_frames = lane.LANE_CONFIRM_FRAMES
                self.full_frames = lane.LANE_CONFIRM_FRAMES
                self.last_full_at = now
                self.set_state('follow')
                self.get_logger().info(f'{self.current_side} 선 합류 완료: 일반 추종')
            return fit, steering_us, speed_us, True

        # ----- camera -----
        def image_callback(self, message):
            callback_started = time.monotonic()
            stamp = message.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            frame_age = ((self.get_clock().now().nanoseconds - stamp_ns) / 1e9
                         if stamp_ns else None)
            if frame_age is not None and (frame_age < -0.05 or frame_age > lane.MAX_IMAGE_AGE_SEC):
                self.reset_tracking()
                self.stop('stale', f'영상 지연/시계 불일치로 정지: age={frame_age:.3f}s')
                return
            self.last_image_at = callback_started
            try:
                image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
                bev, white, yellow = lane_masks(image)
            except Exception as exc:
                self.reset_tracking()
                self.stop('error', f'영상 처리 오류로 정지: {exc}')
                return
            processing_sec = time.monotonic() - callback_started
            if processing_sec + (frame_age or 0.0) > lane.MAX_IMAGE_AGE_SEC:
                self.reset_tracking()
                self.stop('slow_processing', f'처리 지연으로 정지: {processing_sec:.3f}s')
                return

            now = time.monotonic()
            fit = None
            lidar_ok = (self.last_scan_at is not None
                        and now - self.last_scan_at <= SCAN_TIMEOUT_SEC)
            if not lidar_ok:
                steering_us, speed_us, moving = self.stop('no_lidar', 'LiDAR 스캔 없음: 정지')
            elif self.emergency:
                self.set_state('blocked')
                steering_us, speed_us, moving = self.stop(
                    'emergency', f'장애물 {format_m(self.obstacle_m)}: 비상 정지')
            else:
                if self.state == 'blocked' and not self.obstacle:
                    self.set_state('follow')
                if self.state == 'follow' and self.obstacle:
                    driving = (self.last_tracking_at is not None and self.track_x is not None
                               and now - self.last_tracking_at < 0.3)
                    if not (driving and now - self.last_switch_at > SWITCH_COOLDOWN_SEC
                            and self.start_avoid(white)):
                        self.set_state('blocked')
                result = self.avoid_step(white, now) if self.state == 'avoid' else None
                if result is not None:
                    steering_us, speed_us, moving = result
                elif self.state == 'merge':
                    fit, steering_us, speed_us, moving = self.merge_step(white, now)
                elif self.state == 'blocked':
                    steering_us, speed_us, moving = self.stop(
                        'blocked', f'장애물 {format_m(self.obstacle_m)}: 해제될 때까지 정지')
                elif self.state == 'follow':
                    fit, steering_us, speed_us, moving = self.follow_step(white, now)

            if options.show:
                self.show(bev, white, yellow, fit, steering_us, speed_us, moving)

        def show(self, bev, white, yellow, fit, steering_us, speed_us, moving):
            display = bev.copy()
            if fit is not None:
                center_x, _, target_y, track, curve, bottom_x, _ = fit
                display = lane.draw_sliding_windows(display, white, center_x, track)
                cv2.polylines(display, [curve], False, (0, 255, 255), 2)
                cv2.circle(display, (int(center_x), target_y), 7, (0, 255, 0), -1)
                cv2.circle(display, (int(bottom_x), int(curve[-1, 1])), 7, (255, 0, 0), -1)
            if self.state == 'avoid':
                cv2.line(display, (int(self.old_x), display.shape[0] - 1),
                         (int(self.old_x), int(display.shape[0] * 0.7)), (0, 0, 255), 2)
            label = self.state.upper() + ('' if moving else ' STOP')
            cv2.putText(display, f'{label} [{self.current_side or "?"}] steer={steering_us} '
                        f'speed={speed_us}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2)
            cv2.putText(display, f'lidar front={format_m(self.obstacle_m)} '
                        f'trigger={AVOID_TRIGGER_M:.2f}m', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255) if self.obstacle else (255, 255, 255), 1)
            cv2.imshow('lane follow + lidar', display)
            cv2.imshow('white mask - steering input', white)
            cv2.imshow('yellow mask - preview only', yellow)
            cv2.waitKey(1)

    try:
        if options.drive:
            hardware = lane.DriveHardware()
            print('ESC 중립 신호 안정화 중 (3초)')
            time.sleep(3.0)
        rclpy.init(args=ros_args)
        ros_initialized = True
        node = LidarLaneFollower(hardware)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n정지 요청')
    except Exception as exc:
        print(f'실행 오류: {exc}')
        return 1
    finally:
        if hardware is not None:
            hardware.close()
            print(f'ESC 중립 {lane.NEUTRAL_US} us, 조향 중앙 {lane.STEERING_CENTER_US} us')
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if ros_initialized and rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
