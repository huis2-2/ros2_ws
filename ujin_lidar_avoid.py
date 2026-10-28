#!/usr/bin/env python3
"""White lane follower that changes to the neighbouring stripe for obstacles.

Lane detection, steering control, calibration and hardware come from
``ujin_versin.py`` (edit lane parameters there). This file adds the LiDAR:
an obstacle within AVOID_TRIGGER_M in the corridor ahead makes the car leave
its stripe, steer to the other white stripe and follow that one instead.
Obstacles never stop the car; it always tries to drive around them.

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
AVOID_TRIGGER_M = 0.55     # 진행 통로 안 장애물이 이 거리 이내면 옆 선으로 회피(정지 없음)
CORRIDOR_HALF_WIDTH_M = 0.1  # 진행 통로 반폭. 옆 선 위 물체는 무시되도록 선 간격의 절반 이하
LIDAR_IGNORE_BELOW_M = 0.03   # 차체 자기 반사 무시
OBSTACLE_CONFIRM_SCANS = 1    # 연속 감지 스캔 수. 1이면 한 번 감지 즉시 회피
CLEAR_CONFIRM_SCANS = 3       # 장애물 해제 확인 스캔 수
SCAN_TIMEOUT_SEC = 0.5        # LiDAR가 이 시간 이상 끊기면 정지(센서 고장 대비)

# 1-1. 흰 선 인식: 밝은 바닥/장애물이 보여 자동 노출로 영상이 어두워져도 선을 유지
WHITE_V_MIN = 120             # 흰 선 최소 밝기(V). ujin_versin의 172보다 낮춰 어두운 프레임 대응
WHITE_TOPHAT_MIN = 50         # 주변보다 이만큼 밝아야 선. 낮은 V에서 넓은 밝은 바닥을 걸러냄

# 2. 회피 주행: 옆 선이 보일 때까지 고정 조향
AVOID_STEER_RATIO = 0.8       # 회피 조향: 각 방향 최대 조향 대비 비율(0~1). 크면 급하게 꺾음
AVOID_SPEED_US = 1581         # 회피/합류 중 속도
AVOID_MAX_SEC = 2.0           # 이 시간 안에 옆 선을 못 찾으면 보이는 선으로 일반 추종
AVOID_PASS_MARGIN_M = 0.20    # 장애물이 회피 반대쪽으로 이 거리보다 가까우면 지나칠 때까지
                              # 옆 선이 보여도 최대 조향 유지. 차 반폭 + 여유
AVOID_REACH_PX = 40           # 단, 새 선이 차 중심(LINE_CENTER_X_PX)에서 이만큼 안으로 오면
                              # 이미 새 차선이므로 바로 합류. 크면 일찍 풀려 장애물에 가까워짐
RETRIGGER_MIN_M = 0.5         # 회피 시작 후 RETRIGGER_WINDOW_SEC 동안은 이보다 가까운 장애물로
RETRIGGER_WINDOW_SEC = 2.0    # 다시 회피하지 않음(방금 피한 장애물). 새 장애물은 0.5 m 이상에서 잡힘
LANE_GAP_MIN_PX = 90          # 기존 선과 이만큼 떨어진 흰 선만 옆 선으로 인정(640px 기준)
NEIGHBOR_MIN_Y_RATIO = 0.70   # 옆 선은 화면 아래쪽(이 비율 아래)에서만 찾음

# 2-1. 노란 경계선: 장애물 감지 순간 화면에 보이는 노란선의 반대편으로 회피
#      노란선이 안 보이면(오른쪽 선 주행 중에는 노란선이 BEV 밖) 지금 타는 선의
#      반대쪽 선으로 회피. 어느 선인지도 모르면 현재 선 추종을 계속하며 다시 확인
YELLOW_MIN_HEIGHT_RATIO = 0.2  # 노란 덩어리가 BEV 높이의 이 비율 이상 길어야 선. 박스 조각은 짧음
YELLOW_MAX_WIDTH_PX = 30      # 노란 덩어리 평균 폭(640px 기준) 상한. 넓으면 박스 면

# 3. 새 선 합류: 선까지 남은 거리에 맞춘 진입 각도로 다가가 선을 지나치지 않게 함
MERGE_MAX_STEER_RATIO = 0.92  # 합류 중 각 방향 최대 조향 대비 상한 비율(0~1)
MERGE_APPROACH_DEG_PER_PX = 0.15  # 새 선까지 1px당 허용 진입 각도(도). 크면 늦게 펴서 지나침
MERGE_APPROACH_MAX_DEG = 15.0  # 새 선에 다가가는 최대 진입 각도(도)
MERGE_HEADING_KP = 0.08       # 진입 각도 오차(도)당 조향 명령. 크면 좌우로 흔들림
MERGE_DONE_PX = 30            # 새 선 위치 오차가 이 이하이고
MERGE_DONE_DEG = 10.0         # 선 각도가 90°에서 이 이내면 일반 추종으로 전환
MERGE_MAX_SEC = 3.0           # 합류 최대 시간, 넘으면 일반 추종으로 전환


def validate_avoid_params():
    numeric = (
        FRONT_CENTER_DEG, AVOID_TRIGGER_M, CORRIDOR_HALF_WIDTH_M,
        LIDAR_IGNORE_BELOW_M, SCAN_TIMEOUT_SEC,
        WHITE_V_MIN, WHITE_TOPHAT_MIN,
        AVOID_STEER_RATIO, AVOID_SPEED_US, AVOID_MAX_SEC, AVOID_PASS_MARGIN_M,
        AVOID_REACH_PX, RETRIGGER_MIN_M, RETRIGGER_WINDOW_SEC,
        YELLOW_MIN_HEIGHT_RATIO, YELLOW_MAX_WIDTH_PX,
        LANE_GAP_MIN_PX,
        NEIGHBOR_MIN_Y_RATIO,
        MERGE_MAX_STEER_RATIO, MERGE_APPROACH_DEG_PER_PX, MERGE_APPROACH_MAX_DEG,
        MERGE_HEADING_KP, MERGE_DONE_PX, MERGE_DONE_DEG, MERGE_MAX_SEC,
    )
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in numeric):
        raise ValueError('회피 파라미터는 유한한 숫자여야 합니다')
    if not 0 <= LIDAR_IGNORE_BELOW_M < AVOID_TRIGGER_M:
        raise ValueError('0 <= 자기반사 무시 거리 < 회피 거리여야 합니다')
    if not 0 <= RETRIGGER_MIN_M < AVOID_TRIGGER_M or RETRIGGER_WINDOW_SEC < 0:
        raise ValueError('0 <= RETRIGGER_MIN_M < 회피 거리, RETRIGGER_WINDOW_SEC >= 0이어야 합니다')
    if not 0 < YELLOW_MIN_HEIGHT_RATIO <= 1 or YELLOW_MAX_WIDTH_PX <= 0:
        raise ValueError('노란선 모양 기준은 0 < 높이 비율 <= 1, 폭 > 0이어야 합니다')
    if AVOID_PASS_MARGIN_M < 0:
        raise ValueError('AVOID_PASS_MARGIN_M은 0 이상이어야 합니다')
    if CORRIDOR_HALF_WIDTH_M <= 0 or SCAN_TIMEOUT_SEC <= 0:
        raise ValueError('통로 반폭과 LiDAR 타임아웃은 양수여야 합니다')
    for name in ('OBSTACLE_CONFIRM_SCANS', 'CLEAR_CONFIRM_SCANS'):
        value = globals()[name]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name}는 1 이상의 정수여야 합니다')
    if not 0 < AVOID_STEER_RATIO <= 1 or not 0 < MERGE_MAX_STEER_RATIO <= 1:
        raise ValueError('AVOID_STEER_RATIO와 MERGE_MAX_STEER_RATIO는 0~1이어야 합니다')
    if not 0 <= WHITE_V_MIN <= 255 or not 0 <= WHITE_TOPHAT_MIN <= 255:
        raise ValueError('흰 선 밝기 값은 0~255여야 합니다')
    if (MERGE_APPROACH_DEG_PER_PX <= 0 or MERGE_APPROACH_MAX_DEG <= 0
            or MERGE_HEADING_KP <= 0 or MERGE_DONE_DEG <= 0):
        raise ValueError('합류 각도 파라미터는 양수여야 합니다')
    if not 1500 <= AVOID_SPEED_US <= lane.MAX_SPEED_US:
        raise ValueError('회피 속도는 1500~속도상한이어야 합니다')
    if AVOID_MAX_SEC <= 0 or MERGE_MAX_SEC <= 0:
        raise ValueError('회피/합류 시간 값을 확인하세요')
    if LANE_GAP_MIN_PX <= 0 or MERGE_DONE_PX <= 0 or AVOID_REACH_PX < 0:
        raise ValueError('픽셀 값은 양수여야 합니다')
    if not lane.ROI_TOP_RATIO <= NEIGHBOR_MIN_Y_RATIO < 1:
        raise ValueError('NEIGHBOR_MIN_Y_RATIO는 ROI 시작~1 사이여야 합니다')
# ===== 아래는 주행 코드 =====


def scan_points(scan):
    """Valid scan points as (forward, lateral) metres; lateral > 0 is left.

    Returns None for an unusable scan.
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
    ranges, relative = ranges[valid], relative[valid]
    return ranges * np.cos(relative), ranges * np.sin(relative)


def corridor_distance(points):
    """Nearest forward distance inside the driving corridor, inf if clear."""
    forward, lateral = points
    hit = (forward > 0.0) & (np.abs(lateral) <= CORRIDOR_HALF_WIDTH_M)
    return float(forward[hit].min()) if hit.any() else math.inf


def obstacle_not_passed(points, side):
    """True while the avoided obstacle is still ahead of the LiDAR.

    Only points from the far side of the lane change up to the corridor edge
    on the target side count, so an obstacle in the target lane further ahead
    does not hold the turn.
    """
    forward, lateral = points
    toward = lateral if side == 'left' else -lateral
    hit = ((forward > 0.0) & (forward <= AVOID_TRIGGER_M)
           & (toward >= -AVOID_PASS_MARGIN_M) & (toward <= CORRIDOR_HALF_WIDTH_M))
    return bool(hit.any())


def stripe_reached(new_x, side, width):
    """True once the new stripe has come under the car centre.

    While turning left the new stripe slides right across the image (and the
    reverse for a right turn); holding full lock past it leaves the track.
    """
    centre = lane.LINE_CENTER_X_PX * width / 640
    reach = AVOID_REACH_PX * width / 640
    return new_x >= centre - reach if side == 'left' else new_x <= centre + reach


def lane_masks(image):
    """Bird's-eye image and white/yellow masks.

    White uses a lower brightness than ujin_versin plus the top-hat test, so
    a stripe stays visible when auto exposure darkens the frame.
    """
    bev = lane.bird_eye_view(image)
    tophat = lane.bird_eye_view(lane.brightness_tophat(image))
    white_lower = (lane.WHITE_LOWER_HSV[0], lane.WHITE_LOWER_HSV[1], WHITE_V_MIN)
    white = lane.lane_color_mask(bev, white_lower, lane.WHITE_UPPER_HSV,
                                 tophat, WHITE_TOPHAT_MIN)
    yellow = lane.lane_color_mask(
        bev, lane.YELLOW_LOWER_HSV, lane.YELLOW_UPPER_HSV, apply_roi=False)
    return bev, white, yellow


def stripe_seeds(mask, min_y_ratio):
    """White stripe x positions from the bottom band upward."""
    height, width = mask.shape
    seeds = []
    band_top = int(height * 0.85)
    min_y = int(height * min_y_ratio)
    step = max(1, int(height * 0.075))
    band_tops = list(range(band_top, min_y - 1, -step))
    if not band_tops or band_tops[-1] != min_y:
        band_tops.append(min_y)
    for band_top in band_tops:
        histogram = np.count_nonzero(mask[band_top:band_top + int(height * 0.15)], axis=0)
        candidates = np.flatnonzero(histogram >= max(3, int(height * 0.015)))
        if candidates.size:
            groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1)+1)
            seeds += [float(np.mean(g)) for g in groups if g[-1]-g[0] < width*0.18]
    return seeds


def visible_neighbour(seeds, current_x, gap):
    """'left'/'right' if exactly one other stripe side is visible, else None."""
    left = any(x < current_x - gap for x in seeds)
    right = any(x > current_x + gap for x in seeds)
    if left != right:
        return 'left' if left else 'right'
    return None


def visible_yellow_side(mask):
    """Return the screen side of a visible yellow stripe, or ``None``.

    The full height of the bird's-eye image is searched without a yellow-line
    ROI. Direction is based on the image centre, not the followed white stripe.
    The largest long, thin yellow blob is the stripe: specks and pieces of the
    beige obstacle box are short, and a near box face is wide.
    """
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blob_h = stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]
    stripe = ((blob_h >= height * YELLOW_MIN_HEIGHT_RATIO)
              & (area <= blob_h * YELLOW_MAX_WIDTH_PX * width / 640))
    if not stripe.any():
        return None
    largest = 1 + int(np.argmax(np.where(stripe, area, -1)))
    x = stats[largest, cv2.CC_STAT_LEFT] + stats[largest, cv2.CC_STAT_WIDTH] / 2.0
    return 'left' if x < width / 2.0 else 'right'


def yellow_avoidance_side(mask):
    """Return the direction away from a visible yellow stripe, or ``None``."""
    yellow_side = visible_yellow_side(mask)
    if yellow_side is None:
        return None
    return 'right' if yellow_side == 'left' else 'left'


def avoidance_side(mask, current_side):
    """Return (direction, reason) for a lane change, or (None, None).

    Yellow decides when visible. Otherwise move to the other stripe: on the
    right stripe the yellow line is far right, above the bird's-eye view, and
    waiting for it drove the car into the obstacle.
    """
    side = yellow_avoidance_side(mask)
    if side is not None:
        return side, f'가까운 노란선이 {"left" if side == "right" else "right"}쪽'
    if current_side in ('left', 'right'):
        return ('right' if current_side == 'left' else 'left'), '노란선 없음, 반대쪽 선'
    return None, None


def neighbour_stripe(seeds, old_x, gap, side):
    """Nearest stripe on the avoidance side of the stripe being left."""
    if side == 'left':
        candidates = [x for x in seeds if x < old_x - gap]
    else:
        candidates = [x for x in seeds if x > old_x + gap]
    if not candidates:
        return None
    return max(candidates) if side == 'left' else min(candidates)


def tracked_stripe(seeds, last_x, gap):
    """Seed nearest last_x if it moved less than gap, else None."""
    near = [x for x in seeds if abs(x - last_x) < gap]
    return min(near, key=lambda x: abs(x - last_x)) if near else None


def side_steering(side, ratio):
    """Steering pulse turned toward one side by a fraction of that side's lock.

    The left and right ranges differ in microseconds, so a fixed offset would
    turn one way less sharply than the other.
    """
    end = lane.STEERING_LEFT_US if side == 'left' else lane.STEERING_RIGHT_US
    pulse = lane.STEERING_CENTER_US + ratio * (end - lane.STEERING_CENTER_US)
    return int(round(np.clip(pulse, lane.STEERING_RIGHT_US, lane.STEERING_LEFT_US)))


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
            # Lane change: follow -> avoid -> merge -> follow. Obstacles never stop the car.
            self.state = 'follow'
            self.state_since = self.started_at
            # Which of the two stripes we are on ('left'/'right'), learned from
            # frames that show both; None until then.
            self.current_side = None
            self.target_side = None
            self.old_x = None
            self.new_x = None
            self.avoid_started_at = None
            # LiDAR.
            self.last_scan_at = None
            self.points = None
            self.obstacle_m = None
            self.near_scans = 0
            self.clear_scans = 0
            self.obstacle = False

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
            points = scan_points(scan)
            if points is None:
                return  # unusable scan: the timeout stops the car if this persists
            distance = corridor_distance(points)
            self.points = points
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

        def start_avoid(self, side, reason):
            self.target_side = side
            self.old_x = self.track_x
            self.new_x = None
            self.set_state('avoid')
            self.avoid_started_at = self.state_since
            self.get_logger().warning(
                f'장애물 {format_m(self.obstacle_m)}: {self.current_side or "?"} 선 -> '
                f'{self.target_side} 선으로 회피 시작 ({reason})')

        def avoid_step(self, white, now):
            """Steer toward the target side until the other stripe appears."""
            width = white.shape[1]
            scale = width / 640
            gap = LANE_GAP_MIN_PX * scale
            seeds = stripe_seeds(white, NEIGHBOR_MIN_Y_RATIO)
            if self.new_x is None:
                # Follow where the stripe we are leaving has moved to; if it is
                # lost, keep its last position.
                near = [x for x in seeds if abs(x - self.old_x) < gap / 2]
                if near:
                    self.old_x = min(near, key=lambda x: abs(x - self.old_x))
                new_x = neighbour_stripe(seeds, self.old_x, gap, self.target_side)
            else:
                # Once seen, follow the new stripe itself. The stripe being left
                # often drops out of view mid-turn, freezing old_x near the
                # centre; the gap test would then reject the new stripe as it
                # slides in, and old_x would jump onto it.
                new_x = tracked_stripe(seeds, self.new_x, gap)
            if new_x is not None:
                self.new_x = new_x
            # Merging limits the approach angle, so keep full lock until the
            # obstacle has passed the LiDAR even if the new stripe is visible,
            # but no further than onto the new stripe.
            passing = obstacle_not_passed(self.points, self.target_side)
            if new_x is not None and (
                    not passing or stripe_reached(new_x, self.target_side, width)):
                self.reset_tracking()
                self.track_x = new_x
                self.current_side = self.target_side
                self.set_state('merge')
                self.get_logger().info(f'{self.target_side} 선 발견 x={new_x:.0f}: 합류 시작')
                return None
            if now - self.state_since > AVOID_MAX_SEC:
                # Keep driving: follow whatever stripe is nearest the car centre.
                self.get_logger().warning(
                    f'{AVOID_MAX_SEC:.1f}초 안에 옆 선을 못 찾음: 보이는 선으로 일반 추종')
                self.reset_tracking()
                self.current_side = None
                self.set_state('follow')
                return None
            steering_us = side_steering(self.target_side, AVOID_STEER_RATIO)
            speed_us = min(AVOID_SPEED_US, options.speed_us)
            self.tracking(steering_us, None, width)
            self.apply(steering_us, True, speed_us)
            self.report('avoid', f'회피 중: {self.target_side} 조향={steering_us} us, '
                        f'장애물={format_m(self.obstacle_m)}'
                        f'{", 통과 대기" if passing else ""}', warning=True)
            return steering_us, speed_us, True

        def merge_step(self, white, now):
            """Follow the new stripe with limited steering until centred on it."""
            width = white.shape[1]
            fit = lane.fit_lane_track(white, self.track_x) if self.track_x is not None else None
            if fit is None:
                if now - self.last_tracking_at > lane.LOST_SEARCH_SEC:
                    # Hand over to normal following and its lost-lane search.
                    self.set_state('follow')
                    return self.follow_step(white, now)
                # Keep the last command briefly; the stripe is usually only cut once.
                speed_us = min(AVOID_SPEED_US, options.speed_us)
                self.apply(self.last_tracking_steering, True, speed_us)
                return None, self.last_tracking_steering, speed_us, True
            _, angle_deg, _, _, _, bottom_x, _ = fit
            self.track_x = bottom_x
            # Approach the stripe at an angle that shrinks with the remaining
            # distance, so the car is already straightening when it reaches it.
            offset_px = (lane.LINE_CENTER_X_PX * width / 640 - bottom_x) * 640 / width
            heading_deg = angle_deg - 90.0  # > 0: car points left of the stripe
            wanted_deg = float(np.clip(offset_px * MERGE_APPROACH_DEG_PER_PX,
                                       -MERGE_APPROACH_MAX_DEG, MERGE_APPROACH_MAX_DEG))
            command = MERGE_HEADING_KP * (wanted_deg - heading_deg)  # > 0 turns left
            endpoint = lane.STEERING_LEFT_US if command >= 0 else lane.STEERING_RIGHT_US
            steering_us = lane.STEERING_CENTER_US + (
                min(abs(command), lane.MAX_ANGULAR) / lane.MAX_ANGULAR
                * (endpoint - lane.STEERING_CENTER_US))
            steering_us = int(round(np.clip(steering_us,
                                            side_steering('right', MERGE_MAX_STEER_RATIO),
                                            side_steering('left', MERGE_MAX_STEER_RATIO))))
            speed_us = min(AVOID_SPEED_US, options.speed_us)
            self.tracking(steering_us, bottom_x, width)
            self.apply(steering_us, True, speed_us)
            error = abs(offset_px)
            self.report('merge', f'합류 중: near_x={bottom_x:.1f}, 오차={error:.0f}px, '
                        f'각도={heading_deg:+.1f}°(목표 {wanted_deg:+.1f}°), '
                        f'조향={steering_us} us')
            aligned = error <= MERGE_DONE_PX and abs(heading_deg) <= MERGE_DONE_DEG
            if aligned or now - self.state_since > MERGE_MAX_SEC:
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
            else:
                # A merge can run into the next obstacle in the new lane, so it
                # may be interrupted by a fresh lane change as well.
                # self.obstacle stays set for CLEAR_CONFIRM_SCANS after the
                # obstacle is gone, so also require it to be within range now.
                near = self.obstacle_m is not None and self.obstacle_m <= AVOID_TRIGGER_M
                approaching = near and (self.obstacle if self.state == 'follow'
                                        else self.state == 'merge')
                # Right after a lane change the obstacle just avoided can still
                # be close ahead; a new one is first seen near AVOID_TRIGGER_M.
                if (approaching and self.avoid_started_at is not None
                        and now - self.avoid_started_at < RETRIGGER_WINDOW_SEC
                        and self.obstacle_m < RETRIGGER_MIN_M):
                    approaching = False
                if approaching:
                    side, reason = avoidance_side(yellow, self.current_side)
                    if side is None:
                        # No direction yet: keep following the current stripe
                        # and re-check next frame.
                        self.get_logger().warning(
                            '장애물 감지, 노란선도 현재 선도 모름: 회피 보류, 차선 추종 유지',
                            throttle_duration_sec=1.0)
                    else:
                        # Change lanes only after the current stripe is being tracked.
                        driving = (self.last_tracking_at is not None and self.track_x is not None
                                   and now - self.last_tracking_at < 0.3)
                        if driving:
                            self.start_avoid(side, reason)
                result = self.avoid_step(white, now) if self.state == 'avoid' else None
                if result is not None:
                    steering_us, speed_us, moving = result
                elif self.state == 'merge':
                    fit, steering_us, speed_us, moving = self.merge_step(white, now)
                else:
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
