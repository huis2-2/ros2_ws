#!/usr/bin/env python3
"""Two-obstacle, two-white-stripe avoidance for ujin_versin.py (ROS 2).

Follow outer -> detect obstacle <=0.40m -> steer toward inner -> merge with
P control -> follow inner -> next obstacle -> return outer -> follow outer.

Dry-run by default. --drive enables PCA9685 output. Place this alongside
ujin_versin.py. Calibrate camera, LiDAR heading and actual steering first.
"""
import math
import time
import cv2
import numpy as np
import ujin_versin as lane

SCAN_TOPIC = '/scan'
FRONT_CENTER_DEG = 180.0  # verify with /scan; must point toward vehicle front
AVOID_TRIGGER_M = 0.40
CORRIDOR_HALF_WIDTH_M = 0.12
LIDAR_IGNORE_BELOW_M = 0.03
SCAN_TIMEOUT_SEC = 0.5
OBSTACLE_CONFIRM_SCANS = 2
OBSTACLE_REARM_M = 0.55       # require obstacle cleared before next trigger
CLEAR_CONFIRM_SCANS = 3

# The relative position of the other white stripe decides direction.
LINE_GAP_MIN_PX = 90        # calibrated for a 640-pixel BEV
SEED_MIN_Y_RATIO = 0.70
SEED_BAND_RATIO = 0.15
SEED_STEP_RATIO = 0.075
TARGET_CONFIRM_FRAMES = 2
# Full lock is used only during early lane change. Reduce it before crossing
# the destination stripe; use base LaneController P for final alignment.
MAX_STEER_STAGE_SEC = 0.65    # failsafe upper bound; not a guaranteed maneuver duration
APPROACH_MARGIN_PX = 95      # target proximity threshold, /640 width
MERGE_DONE_PX = 30
MERGE_HEADING_TOL_DEG = 12.0
MERGE_CONFIRM_FRAMES = 3
MERGE_MAX_SEC = 2.5
AVOID_MAX_SEC = 1.5
AVOID_SPEED_US = 1585
MERGE_SPEED_US = 1585
TARGET_LOST_SEC = 0.35
# A new obstacle may be recognized only after the first change completes and
# the forward LiDAR corridor has been clear beyond the rearm distance.


def scan_points(scan):
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    if (ranges.size == 0 or not np.isfinite(scan.angle_min)
            or not np.isfinite(scan.angle_increment) or scan.angle_increment == 0
            or not np.isfinite(scan.range_max)):
        return None
    angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
    relative = angles - math.radians(FRONT_CENTER_DEG)
    valid = (np.isfinite(ranges) &
             (ranges >= max(scan.range_min, LIDAR_IGNORE_BELOW_M)) &
             (ranges <= scan.range_max))
    return ranges[valid] * np.cos(relative[valid]), ranges[valid] * np.sin(relative[valid])


def front_distance(points):
    fwd, lat = points
    hit = (fwd > 0) & (np.abs(lat) <= CORRIDOR_HALF_WIDTH_M)
    return float(fwd[hit].min()) if hit.any() else math.inf


def get_masks(image):
    bev = lane.bird_eye_view(image)
    top = lane.bird_eye_view(lane.brightness_tophat(image))
    white = lane.lane_color_mask(bev, lane.WHITE_LOWER_HSV, lane.WHITE_UPPER_HSV,
                                 top, lane.WHITE_TOPHAT_MIN)
    yellow = lane.lane_color_mask(
        bev, lane.YELLOW_LOWER_HSV, lane.YELLOW_UPPER_HSV, apply_roi=False)
    return bev, white, yellow


def stripe_seeds(mask, min_y_ratio=SEED_MIN_Y_RATIO):
    """Group bright stripe x-coordinates from the lower BEV region."""
    h, w = mask.shape
    seeds = []
    y = int(h * .85)
    min_y = int(h * min_y_ratio)
    step = max(1, int(h * SEED_STEP_RATIO))
    band_h = max(1, int(h * SEED_BAND_RATIO))
    band_tops = list(range(y, min_y - 1, -step))
    if not band_tops or band_tops[-1] != min_y:
        band_tops.append(min_y)
    for y in band_tops:
        counts = np.count_nonzero(mask[y:min(h, y+band_h)], axis=0)
        xs = np.flatnonzero(counts >= max(3, int(h * .015)))
        if xs.size:
            for group in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1):
                if group.size and group[-1] - group[0] < w * .18:
                    seeds.append(float(np.mean(group)))
    # Adjacent bands of the same curved stripe should not become fake lanes.
    unique = []
    for x in sorted(seeds):
        if not unique or x - unique[-1][-1] > w * .035:
            unique.append([x])
        else:
            unique[-1].append(x)
    return [float(np.median(g)) for g in unique]


def choose_target(white_mask, yellow_mask, current_x, direction=None):
    """Choose avoidance direction from the yellow line's screen side first.

    A yellow line on the screen right selects LEFT and one on the screen left
    selects RIGHT, regardless of the followed white stripe position.
    Yellow supplies direction evidence only, never a white target position.
    White-stripe evidence is used only when no yellow line is visible.
    """
    if current_x is None:
        return None, None, 'current stripe unknown'
    w = white_mask.shape[1]
    gap = LINE_GAP_MIN_PX * w / 640.
    # Reject pixels near the followed stripe and use comparable separation
    # thresholds for both mask colors.
    white_seeds = stripe_seeds(white_mask)
    yellow_seeds = stripe_seeds(yellow_mask, min_y_ratio=0.0)
    left_white = [x for x in white_seeds if x < current_x - gap]
    right_white = [x for x in white_seeds if x > current_x + gap]
    screen_center = w / 2.0
    if yellow_seeds:
        yellow_x = min(yellow_seeds, key=lambda x: abs(x - screen_center))
        side = 'right' if yellow_x < screen_center else 'left'
        candidates = left_white if side == 'left' else right_white
        target = (max(candidates) if side == 'left' else min(candidates)) if candidates else None
        return side, target, 'white+yellow' if candidates else 'yellow'
    left_evidence = bool(left_white)
    right_evidence = bool(right_white)
    if left_evidence and right_evidence:
        return None, None, 'conflicting left/right boundary evidence'
    if not left_evidence and not right_evidence:
        return None, None, 'no directional stripe evidence'
    side = 'left' if left_evidence else 'right'
    if direction is not None and side != direction:
        return None, None, 'direction disagrees with prior lane change'
    # Require an actual white target before MERGE; yellow justifies the
    # initial steering direction but is never tracked as the target line.
    candidates = left_white if side == 'left' else right_white
    target = (max(candidates) if side == 'left' else min(candidates)) if candidates else None
    return side, target, 'white'


def steer_max(side):
    return lane.STEERING_LEFT_US if side == 'left' else lane.STEERING_RIGHT_US


def validate():
    lane.validate_control_params()
    if not 0 < AVOID_TRIGGER_M < OBSTACLE_REARM_M:
        raise ValueError('Rearm distance must be larger than trigger distance')
    if not 0 < CORRIDOR_HALF_WIDTH_M or not 0 < SCAN_TIMEOUT_SEC:
        raise ValueError('LiDAR corridor and timeout must be positive')
    if not lane.NEUTRAL_US <= AVOID_SPEED_US <= lane.MAX_SPEED_US:
        raise ValueError('Avoid speed outside throttle limits')
    if not lane.NEUTRAL_US <= MERGE_SPEED_US <= lane.MAX_SPEED_US:
        raise ValueError('Merge speed outside throttle limits')
    if not 0 < SEED_MIN_Y_RATIO < 1:
        raise ValueError('Invalid stripe seed ROI')


def main(argv=None):
    parser = lane.build_parser()
    parser.description = __doc__
    parser.add_argument('--scan-topic', default=SCAN_TOPIC)
    options, ros_args = parser.parse_known_args(argv)
    try:
        validate()
    except ValueError as e:
        parser.error(str(e))
    if not lane.NEUTRAL_US <= options.speed_us <= lane.MAX_SPEED_US:
        parser.error('Invalid --speed-us value')

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan
        from cv_bridge import CvBridge
    except ImportError as e:
        print('ROS 2 dependencies missing:', e)
        return 2

    class AvoidNode(Node):
        def __init__(self, hw):
            super().__init__('ujin_lidar_avoid_v2')
            self.hw = hw
            self.bridge = CvBridge()
            self.ctrl = lane.LaneController()
            self.state = 'FOLLOW_OUTER'
            self.state_at = time.monotonic()
            self.boot_at = self.state_at
            self.last_scan_at = None
            self.last_image_at = None
            self.obstacle_dist = math.inf
            self.hit_scans = 0
            self.clear_scans = 0
            self.armed = True
            self.change_count = 0
            self.track_x = None
            self.current_x = None
            self.target_x = None
            self.direction = None
            self.target_seen_at = None
            self.candidate_side = None
            self.candidate_count = 0
            self.target_source = None
            self.merge_ok_count = 0
            self.stopped_reason = None
            self.last_steer = lane.STEERING_CENTER_US
            self.last_log_at = 0
            camera_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                    reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(Image, options.camera_topic, self.on_image, camera_qos)
            self.create_subscription(LaserScan, options.scan_topic, self.on_scan, qos_profile_sensor_data)
            self.create_timer(.05, self.watchdog)
            self.get_logger().info('DRIVE enabled' if hw else 'DRY RUN: no motor output')

        def log(self, message, force=False):
            now = time.monotonic()
            if force or now - self.last_log_at >= 1:
                self.get_logger().info(message)
                self.last_log_at = now

        def command(self, steer, speed):
            self.last_steer = int(np.clip(steer, lane.STEERING_RIGHT_US, lane.STEERING_LEFT_US))
            if self.hw is not None:
                self.hw.set_steering(self.last_steer)
                self.hw.set_speed(int(speed))

        def stop(self, reason):
            self.stopped_reason = reason
            self.state = 'STOP'
            self.command(lane.STEERING_CENTER_US, lane.NEUTRAL_US)
            self.get_logger().error('STOP: ' + reason)

        def on_scan(self, msg):
            pts = scan_points(msg)
            if pts is None:
                return
            self.last_scan_at = time.monotonic()
            self.obstacle_dist = front_distance(pts)
            if self.obstacle_dist <= AVOID_TRIGGER_M:
                self.hit_scans += 1
                self.clear_scans = 0
            else:
                self.hit_scans = 0
                if self.obstacle_dist >= OBSTACLE_REARM_M:
                    self.clear_scans += 1
                else:
                    self.clear_scans = 0
            if self.clear_scans >= CLEAR_CONFIRM_SCANS and self.state.startswith('FOLLOW'):
                self.armed = True

        def watchdog(self):
            now = time.monotonic()
            if self.state == 'STOP':
                self.command(lane.STEERING_CENTER_US, lane.NEUTRAL_US)
                return
            if self.last_scan_at is None or now - self.last_scan_at > SCAN_TIMEOUT_SEC:
                self.stop('LiDAR missing/stale')
            elif self.last_image_at is None:
                if now - self.boot_at > lane.IMAGE_TIMEOUT_SEC + .5:
                    self.stop('camera missing')
            elif now - self.last_image_at > lane.IMAGE_TIMEOUT_SEC:
                self.stop('camera missing/stale')

        def transition(self, state):
            self.state = state
            self.state_at = time.monotonic()
            self.log('state=' + state, True)

        def fit_target(self, white):
            if self.target_x is None:
                return None
            # Use target stripe location, rather than letting the base controller
            # re-acquire whichever white stripe is closest to car center.
            fit = lane.fit_lane_track(white, self.target_x)
            if fit is None:
                return None
            self.target_x = float(fit[5])
            self.target_seen_at = time.monotonic()
            return fit

        def follow(self, white, yellow, now):
            fit = lane.fit_lane_track(white, self.track_x)
            if fit is None:
                self.stop('current stripe lost')
                return None
            near_x, angle, _, _, _, bottom_x, partial = fit
            self.track_x = self.current_x = float(bottom_x)
            steering = self.ctrl.steering_pulse(bottom_x if partial else near_x,
                                                 90.0 if partial else angle, white.shape[1])
            speed = lane.curve_speed_us(options.speed_us, angle, steering)
            if partial:
                speed = min(speed, lane.CURVE_SPEED_US)
            self.command(steering, speed)
            if self.change_count < 2 and self.armed and self.hit_scans >= OBSTACLE_CONFIRM_SCANS:
                # 2nd event must be opposite direction to the 1st.
                required = None if self.change_count == 0 else ('right' if self.direction == 'left' else 'left')
                direction, target, source = choose_target(white, yellow, bottom_x, required)
                if direction is None:
                    self.stop('obstacle ahead: no unique safe direction from stripes (' + source + ')')
                    return fit
                # The first white stripe detected is not enough; verify it
                # over multiple frames, while applying neutral throttle.
                if (self.candidate_side == direction and
                        (target is None or self.target_x is None or
                         abs(self.target_x - target) < white.shape[1] * .12)):
                    self.candidate_count += 1
                else:
                    self.candidate_side = direction
                    self.candidate_count = 1
                self.target_x = target
                self.target_source = source
                self.command(lane.STEERING_CENTER_US, lane.NEUTRAL_US)
                if self.candidate_count >= TARGET_CONFIRM_FRAMES:
                    self.direction = direction
                    self.armed = False
                    self.target_seen_at = now
                    self.merge_ok_count = 0
                    self.transition('AVOID')
                    self.log('avoid direction=' + direction + ' from=' + source, True)
            else:
                self.candidate_count = 0
                self.candidate_side = None
            return fit

        def avoid(self, white, yellow, now):
            if now - self.state_at > AVOID_MAX_SEC:
                self.stop('avoid timeout')
                return None
            # If yellow initially indicated the direction, the white target
            # can appear later as the car starts its lane change.
            if self.target_x is None:
                _, candidate, _ = choose_target(white, yellow, self.current_x, self.direction)
                if candidate is not None:
                    self.target_x = candidate
                    self.target_seen_at = now
            fit = self.fit_target(white)
            if self.target_x is not None and fit is None and now - (self.target_seen_at or now) > TARGET_LOST_SEC:
                self.stop('target white stripe lost while avoiding')
                return None
            if fit is not None:
                target_bottom = fit[5]
                reference_x = lane.LINE_CENTER_X_PX * white.shape[1] / 640.
                close_enough = abs(target_bottom - reference_x) <= APPROACH_MARGIN_PX * white.shape[1] / 640.
                if close_enough or now - self.state_at >= MAX_STEER_STAGE_SEC:
                    self.ctrl.reset()
                    self.transition('MERGE')
                    return self.merge(white, now)
            elif now - self.state_at >= MAX_STEER_STAGE_SEC:
                self.stop('no target white stripe visible after max-steer stage')
                return None
            self.command(steer_max(self.direction), min(AVOID_SPEED_US, options.speed_us))
            return fit

        def merge(self, white, now):
            if now - self.state_at > MERGE_MAX_SEC:
                self.stop('merge timeout')
                return None
            fit = self.fit_target(white)
            if fit is None:
                self.stop('target stripe lost during merge')
                return None
            near_x, heading, _, _, _, bottom_x, partial = fit
            steer = self.ctrl.steering_pulse(bottom_x if partial else near_x,
                                             90.0 if partial else heading, white.shape[1])
            self.command(steer, min(MERGE_SPEED_US, options.speed_us))
            error = abs(bottom_x - lane.LINE_CENTER_X_PX * white.shape[1] / 640.)
            if (not partial and error <= MERGE_DONE_PX * white.shape[1] / 640.
                    and abs(heading - 90) <= MERGE_HEADING_TOL_DEG):
                self.merge_ok_count += 1
            else:
                self.merge_ok_count = 0
            if self.merge_ok_count >= MERGE_CONFIRM_FRAMES:
                self.track_x = float(bottom_x)
                self.current_x = float(bottom_x)
                self.change_count += 1
                self.target_x = None
                self.candidate_count = 0
                self.ctrl.reset()
                self.transition('FOLLOW_INNER' if self.change_count == 1 else 'FOLLOW_OUTER_FINAL')
                self.get_logger().info('lane changes completed: %d/2' % self.change_count)
            return fit

        def on_image(self, msg):
            now = time.monotonic()
            if self.state == 'STOP':
                return
            stamp = msg.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if stamp_ns:
                age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
                if age < -.05 or age > lane.MAX_IMAGE_AGE_SEC:
                    self.stop('camera timestamp stale/out of sync')
                    return
            else:
                age = 0.
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                bev, white, yellow = get_masks(img)
            except Exception as e:
                self.stop('image processing failed: ' + str(e))
                return
            if time.monotonic() - now + age > lane.MAX_IMAGE_AGE_SEC:
                self.stop('image processing too slow')
                return
            self.last_image_at = time.monotonic()
            if self.last_scan_at is None or self.last_image_at - self.last_scan_at > SCAN_TIMEOUT_SEC:
                self.stop('LiDAR unavailable')
                return
            fit = None
            if self.state.startswith('FOLLOW'):
                fit = self.follow(white, yellow, self.last_image_at)
            elif self.state == 'AVOID':
                fit = self.avoid(white, yellow, self.last_image_at)
            elif self.state == 'MERGE':
                fit = self.merge(white, self.last_image_at)
            self.log(f'{self.state} changes={self.change_count}/2 distance={self.obstacle_dist:.2f}m '
                     f'target={self.target_x} steering={self.last_steer}')
            if options.show:
                debug = bev.copy()
                if self.target_x is not None:
                    cv2.line(debug, (int(self.target_x), 0),
                             (int(self.target_x), debug.shape[0]-1), (0, 255, 0), 2)
                cv2.putText(debug, f'{self.state} {self.change_count}/2 {self.obstacle_dist:.2f}m',
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
                cv2.imshow('avoid v2 / BEV', debug)
                cv2.imshow('white mask', white)
                cv2.imshow('yellow boundary preview', yellow)
                cv2.waitKey(1)

    hw = None
    node = None
    ready = False
    try:
        if options.drive:
            hw = lane.DriveHardware()
            time.sleep(3)
        rclpy.init(args=ros_args)
        ready = True
        node = AvoidNode(hw)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print('Error:', exc)
        return 1
    finally:
        if hw is not None:
            hw.close()
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if ready and rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
