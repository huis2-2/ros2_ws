#!/usr/bin/env python3
"""Follow a white lane and let LiDAR obstacle avoidance take priority."""

import argparse
from dataclasses import dataclass
import time

import cv2

from camera_lane_follow import (
    DriveHardware,
    LANE_CONFIRM_FRAMES,
    LaneController,
    NEUTRAL_US,
    STEERING_CENTER_US,
    STEERING_LEFT_US,
    STEERING_RIGHT_US,
    detect_white_lane,
)
from lidar_avoidance_logic import AvoidanceController, analyze_scan


CAMERA_TOPIC = '/camera/image_raw'
SCAN_TOPIC = '/scan'
IMAGE_TIMEOUT_SEC = 0.5
SCAN_TIMEOUT_SEC = 0.5

LANE_SPEED_US = 1565
AVOID_SPEED_US = 1565

FRONT_CENTER_DEG = 180.0
FRONT_HALF_WIDTH_DEG = 25.0
SIDE_CENTER_OFFSET_DEG = 50.0
SIDE_HALF_WIDTH_DEG = 25.0


@dataclass(frozen=True)
class MotionDecision:
    """One combined steering and throttle decision."""

    mode: str
    steering_us: int
    speed_us: int
    moving: bool
    reason: str


def choose_motion(
    lidar_command,
    lidar_fresh,
    image_fresh,
    lane_confirmed,
    lane_steering_us,
    lane_speed_us=LANE_SPEED_US,
    avoid_speed_us=AVOID_SPEED_US,
):
    """Give LiDAR safety and avoidance priority over lane following."""
    if lidar_command is None:
        return MotionDecision(
            'waiting_lidar',
            STEERING_CENTER_US,
            NEUTRAL_US,
            False,
            'LiDAR 데이터를 기다립니다',
        )

    if not lidar_fresh:
        return MotionDecision(
            'scan_timeout',
            STEERING_CENTER_US,
            NEUTRAL_US,
            False,
            'LiDAR 데이터가 끊겼습니다',
        )

    if not lidar_command.moving:
        return MotionDecision(
            lidar_command.mode,
            STEERING_CENTER_US,
            NEUTRAL_US,
            False,
            lidar_command.reason,
        )

    if lidar_command.mode in ('avoiding_left', 'avoiding_right'):
        steering_us = (
            STEERING_LEFT_US
            if lidar_command.mode == 'avoiding_left'
            else STEERING_RIGHT_US
        )
        return MotionDecision(
            lidar_command.mode,
            steering_us,
            avoid_speed_us,
            True,
            lidar_command.reason,
        )

    if lidar_command.mode != 'straight':
        return MotionDecision(
            'unknown_lidar_state',
            STEERING_CENTER_US,
            NEUTRAL_US,
            False,
            f'알 수 없는 LiDAR 상태: {lidar_command.mode}',
        )

    if not image_fresh:
        return MotionDecision(
            'camera_timeout',
            STEERING_CENTER_US,
            NEUTRAL_US,
            False,
            '카메라 영상이 끊겼습니다',
        )

    if not lane_confirmed or lane_steering_us is None:
        return MotionDecision(
            'lane_lost',
            STEERING_CENTER_US,
            NEUTRAL_US,
            False,
            '흰 차선을 찾지 못했습니다',
        )

    return MotionDecision(
        'lane_follow',
        lane_steering_us,
        lane_speed_us,
        True,
        '전방이 깨끗해 흰 차선을 따라갑니다',
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--drive',
        action='store_true',
        help='PCA9685에 실제 조향 및 속도 PWM 출력',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='차선 인식 결과와 흰색/노란색 마스크 표시',
    )
    parser.add_argument('--camera-topic', default=CAMERA_TOPIC)
    parser.add_argument('--scan-topic', default=SCAN_TOPIC)
    parser.add_argument(
        '--lane-speed-us',
        type=int,
        default=LANE_SPEED_US,
        help=f'차선 주행 속도 펄스(기본 {LANE_SPEED_US} us)',
    )
    parser.add_argument(
        '--avoid-speed-us',
        type=int,
        default=AVOID_SPEED_US,
        help=f'장애물 회피 속도 펄스(기본 {AVOID_SPEED_US} us)',
    )
    return parser


def main(argv=None):
    parser = build_parser()
    options, ros_args = parser.parse_known_args(argv)
    for name, value in (
        ('--lane-speed-us', options.lane_speed_us),
        ('--avoid-speed-us', options.avoid_speed_us),
    ):
        if not NEUTRAL_US <= value <= 1565:
            parser.error(f'{name}는 안전상 1500~1565 범위만 허용합니다')

    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, LaserScan
    except ModuleNotFoundError as exc:
        print(f'ROS 2 모듈을 불러올 수 없습니다: {exc}')
        print('source /opt/ros/jazzy/setup.bash 후 다시 실행하세요.')
        return 2

    class CameraLidarFollower(Node):
        """Combine lane observations and LiDAR avoidance commands."""

        def __init__(self, hardware):
            super().__init__('camera_lidar_follower')
            self.hardware = hardware
            self.bridge = CvBridge()
            self.lane_controller = LaneController()
            self.avoidance_controller = AvoidanceController()

            self.last_image_at = None
            self.last_scan_at = None
            self.lane_frames = 0
            self.lane_steering_us = None
            self.lidar_command = None
            self.last_decision = None
            self.last_log_key = None
            self.last_log_at = 0.0

            self.image_subscription = self.create_subscription(
                Image,
                options.camera_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
            self.scan_subscription = self.create_subscription(
                LaserScan,
                options.scan_topic,
                self.scan_callback,
                qos_profile_sensor_data,
            )
            self.control_timer = self.create_timer(0.05, self.control)

            mode = '실제 통합 주행' if hardware is not None else '판단 로그 시험'
            self.get_logger().info(
                f'{mode}: camera={options.camera_topic}, '
                f'lidar={options.scan_topic}'
            )
            self.get_logger().info(
                f'차선 속도={options.lane_speed_us} us, '
                f'회피 속도={options.avoid_speed_us} us'
            )

        def image_callback(self, message):
            self.last_image_at = time.monotonic()
            try:
                image = self.bridge.imgmsg_to_cv2(
                    message,
                    desired_encoding='bgr8',
                )
                observation, bev, white_mask, yellow_mask = (
                    detect_white_lane(image)
                )
            except Exception as exc:
                self.lane_controller.reset()
                self.lane_frames = 0
                self.lane_steering_us = None
                self.get_logger().warning(f'영상 처리 오류: {exc}')
                return

            if observation is None:
                self.lane_controller.reset()
                self.lane_frames = 0
                self.lane_steering_us = None
            else:
                self.lane_frames += 1
                self.lane_steering_us = self.lane_controller.steering_pulse(
                    observation.center_x,
                    observation.angle_deg,
                    image.shape[1],
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

                mode = (
                    self.last_decision.mode
                    if self.last_decision is not None
                    else 'WAITING'
                )
                cv2.putText(
                    display,
                    mode,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow('camera + lidar follow', display)
                cv2.imshow('white mask - lane input', white_mask)
                cv2.imshow('yellow mask - preview only', yellow_mask)
                cv2.waitKey(1)

        def scan_callback(self, scan):
            self.last_scan_at = time.monotonic()
            sectors = analyze_scan(
                scan,
                front_center_deg=FRONT_CENTER_DEG,
                front_half_width_deg=FRONT_HALF_WIDTH_DEG,
                side_center_offset_deg=SIDE_CENTER_OFFSET_DEG,
                side_half_width_deg=SIDE_HALF_WIDTH_DEG,
            )
            self.lidar_command = self.avoidance_controller.update(sectors)

        def control(self):
            now = time.monotonic()
            lidar_fresh = (
                self.last_scan_at is not None
                and now - self.last_scan_at <= SCAN_TIMEOUT_SEC
            )
            image_fresh = (
                self.last_image_at is not None
                and now - self.last_image_at <= IMAGE_TIMEOUT_SEC
            )
            lane_confirmed = self.lane_frames >= LANE_CONFIRM_FRAMES
            decision = choose_motion(
                self.lidar_command,
                lidar_fresh,
                image_fresh,
                lane_confirmed,
                self.lane_steering_us,
                options.lane_speed_us,
                options.avoid_speed_us,
            )
            self.last_decision = decision

            if self.hardware is not None:
                if decision.moving:
                    self.hardware.set_steering(decision.steering_us)
                    self.hardware.set_speed(decision.speed_us)
                else:
                    self.hardware.stop()

            log_key = decision.mode
            if log_key != self.last_log_key or now - self.last_log_at >= 1.0:
                text = (
                    f'{decision.mode}: {decision.reason}; '
                    f'조향={decision.steering_us} us, '
                    f'속도={decision.speed_us} us'
                )
                if decision.moving:
                    self.get_logger().info(text)
                else:
                    self.get_logger().warning(text)
                self.last_log_key = log_key
                self.last_log_at = now

    hardware = None
    node = None
    ros_initialized = False

    try:
        if options.drive:
            hardware = DriveHardware()
            print('ESC 중립 신호 안정화 중 (3초)')
            time.sleep(3.0)

        rclpy.init(args=ros_args)
        ros_initialized = True
        node = CameraLidarFollower(hardware)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
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
