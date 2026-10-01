#!/usr/bin/env python3
"""Print the nearest front LiDAR distance once per second without driving."""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


SCAN_TOPIC = "/scan"
REPORT_INTERVAL_SEC = 1.0
SCAN_TIMEOUT_SEC = 2.0
FRONT_CENTER_DEG = 0.0
FRONT_HALF_ANGLE_DEG = 30.0


def minimum_front_distance(scan):
    """Return the nearest valid range in the configured front sector."""
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

        if math.isinf(distance) and distance > 0:
            has_measurement = True
        elif math.isfinite(distance) and scan.range_min <= distance <= scan.range_max:
            has_measurement = True
            nearest = min(nearest, distance)

    if not has_measurement:
        return None
    return nearest


class LidarDistanceMonitor(Node):
    def __init__(self):
        super().__init__("lidar_distance_monitor")
        self.nearest_distance = None
        self.last_scan_time = None
        self.range_max = None

        self.subscription = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.report_timer = self.create_timer(
            REPORT_INTERVAL_SEC,
            self.report_distance,
        )

        self.get_logger().info(
            f"{SCAN_TOPIC} 대기 중: 정면 ±{FRONT_HALF_ANGLE_DEG:.0f}° "
            f"최근접 거리를 {REPORT_INTERVAL_SEC:.0f}초마다 출력합니다"
        )
        self.get_logger().info("이 노드는 모터 또는 조향 장치를 제어하지 않습니다")

    def scan_callback(self, scan):
        self.nearest_distance = minimum_front_distance(scan)
        self.last_scan_time = time.monotonic()
        self.range_max = scan.range_max

    def report_distance(self):
        if self.last_scan_time is None:
            self.get_logger().warning("/scan 데이터 대기 중")
            return

        scan_age = time.monotonic() - self.last_scan_time
        if scan_age > SCAN_TIMEOUT_SEC:
            self.get_logger().warning(
                f"/scan 데이터가 {scan_age:.1f}초 동안 들어오지 않았습니다"
            )
            return

        if self.nearest_distance is None:
            self.get_logger().warning("정면 구간에 유효한 거리값이 없습니다")
        elif math.isinf(self.nearest_distance):
            max_text = (
                f"{self.range_max:.1f} m 이내"
                if self.range_max is not None and math.isfinite(self.range_max)
                else "측정 범위 내"
            )
            self.get_logger().info(f"정면 장애물 없음 ({max_text})")
        else:
            self.get_logger().info(
                f"정면 최근접 장애물: {self.nearest_distance:.3f} m"
            )


def main(args=None):
    rclpy.init(args=args)
    node = LidarDistanceMonitor()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("\n거리 모니터 종료")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
