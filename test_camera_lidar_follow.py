"""Unit tests for combined camera and LiDAR command arbitration."""

import math
import unittest

from camera_lane_follow import (
    NEUTRAL_US,
    STEERING_CENTER_US,
    STEERING_LEFT_US,
    STEERING_RIGHT_US,
)
from camera_lidar_follow import (
    build_avoidance_controller,
    choose_motion,
    format_distance,
)
from lidar_avoidance_logic import DriveCommand, ScanSectors


class CombinedDecisionTests(unittest.TestCase):
    def test_clear_lidar_uses_lane_steering(self):
        lidar = DriveCommand('straight', 'center', True, 'clear')

        decision = choose_motion(lidar, True, True, True, 1710)

        self.assertEqual(decision.mode, 'lane_follow')
        self.assertEqual(decision.steering_us, 1710)
        self.assertEqual(decision.speed_us, 1565)
        self.assertTrue(decision.moving)

    def test_lidar_avoidance_overrides_lane(self):
        lidar = DriveCommand('avoiding_left', 'left', True, 'obstacle')

        decision = choose_motion(lidar, True, True, True, 1400)

        self.assertEqual(decision.mode, 'avoiding_left')
        self.assertEqual(decision.steering_us, STEERING_LEFT_US)
        self.assertEqual(decision.speed_us, 1565)
        self.assertTrue(decision.moving)

    def test_right_avoidance_uses_calibrated_endpoint(self):
        lidar = DriveCommand('avoiding_right', 'right', True, 'obstacle')

        decision = choose_motion(lidar, True, False, False, None)

        self.assertEqual(decision.steering_us, STEERING_RIGHT_US)
        self.assertTrue(decision.moving)

    def test_emergency_stop_overrides_lane(self):
        lidar = DriveCommand('emergency_stop', 'center', False, 'too close')

        decision = choose_motion(lidar, True, True, True, 1710)

        self.assertEqual(decision.mode, 'emergency_stop')
        self.assertEqual(decision.steering_us, STEERING_CENTER_US)
        self.assertEqual(decision.speed_us, NEUTRAL_US)
        self.assertFalse(decision.moving)

    def test_stale_lidar_stops(self):
        lidar = DriveCommand('straight', 'center', True, 'clear')

        decision = choose_motion(lidar, False, True, True, 1710)

        self.assertEqual(decision.mode, 'scan_timeout')
        self.assertFalse(decision.moving)

    def test_clear_path_without_lane_stops(self):
        lidar = DriveCommand('straight', 'center', True, 'clear')

        decision = choose_motion(lidar, True, True, False, None)

        self.assertEqual(decision.mode, 'lane_lost')
        self.assertFalse(decision.moving)

    def test_stale_camera_stops_lane_following(self):
        lidar = DriveCommand('straight', 'center', True, 'clear')

        decision = choose_motion(lidar, True, False, True, 1710)

        self.assertEqual(decision.mode, 'camera_timeout')
        self.assertFalse(decision.moving)


class DistanceFormattingTests(unittest.TestCase):
    def test_formats_measured_distance(self):
        self.assertEqual(format_distance(0.834), '0.83 m')

    def test_formats_clear_and_invalid_sectors(self):
        self.assertEqual(format_distance(math.inf), '감지 없음')
        self.assertEqual(format_distance(None), '데이터 없음')


class ReturnToLaneTests(unittest.TestCase):
    def test_returns_to_lane_after_obstacle_clears(self):
        controller = build_avoidance_controller()
        avoiding = controller.update(ScanSectors(0.60, 1.20, 0.80))
        cleared = controller.update(ScanSectors(0.76, 1.20, 0.80))

        self.assertEqual(avoiding.mode, 'avoiding_left')
        self.assertEqual(cleared.mode, 'straight')

        decision = choose_motion(cleared, True, True, True, 1670)
        self.assertEqual(decision.mode, 'lane_follow')
        self.assertEqual(decision.steering_us, 1670)
        self.assertTrue(decision.moving)

    def test_keeps_avoiding_inside_return_threshold(self):
        controller = build_avoidance_controller()
        controller.update(ScanSectors(0.60, 1.20, 0.80))

        command = controller.update(ScanSectors(0.74, 1.20, 0.80))

        self.assertEqual(command.mode, 'avoiding_left')


if __name__ == '__main__':
    unittest.main()
