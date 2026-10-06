#!/usr/bin/env python3
"""Unit tests for LiDAR sector analysis and avoidance decisions."""

import math
from types import SimpleNamespace
import unittest

from lidar_avoidance_logic import (
    analyze_scan,
    AvoidanceController,
    ScanSectors,
)


def make_scan(default=math.inf, points=None):
    ranges = [default] * 360
    for angle_deg, distance in (points or {}).items():
        ranges[angle_deg % 360] = distance
    return SimpleNamespace(
        ranges=ranges,
        angle_min=0.0,
        angle_increment=math.radians(1.0),
        range_min=0.05,
        range_max=16.0,
    )


class ScanAnalysisTests(unittest.TestCase):
    def test_front_and_side_sectors_use_180_degree_vehicle_front(self):
        sectors = analyze_scan(
            make_scan(points={180: 0.60, 230: 1.20, 130: 0.80})
        )
        self.assertAlmostEqual(sectors.front_m, 0.60)
        self.assertAlmostEqual(sectors.left_m, 1.20)
        self.assertAlmostEqual(sectors.right_m, 0.80)

    def test_invalid_sector_returns_none(self):
        scan = make_scan(default=0.0)
        sectors = analyze_scan(scan)
        self.assertIsNone(sectors.front_m)
        self.assertIsNone(sectors.left_m)
        self.assertIsNone(sectors.right_m)


class AvoidanceControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = AvoidanceController()

    def test_clear_path_drives_straight(self):
        command = self.controller.update(ScanSectors(2.0, 2.0, 2.0))
        self.assertEqual(command.mode, 'straight')
        self.assertTrue(command.moving)

    def test_selects_side_with_more_clearance(self):
        command = self.controller.update(ScanSectors(0.60, 1.50, 0.80))
        self.assertEqual(command.mode, 'avoiding_left')
        self.assertEqual(command.steering, 'left')

    def test_keeps_direction_until_clear_distance(self):
        self.controller.update(ScanSectors(0.60, 1.50, 0.80))
        command = self.controller.update(ScanSectors(0.80, 0.70, 3.00))
        self.assertEqual(command.mode, 'avoiding_left')
        command = self.controller.update(ScanSectors(0.90, 0.70, 3.00))
        self.assertEqual(command.mode, 'straight')

    def test_stops_for_emergency_distance(self):
        command = self.controller.update(ScanSectors(0.20, 3.0, 3.0))
        self.assertEqual(command.mode, 'emergency_stop')
        self.assertFalse(command.moving)

    def test_stops_when_both_corridors_are_blocked(self):
        command = self.controller.update(ScanSectors(0.60, 0.30, 0.40))
        self.assertEqual(command.mode, 'blocked')
        self.assertFalse(command.moving)

    def test_invalid_scan_is_fail_safe(self):
        command = self.controller.update(ScanSectors(None, 1.0, 1.0))
        self.assertEqual(command.mode, 'sensor_fault')
        self.assertFalse(command.moving)


if __name__ == '__main__':
    unittest.main()
