#!/usr/bin/env python3
"""Tests for the time-based LiDAR avoidance state machine."""

import math
import unittest

from lidar_avoid import AvoidController, Distances


class AvoidControllerTests(unittest.TestCase):

    def test_clear_path_drives_straight(self):
        controller = AvoidController()
        command = controller.update(
            Distances(math.inf, math.inf, math.inf),
            0.0,
        )
        self.assertEqual(command, 'STRAIGHT')

    def test_completes_left_then_right_avoidance(self):
        controller = AvoidController()
        obstacle = Distances(0.70, 1.50, 0.80)
        clear = Distances(1.20, 1.50, 1.50)

        self.assertEqual(controller.update(obstacle, 0.00), 'STOP_LEFT')
        self.assertEqual(controller.update(obstacle, 0.21), 'LEFT')
        self.assertEqual(controller.update(clear, 0.70), 'LEFT')
        self.assertEqual(controller.update(clear, 0.80), 'LEFT')
        self.assertEqual(controller.update(clear, 0.90), 'STOP_RIGHT')
        self.assertEqual(controller.update(clear, 1.11), 'RIGHT')
        self.assertEqual(controller.update(clear, 1.90), 'STRAIGHT')

    def test_emergency_stop_is_latched(self):
        controller = AvoidController()
        self.assertEqual(
            controller.update(Distances(0.30, 1.0, 1.0), 0.0),
            'STOP',
        )
        self.assertEqual(
            controller.update(Distances(2.0, 2.0, 2.0), 1.0),
            'STOP',
        )
        self.assertEqual(controller.state, 'HALT')


if __name__ == '__main__':
    unittest.main()
