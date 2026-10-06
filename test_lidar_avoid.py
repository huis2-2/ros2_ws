#!/usr/bin/env python3
"""Tests for the time-based LiDAR avoidance state machine."""

import math
import unittest
from unittest.mock import patch

from lidar_avoid import AvoidController, Distances, DriveHardware, STEERING_CH


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

        self.assertEqual(controller.update(obstacle, 0.00), 'LEFT')
        self.assertEqual(controller.update(clear, 0.40), 'LEFT')
        self.assertEqual(controller.update(clear, 0.50), 'LEFT')
        self.assertEqual(controller.update(clear, 0.60), 'RIGHT')
        self.assertEqual(controller.update(clear, 1.30), 'STRAIGHT')

    def test_keeps_avoiding_after_old_turn_timeout(self):
        controller = AvoidController()
        obstacle = Distances(0.70, 1.50, 0.80)

        self.assertEqual(controller.update(obstacle, 0.00), 'LEFT')
        self.assertEqual(controller.update(obstacle, 2.00), 'LEFT')
        self.assertEqual(controller.update(obstacle, 10.00), 'LEFT')
        self.assertNotEqual(controller.state, 'PAUSED')

    def test_reselects_direction_if_corridor_closes(self):
        controller = AvoidController()
        self.assertEqual(
            controller.update(Distances(0.70, 1.50, 0.80), 0.00),
            'LEFT',
        )
        self.assertEqual(
            controller.update(Distances(0.70, 0.40, 1.20), 0.21),
            'RIGHT',
        )
        self.assertNotEqual(controller.state, 'PAUSED')

    def test_chooses_wider_side_below_old_clearance_limit(self):
        controller = AvoidController()
        command = controller.update(
            Distances(0.70, 0.50, 0.40),
            0.0,
        )
        self.assertEqual(command, 'LEFT')
        self.assertNotEqual(controller.state, 'PAUSED')

    def test_emergency_stop_resumes_when_obstacle_clears(self):
        controller = AvoidController()
        self.assertEqual(
            controller.update(Distances(0.30, 1.0, 1.0), 0.0),
            'STOP',
        )
        self.assertEqual(controller.state, 'PAUSED')
        self.assertEqual(
            controller.update(Distances(0.70, 1.0, 0.8), 1.0),
            'LEFT',
        )
        self.assertEqual(controller.state, 'TURN_OUT')


class DriveHardwareTests(unittest.TestCase):

    def test_steering_command_writes_channel_nine(self):
        hardware = DriveHardware.__new__(DriveHardware)
        hardware.bus = object()
        hardware.current_steering_us = 1640

        with patch('lidar_avoid.set_pwm_us') as set_pwm:
            hardware.set_steering(1880)
            set_pwm.assert_called_once_with(hardware.bus, STEERING_CH, 1880)

            hardware.set_steering(1880)
            set_pwm.assert_called_once()


if __name__ == '__main__':
    unittest.main()
