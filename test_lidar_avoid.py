#!/usr/bin/env python3
"""Tests for the time-based LiDAR avoidance state machine."""

import math
import unittest
from unittest.mock import Mock, patch

from lidar_avoid import (
    AvoidController,
    Distances,
    DriveHardware,
    STEERING_CH,
    THROTTLE_CH,
    format_distance,
    start_rviz,
    stop_rviz,
)


class AvoidControllerTests(unittest.TestCase):

    def test_clear_path_drives_straight(self):
        controller = AvoidController()
        command = controller.update(
            Distances(math.inf, math.inf, math.inf),
            0.0,
        )
        self.assertEqual(command, 'STRAIGHT')

    def test_invalid_distances_keep_driving_straight(self):
        controller = AvoidController()
        command = controller.update(None, 0.0)
        self.assertEqual(command, 'STRAIGHT')
        self.assertEqual(controller.state, 'CRUISE')

    def test_clear_front_ignores_close_side_and_keeps_driving(self):
        controller = AvoidController()
        command = controller.update(
            Distances(2.0, 0.10, 0.10),
            0.0,
        )
        self.assertEqual(command, 'STRAIGHT')
        self.assertEqual(controller.state, 'CRUISE')

    def test_centers_and_drives_straight_when_obstacle_clears(self):
        controller = AvoidController()
        obstacle = Distances(0.35, 1.50, 0.80)
        clear = Distances(0.60, 1.50, 1.50)

        self.assertEqual(controller.update(obstacle, 0.00), 'LEFT')
        self.assertEqual(controller.update(clear, 0.40), 'LEFT')
        self.assertEqual(controller.update(clear, 0.50), 'LEFT')
        self.assertEqual(controller.update(clear, 0.60), 'RIGHT')
        self.assertEqual(controller.state, 'ALIGN')
        self.assertEqual(controller.update(clear, 1.21), 'STRAIGHT')
        self.assertEqual(controller.state, 'CRUISE')

    def test_max_turn_timeout_starts_opposite_alignment(self):
        controller = AvoidController()
        obstacle = Distances(0.35, 1.50, 0.80)

        self.assertEqual(controller.update(obstacle, 0.00), 'LEFT')
        self.assertEqual(controller.update(obstacle, 0.79), 'LEFT')
        self.assertEqual(controller.update(obstacle, 0.80), 'RIGHT')
        self.assertEqual(controller.state, 'ALIGN')
        self.assertEqual(controller.update(obstacle, 1.61), 'STRAIGHT')

    def test_turn_direction_remains_selected_during_turn_out(self):
        controller = AvoidController()
        self.assertEqual(
            controller.update(Distances(0.35, 1.50, 0.80), 0.00),
            'LEFT',
        )
        self.assertEqual(
            controller.update(Distances(0.35, 0.40, 1.20), 0.21),
            'LEFT',
        )
        self.assertEqual(controller.state, 'TURN_OUT')

    def test_chooses_wider_side_below_old_clearance_limit(self):
        controller = AvoidController()
        command = controller.update(
            Distances(0.35, 0.50, 0.40),
            0.0,
        )
        self.assertEqual(command, 'LEFT')
        self.assertEqual(controller.state, 'TURN_OUT')

    def test_close_obstacle_keeps_driving_and_steering(self):
        controller = AvoidController()
        self.assertEqual(
            controller.update(Distances(0.10, 1.0, 0.8), 0.0),
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

    def test_drive_speed_writes_throttle_channel(self):
        hardware = DriveHardware.__new__(DriveHardware)
        hardware.bus = object()
        hardware.current_speed_us = 1500

        with patch('lidar_avoid.set_pwm_us') as set_pwm:
            hardware.set_speed(1565)
            set_pwm.assert_called_once_with(hardware.bus, THROTTLE_CH, 1565)


class DisplayHelperTests(unittest.TestCase):

    def test_distance_formatting(self):
        self.assertEqual(format_distance(1.234), '1.23 m')
        self.assertEqual(format_distance(math.inf), '감지 없음')
        self.assertEqual(format_distance(None), '없음')

    @patch('lidar_avoid.subprocess.Popen')
    @patch('lidar_avoid.RVIZ_CONFIG')
    @patch.dict('lidar_avoid.os.environ', {'DISPLAY': ':1'}, clear=True)
    def test_rviz_uses_workspace_config(self, config, popen):
        config.is_file.return_value = True
        config.__str__.return_value = '/tmp/lidar.rviz'

        process = start_rviz()

        self.assertIs(process, popen.return_value)
        popen.assert_called_once_with(
            ['rviz2', '-d', '/tmp/lidar.rviz'],
            start_new_session=True,
        )

    def test_stop_rviz_terminates_child(self):
        process = Mock()
        process.poll.return_value = None

        stop_rviz(process)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=3.0)


if __name__ == '__main__':
    unittest.main()
