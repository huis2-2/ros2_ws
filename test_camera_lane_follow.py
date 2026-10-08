"""Unit tests for white-lane detection and steering conversion."""

import unittest
from unittest.mock import Mock

from camera_lane_follow import (
    LaneController,
    MAX_DELTA_ANGULAR,
    MAX_SPEED_US,
    SLOW_SPEED_US,
    STEERING_CENTER_US,
    WHITE_LOWER_HSV,
    WHITE_UPPER_HSV,
    YELLOW_LOWER_HSV,
    YELLOW_UPPER_HSV,
    calculate_lane_info,
    build_parser,
    detect_edges,
    detect_white_lane,
    lane_color_mask,
    write_status_log,
)

import numpy as np


class LaneDetectionTests(unittest.TestCase):
    """Check image-mask and Hough-line interpretation helpers."""

    def test_requested_hsv_ranges_are_used(self):
        self.assertEqual(WHITE_LOWER_HSV, (0, 0, 252))
        self.assertEqual(WHITE_UPPER_HSV, (0, 116, 255))
        self.assertEqual(YELLOW_LOWER_HSV, (7, 21, 201))
        self.assertEqual(YELLOW_UPPER_HSV, (179, 255, 255))

    def test_lane_info_uses_visible_vertical_line(self):
        """A vertical lower-image line should produce its x and 90 degrees."""
        lines = np.array([[[200, 300, 200, 450]]], dtype=np.int32)

        center_x, angle = calculate_lane_info(lines, 480)

        self.assertEqual(center_x, 200.0)
        self.assertEqual(angle, 90.0)

    def test_white_mask_ignores_upper_image(self):
        """Only white pixels in the lower road ROI should remain."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[50:100, 300:340] = (255, 255, 255)
        image[350:430, 300:340] = (255, 255, 255)

        mask = lane_color_mask(image, WHITE_LOWER_HSV, WHITE_UPPER_HSV)

        self.assertEqual(int(mask[75, 320]), 0)
        self.assertEqual(int(mask[390, 320]), 255)

    def test_yellow_mask_uses_calibrated_range(self):
        """The separate yellow mask should use its calibrated range."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[350:430, 300:340] = (100, 200, 220)

        mask = lane_color_mask(image, YELLOW_LOWER_HSV, YELLOW_UPPER_HSV)

        self.assertGreater(int(np.count_nonzero(mask)), 0)

    def test_white_detector_does_not_follow_yellow_only_image(self):
        """A yellow-only road image must not produce a steering lane."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[300:450, 280:340] = (100, 200, 220)

        observation, _, white_mask, yellow_mask = detect_white_lane(image)

        self.assertIsNone(observation)
        self.assertEqual(int(np.count_nonzero(white_mask)), 0)
        self.assertGreater(int(np.count_nonzero(yellow_mask)), 0)

    def test_canny_edges_follow_white_mask_boundary(self):
        """Canny processing should retain the lower white stripe edges."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[300:440, 300:340] = 255

        blurred, edges = detect_edges(mask)

        self.assertGreater(int(np.count_nonzero(blurred)), 0)
        self.assertGreater(int(np.count_nonzero(edges)), 0)


class LaneControllerTests(unittest.TestCase):
    """Check calibrated steering direction and rate limiting."""

    def test_target_lane_keeps_steering_centered(self):
        """A lane at the configured target should retain center steering."""
        controller = LaneController()

        pulse = controller.steering_pulse(320.0, 90.0, 640)

        self.assertEqual(pulse, STEERING_CENTER_US)

    def test_left_error_uses_supplied_control_gains(self):
        """Left error should use the supplied gain and smoothing values."""
        controller = LaneController()

        pulse = controller.steering_pulse(100.0, 60.0, 640)

        self.assertEqual(pulse, 1784)

    def test_right_error_uses_supplied_control_gains(self):
        """Right error should use the supplied gain and smoothing values."""
        controller = LaneController()

        pulse = controller.steering_pulse(500.0, 120.0, 640)

        self.assertEqual(pulse, 1496)

    def test_direction_reversal_limits_angular_change(self):
        """A sudden left-to-right change must obey the supplied delta limit."""
        controller = LaneController()
        controller.previous_command = 3.2

        controller.steering_pulse(500.0, 120.0, 640)

        self.assertAlmostEqual(
            controller.previous_command,
            3.2 - MAX_DELTA_ANGULAR,
        )


class StatusLoggingTests(unittest.TestCase):
    """Check that status severity uses separate logger methods."""

    def test_info_and_warning_use_stable_calls(self):
        """Switching status severity should call each dedicated method."""
        logger = Mock()

        write_status_log(logger, 'lane found')
        write_status_log(logger, 'lane lost', warning=True)
        write_status_log(logger, 'lane found again')

        self.assertEqual(logger.info.call_count, 2)
        logger.warning.assert_called_once_with('lane lost')


class CommandLineTests(unittest.TestCase):
    """Check the requested standalone lane-following speed."""

    def test_default_speed_is_1600_us(self):
        options = build_parser().parse_args([])

        self.assertEqual(SLOW_SPEED_US, 1600)
        self.assertEqual(MAX_SPEED_US, 1600)
        self.assertEqual(options.speed_us, 1600)


if __name__ == '__main__':
    unittest.main()
