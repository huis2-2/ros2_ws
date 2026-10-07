"""Unit tests for white-lane detection and steering conversion."""

import unittest

from camera_lane_follow import (
    LaneController,
    MAX_DELTA_ANGULAR,
    STEERING_CENTER_US,
    calculate_lane_info,
    detect_edges,
    white_lane_mask,
)

import numpy as np


class LaneDetectionTests(unittest.TestCase):
    """Check image-mask and Hough-line interpretation helpers."""

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

        mask = white_lane_mask(image)

        self.assertEqual(int(mask[75, 320]), 0)
        self.assertEqual(int(mask[390, 320]), 255)

    def test_white_mask_rejects_yellow(self):
        """A saturated yellow marking must not be treated as a white lane."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[350:430, 300:340] = (0, 255, 255)

        mask = white_lane_mask(image)

        self.assertEqual(int(np.count_nonzero(mask)), 0)

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


if __name__ == '__main__':
    unittest.main()
