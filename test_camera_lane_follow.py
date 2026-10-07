"""Unit tests for yellow-lane detection and steering conversion."""

import unittest

from camera_lane_follow import (
    LaneController,
    STEERING_CENTER_US,
    calculate_lane_info,
    yellow_lane_mask,
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

    def test_yellow_mask_ignores_upper_image(self):
        """Only yellow pixels in the lower road ROI should remain."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[50:100, 300:340] = (0, 255, 255)
        image[350:430, 300:340] = (0, 255, 255)

        mask = yellow_lane_mask(image)

        self.assertEqual(int(mask[75, 320]), 0)
        self.assertEqual(int(mask[390, 320]), 255)


class LaneControllerTests(unittest.TestCase):
    """Check calibrated steering direction and rate limiting."""

    def test_target_lane_keeps_steering_centered(self):
        """A lane at the configured target should retain center steering."""
        controller = LaneController()

        pulse = controller.steering_pulse(264.0, 90.0, 640)

        self.assertEqual(pulse, STEERING_CENTER_US)

    def test_left_command_is_rate_limited(self):
        """Left error should increase the pulse by only one frame step."""
        controller = LaneController()

        pulse = controller.steering_pulse(100.0, 60.0, 640)

        self.assertEqual(pulse, STEERING_CENTER_US + 12)

    def test_right_command_is_rate_limited(self):
        """Right error should decrease the pulse by only one frame step."""
        controller = LaneController()

        pulse = controller.steering_pulse(500.0, 120.0, 640)

        self.assertEqual(pulse, STEERING_CENTER_US - 12)


if __name__ == '__main__':
    unittest.main()
