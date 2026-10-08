"""Unit tests for the standalone HSV tuning helpers."""

import unittest

from camera_lane_follow import (
    WHITE_LOWER_HSV,
    WHITE_UPPER_HSV,
    YELLOW_LOWER_HSV,
    YELLOW_UPPER_HSV,
)
from hsv_tuner import build_parser, default_hsv_range, format_hsv_values


class HsvTunerTests(unittest.TestCase):
    def test_white_is_the_default_color(self):
        options = build_parser().parse_args([])

        self.assertEqual(options.color, 'white')
        self.assertEqual(
            default_hsv_range(options.color),
            (WHITE_LOWER_HSV, WHITE_UPPER_HSV),
        )

    def test_yellow_uses_lane_follower_values(self):
        self.assertEqual(
            default_hsv_range('yellow'),
            (YELLOW_LOWER_HSV, YELLOW_UPPER_HSV),
        )

    def test_printed_values_can_be_copied(self):
        text = format_hsv_values((1, 2, 3), (40, 50, 60))

        self.assertIn('lower=[1, 2, 3]', text)
        self.assertIn('LOWER_HSV = (1, 2, 3)', text)
        self.assertIn('UPPER_HSV = (40, 50, 60)', text)


if __name__ == '__main__':
    unittest.main()
