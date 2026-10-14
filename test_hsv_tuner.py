"""Unit tests for the standalone HSV tuning helpers."""

import unittest

from camera_lane_follow import (
    WHITE_LOWER_HSV,
    WHITE_UPPER_HSV,
    YELLOW_LOWER_HSV,
    YELLOW_UPPER_HSV,
)
from hsv_tuner import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    brightness_to_trackbar,
    build_parser,
    default_hsv_range,
    format_hsv_values,
    trackbar_to_brightness,
)


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

    def test_brightness_trackbar_covers_camera_range(self):
        self.assertEqual(brightness_to_trackbar(BRIGHTNESS_MIN), 0)
        self.assertEqual(
            brightness_to_trackbar(BRIGHTNESS_MAX),
            BRIGHTNESS_MAX - BRIGHTNESS_MIN,
        )
        self.assertEqual(trackbar_to_brightness(64), 0)

    def test_brightness_trackbar_clamps_out_of_range_values(self):
        self.assertEqual(brightness_to_trackbar(-100), 0)
        self.assertEqual(
            brightness_to_trackbar(100),
            BRIGHTNESS_MAX - BRIGHTNESS_MIN,
        )


if __name__ == '__main__':
    unittest.main()
