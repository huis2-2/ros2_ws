"""Unit tests for white-lane detection and steering conversion."""

import unittest
from unittest.mock import Mock, patch

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
    draw_sliding_windows,
    fit_lane_track,
    fit_road_center,
    curve_speed_us,
    lane_color_mask,
    sliding_window_track,
    write_status_log,
)

import numpy as np


class LaneDetectionTests(unittest.TestCase):
    """Check image-mask and Hough-line interpretation helpers."""

    def test_requested_hsv_ranges_are_used(self):
        self.assertEqual(WHITE_LOWER_HSV, (0, 0, 252))
        self.assertEqual(WHITE_UPPER_HSV, (0, 116, 255))
        self.assertEqual(YELLOW_LOWER_HSV, (15, 40, 120))
        self.assertEqual(YELLOW_UPPER_HSV, (40, 255, 255))

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

    def test_lane_mask_uses_rectangular_roi(self):
        """The lower ROI should retain equal width at its top and bottom."""
        image = np.full((480, 640, 3), 255, dtype=np.uint8)

        mask = lane_color_mask(image, WHITE_LOWER_HSV, WHITE_UPPER_HSV)

        self.assertEqual(int(mask[300, 20]), 255)
        self.assertEqual(int(mask[450, 20]), 255)
        self.assertEqual(int(mask[300, 5]), 0)

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

    def test_sliding_windows_follow_mask_pixels_upward(self):
        """Windows should recenter as a white lane bends across the ROI."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        for y in range(280, 470):
            x = 320 + (470 - y) // 4
            mask[y, x - 6:x + 7] = 255

        windows, centers, selected_x, selected_y = sliding_window_track(
            mask,
            320,
        )

        self.assertEqual(len(windows), 8)
        self.assertEqual(len(centers), 8)
        self.assertGreater(selected_x.size, 0)
        self.assertEqual(selected_x.size, selected_y.size)
        self.assertGreater(centers[-1][0], centers[0][0])

    def test_sliding_window_overlay_draws_tracking_colors(self):
        """The preview should contain window and selected-pixel colors."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[280:470, 314:327] = 255

        display = draw_sliding_windows(image, mask, 320)

        self.assertTrue(np.any(np.all(display == (255, 255, 0), axis=2)))
        self.assertTrue(np.any(np.all(display == (255, 0, 255), axis=2)))


class CurveTrackingTests(unittest.TestCase):
    """Exercise the actual mask-to-steering path on synthetic road curves."""

    @staticmethod
    def road_mask(direction=0, width=640, height=480):
        mask = np.zeros((height, width), dtype=np.uint8)
        for y in range(int(height * 0.58), int(height * 0.98) + 1):
            progress = (0.98 - y / height) / 0.40
            x = int(width * (0.5 + direction * 0.18 * progress ** 2))
            radius = max(3, int(width * 0.01))
            mask[y, x - radius:x + radius + 1] = 255
        return mask

    def test_visible_left_and_right_curves_control_in_matching_direction(self):
        for direction in (-1, 0, 1):
            with self.subTest(direction=direction):
                stripe = self.road_mask(direction)
                mask = np.roll(stripe, -160, axis=1) | np.roll(stripe, 160, axis=1)
                image = np.repeat(mask[:, :, None], 3, axis=2)
                with patch('camera_lane_follow.bird_eye_view', side_effect=lambda value: value):
                    observation, _, _, _ = detect_white_lane(image)
                self.assertIsNotNone(observation)
                controller = LaneController()
                pulse = controller.steering_pulse(observation.center_x, observation.angle_deg, 640)
                speed = curve_speed_us(1600, observation.angle_deg, pulse)
                if direction == 0:
                    self.assertEqual(pulse, STEERING_CENTER_US)
                    self.assertEqual(speed, 1600)
                else:
                    self.assertGreater((STEERING_CENTER_US - pulse) * direction, 0)
                    self.assertLess(speed, 1600)
                self.assertGreater(observation.track[2].size, 0)
                self.assertAlmostEqual(observation.center_x,
                                       observation.curve[observation.curve[:, 1] == observation.target_y, 0][0],
                                       delta=1)

    def test_empty_and_short_tracks_do_not_produce_a_target(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        self.assertIsNone(fit_lane_track(mask))
        mask[430:471, 310:331] = 255
        self.assertIsNone(fit_lane_track(mask))

    def test_white_left_yellow_right_follow_center_not_boundary(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[278:471, 150:171] = (255, 255, 255)
        image[278:471, 470:491] = (0, 255, 255)
        with patch('camera_lane_follow.bird_eye_view', side_effect=lambda value: value):
            observation, _, _, yellow = detect_white_lane(image)
        self.assertIsNotNone(observation)
        self.assertGreater(np.count_nonzero(yellow), 0)
        self.assertAlmostEqual(observation.center_x, 320, delta=1)
        pulse = LaneController().steering_pulse(observation.center_x, observation.angle_deg, 640)
        self.assertEqual(pulse, STEERING_CENTER_US)

    def test_one_boundary_cannot_be_mistaken_for_road_center(self):
        self.assertIsNone(fit_road_center(self.road_mask()))

    def test_pair_offset_steers_toward_middle(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[278:471, 220:241] = 255
        mask[278:471, 540:561] = 255
        fit = fit_road_center(mask)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit[0], 390, delta=1)
        self.assertLess(LaneController().steering_pulse(fit[0], fit[1], 640), STEERING_CENTER_US)

    def test_increased_cruise_and_curve_outputs(self):
        self.assertEqual(curve_speed_us(1620, 90, STEERING_CENTER_US), 1620)
        self.assertEqual(curve_speed_us(1620, 120, 1400), 1590)

    def test_missing_window_has_no_fabricated_center(self):
        mask = self.road_mask()
        mask[350:375] = 0
        track = sliding_window_track(mask, 320)
        self.assertIn(None, track[1])
        display = draw_sliding_windows(np.zeros((480, 640, 3), dtype=np.uint8), mask, 320, track)
        self.assertTrue(np.any(np.all(display == (0, 0, 255), axis=2)))

    def test_previous_lane_selects_same_stripe_among_two_candidates(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[278:471, 190:211] = 255
        mask[278:471, 430:451] = 255
        self.assertAlmostEqual(fit_lane_track(mask, 200)[0], 200, delta=1)
        self.assertAlmostEqual(fit_lane_track(mask, 440)[0], 440, delta=1)
        self.assertIsNone(fit_lane_track(mask, 50))

    def test_smaller_resolution_retains_curve_direction(self):
        fit = fit_lane_track(self.road_mask(1, 320, 240))
        self.assertIsNotNone(fit)
        self.assertGreater(fit[0], 160)
        self.assertGreater(fit[1], 90)

    def test_curve_speed_respects_lower_user_speed_and_neutral(self):
        for speed in (1500, 1540, 1560, 1600):
            for angle in (30, 90, 150):
                result = curve_speed_us(speed, angle, 1880)
                self.assertGreaterEqual(result, 1500)
                self.assertLessEqual(result, speed)


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

    def test_default_speed_is_1620_us(self):
        options = build_parser().parse_args([])

        self.assertEqual(SLOW_SPEED_US, 1620)
        self.assertEqual(MAX_SPEED_US, 1620)
        self.assertEqual(options.speed_us, 1620)


if __name__ == '__main__':
    unittest.main()
