"""Unit tests for the BEV homography helpers (no camera or ROS needed)."""

import math
import os
import tempfile
import unittest

import numpy as np

from bev_transform import (
    BevCalibration,
    BevConfig,
    CalibrationError,
    correct_raised_points,
    reference_rectangle,
)


IMAGE_SIZE = (640, 480)


def synthetic_camera(tilt_deg=32.0, forward=0.32, height=0.116, focal=500.0):
    """Project vehicle-frame 3D points with an ideal pinhole camera."""
    tilt = math.radians(tilt_deg)
    # Camera axes in the vehicle frame: right, down (image v), forward.
    right = np.array([1.0, 0.0, 0.0])
    forward_axis = np.array([0.0, math.cos(tilt), -math.sin(tilt)])
    down = np.cross(forward_axis, right)
    rotation = np.vstack([right, down, forward_axis])
    centre = np.array([0.0, forward, height])
    cx, cy = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0

    def project(x, y, z=0.0):
        p = rotation @ (np.array([x, y, z]) - centre)
        return focal * p[0] / p[2] + cx, focal * p[1] / p[2] + cy

    return project


def calibrate(project, y_near=0.50, x_center=0.0, height=0.0):
    ground = reference_rectangle(y_near, x_center)
    clicks = [project(x, y, height) for x, y in ground]
    return clicks, ground


class BevTransformTests(unittest.TestCase):
    def setUp(self):
        self.project = synthetic_camera()
        clicks, ground = calibrate(self.project)
        self.calibration = BevCalibration.from_points(
            clicks, ground, IMAGE_SIZE)

    def test_reference_rectangle_order(self):
        a, b, c, d = reference_rectangle(0.5, 0.01)
        np.testing.assert_allclose(a, (0.01 - 0.0358, 0.5))
        np.testing.assert_allclose(b, (0.01 + 0.0358, 0.5))
        np.testing.assert_allclose(c, (0.01 + 0.0358, 0.6476))
        np.testing.assert_allclose(d, (0.01 - 0.0358, 0.6476))

    def test_points_outside_the_rectangle_map_to_the_floor(self):
        for x, y in ((0.0, 0.40), (0.20, 0.80), (-0.25, 0.65)):
            u, v = self.project(x, y)
            gx, gy = self.calibration.image_to_vehicle(u, v)
            self.assertAlmostEqual(gx, x, places=6)
            self.assertAlmostEqual(gy, y, places=6)

    def test_bev_pixel_round_trip_and_scale(self):
        config = BevConfig()
        self.assertEqual(config.pixels_per_meter, 1000.0)
        u, v = config.vehicle_to_pixel(0.1, 0.5)
        x, y = config.pixel_to_vehicle(u, v)
        self.assertAlmostEqual(x, 0.1)
        self.assertAlmostEqual(y, 0.5)
        # Forward is up and right is right in the BEV image.
        u2, v2 = config.vehicle_to_pixel(0.2, 0.6)
        self.assertAlmostEqual(u2 - u, 100.0)
        self.assertAlmostEqual(v2 - v, -100.0)
        self.assertEqual(config.origin_pixel, (400.0, 900.0))

    def test_camera_pixel_lands_on_matching_bev_pixel(self):
        x, y = 0.15, 0.70
        u, v = self.project(x, y)
        bev = self.calibration.image_to_bev @ np.array([u, v, 1.0])
        bev = bev[:2] / bev[2]
        np.testing.assert_allclose(bev, self.calibration.vehicle_to_pixel(x, y),
                                   atol=1e-3)

    def test_warp_checks_resolution(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        width, height = self.calibration.bev_size
        self.assertEqual(self.calibration.warp(image).shape,
                         (height, width, 3))
        with self.assertRaises(CalibrationError):
            self.calibration.warp(np.zeros((720, 1280, 3), dtype=np.uint8))

    def test_wrong_click_order_is_rejected(self):
        clicks, ground = calibrate(self.project)
        swapped = [clicks[1], clicks[0], clicks[2], clicks[3]]
        with self.assertRaises(CalibrationError):
            BevCalibration.from_points(swapped, ground, IMAGE_SIZE)
        far_first = [clicks[3], clicks[2], clicks[1], clicks[0]]
        with self.assertRaises(CalibrationError):
            BevCalibration.from_points(far_first, ground, IMAGE_SIZE)

    def test_points_outside_image_are_rejected(self):
        clicks, ground = calibrate(self.project)
        clicks[2] = (700.0, clicks[2][1])
        with self.assertRaises(CalibrationError):
            BevCalibration.from_points(clicks, ground, IMAGE_SIZE)

    def test_raised_corner_correction(self):
        # Clicking the phone's top face (7.8 mm up) and correcting for it
        # must give the same floor mapping as clicking floor marks.
        height = 0.0078
        clicks, nominal = calibrate(self.project, height=height)
        corrected = correct_raised_points(nominal, height)
        calibration = BevCalibration.from_points(clicks, corrected, IMAGE_SIZE)
        u, v = self.project(0.1, 0.8)
        x, y = calibration.image_to_vehicle(u, v)
        self.assertAlmostEqual(x, 0.1, places=6)
        self.assertAlmostEqual(y, 0.8, places=6)
        # Without the correction the far point is clearly off.
        uncorrected = BevCalibration.from_points(clicks, nominal, IMAGE_SIZE)
        _, y_bad = uncorrected.image_to_vehicle(u, v)
        self.assertGreater(abs(y_bad - 0.8), 0.01)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'bev_calibration.yaml')
            self.calibration.save(path)
            loaded = BevCalibration.load(path, IMAGE_SIZE)
            np.testing.assert_allclose(loaded.image_to_ground,
                                       self.calibration.image_to_ground)
            with self.assertRaises(CalibrationError):
                BevCalibration.load(path, (1280, 720))

    def test_template_file_is_not_usable(self):
        with self.assertRaises(CalibrationError):
            BevCalibration.load(os.path.join(os.path.dirname(__file__),
                                             'bev_calibration.yaml'))


if __name__ == '__main__':
    unittest.main()
