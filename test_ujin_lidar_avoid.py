import unittest

import numpy as np

import ujin_lidar_avoid as avoid


class NearbyYellowSideTests(unittest.TestCase):
    def yellow_mask(self, x, y_start=360):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[y_start:480, x:x + 8] = 255
        return mask

    def test_close_yellow_on_left(self):
        self.assertEqual(avoid.visible_yellow_side(self.yellow_mask(180)), 'left')
        self.assertEqual(avoid.yellow_avoidance_side(self.yellow_mask(180)), 'right')

    def test_close_yellow_on_right(self):
        self.assertEqual(avoid.visible_yellow_side(self.yellow_mask(450)), 'right')
        self.assertEqual(avoid.yellow_avoidance_side(self.yellow_mask(450)), 'left')

    def test_yellow_at_screen_edge_still_controls_avoidance(self):
        self.assertEqual(avoid.visible_yellow_side(self.yellow_mask(20)), 'left')
        self.assertEqual(avoid.yellow_avoidance_side(self.yellow_mask(20)), 'right')

    def test_yellow_high_in_image_is_detected_without_roi(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[0:20, 180:188] = 255
        self.assertEqual(avoid.visible_yellow_side(mask), 'left')
        self.assertEqual(avoid.yellow_avoidance_side(mask), 'right')

    def test_no_yellow_has_no_avoidance_direction(self):
        mask = np.zeros((480, 640), dtype=np.uint8)
        self.assertIsNone(avoid.visible_yellow_side(mask))
        self.assertIsNone(avoid.yellow_avoidance_side(mask))


if __name__ == '__main__':
    unittest.main()
