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


class ObstaclePassTests(unittest.TestCase):
    def points(self, forward, lateral):
        return np.array([forward]), np.array([lateral])

    def test_obstacle_ahead_holds_turn(self):
        self.assertTrue(avoid.obstacle_not_passed(self.points(0.4, 0.03), 'left'))
        self.assertTrue(avoid.obstacle_not_passed(self.points(0.4, -0.03), 'right'))

    def test_obstacle_beside_far_side_still_holds_until_behind(self):
        self.assertTrue(avoid.obstacle_not_passed(self.points(0.05, -0.13), 'left'))
        self.assertFalse(avoid.obstacle_not_passed(self.points(-0.05, -0.13), 'left'))

    def test_obstacle_cleared_sideways_releases_turn(self):
        self.assertFalse(avoid.obstacle_not_passed(self.points(0.1, -0.25), 'left'))
        self.assertFalse(avoid.obstacle_not_passed(self.points(0.1, 0.25), 'right'))

    def test_obstacle_in_target_lane_does_not_hold_turn(self):
        self.assertFalse(avoid.obstacle_not_passed(self.points(0.4, 0.3), 'left'))
        self.assertFalse(avoid.obstacle_not_passed(self.points(0.4, -0.3), 'right'))

    def test_far_obstacle_does_not_hold_turn(self):
        self.assertFalse(avoid.obstacle_not_passed(self.points(0.9, 0.0), 'left'))


class StripeReachedTests(unittest.TestCase):
    def test_left_turn_reaches_stripe_near_centre(self):
        self.assertFalse(avoid.stripe_reached(150, 'left', 640))
        self.assertTrue(avoid.stripe_reached(243, 'left', 640))

    def test_right_turn_reaches_stripe_near_centre(self):
        self.assertFalse(avoid.stripe_reached(450, 'right', 640))
        self.assertTrue(avoid.stripe_reached(305, 'right', 640))


if __name__ == '__main__':
    unittest.main()
