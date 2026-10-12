"""ROS worker shutdown test with simulated inference and hardware only.

Run after sourcing /opt/ros/jazzy/setup.bash. No device is opened.
"""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import rclpy
from sensor_msgs.msg import Image

import camera_lane_follow_dl_jetson_v3 as follower


class ShutdownTests(unittest.TestCase):
    def test_shutdown_stops_before_inflight_inference_returns(self):
        entered = threading.Event()
        release = threading.Event()
        hardware = Mock()
        segmenter = SimpleNamespace(
            backend='mock', sess=Mock(), warmup_ms=0,
            input_width=320, input_height=256, last_infer_ms=0,
        )

        def predict(frame):
            entered.set()
            release.wait(5)
            return np.zeros(frame.shape[:2], np.uint8), np.zeros(frame.shape[:2], np.uint8)

        segmenter.predict = predict
        message = Image(height=48, width=64, encoding='bgr8', step=192,
                        data=bytes(48 * 64 * 3))
        lane = SimpleNamespace(x=32., angle=90.)

        def spin(node):
            node.cb(message)
            self.assertTrue(entered.wait(2), 'Inference worker did not start')
            try:
                node.destroy_node()
                self.assertFalse(hardware.command.call_args.args[1])
                calls_at_stop = hardware.command.call_count
                release.set()
                node.worker.join(2)
                self.assertFalse(node.worker.is_alive())
                self.assertEqual(hardware.command.call_count, calls_at_stop)
            finally:
                release.set()

        # Skip only the initial 3-second ESC wait; preserve worker sleep calls.
        sleep = time.sleep
        with patch.object(follower, 'Segmenter', return_value=segmenter), \
                patch.object(follower, 'Hardware', return_value=hardware), \
                patch.object(follower.LaneTracker, 'update', return_value=(lane, [], [])), \
                patch.object(rclpy, 'spin', side_effect=spin), \
                patch.object(follower.time, 'sleep', side_effect=lambda t: sleep(t) if t != 3 else None), \
                patch('sys.argv', ['follower', '--drive', '--allow-drive']):
            follower.main()
        hardware.close.assert_called_once()
        self.assertTrue(all(not call.args[1] for call in hardware.command.call_args_list))


if __name__ == '__main__':
    unittest.main()
