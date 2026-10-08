#!/usr/bin/env python3
"""Tune lane HSV limits from a camera without driving the vehicle."""

import argparse

import cv2
import numpy as np

from camera_lane_follow import (
    WHITE_LOWER_HSV,
    WHITE_UPPER_HSV,
    YELLOW_LOWER_HSV,
    YELLOW_UPPER_HSV,
    bird_eye_view,
    lane_color_mask,
)


CONTROL_WINDOW = 'HSV controls'
SOURCE_WINDOW = 'bird eye view'
RAW_MASK_WINDOW = 'raw HSV mask'
LANE_MASK_WINDOW = 'lane ROI mask'
RESULT_WINDOW = 'selected lane color'

TRACKBARS = (
    ('H min', 179),
    ('S min', 255),
    ('V min', 255),
    ('H max', 179),
    ('S max', 255),
    ('V max', 255),
)


def default_hsv_range(color):
    """Return the lane follower's current defaults for one color."""
    if color == 'yellow':
        return YELLOW_LOWER_HSV, YELLOW_UPPER_HSV
    return WHITE_LOWER_HSV, WHITE_UPPER_HSV


def format_hsv_values(lower_hsv, upper_hsv):
    """Return values in a form that can be pasted into the follower."""
    return (
        f'lower={list(lower_hsv)}, upper={list(upper_hsv)}\n'
        f'LOWER_HSV = {tuple(lower_hsv)}\n'
        f'UPPER_HSV = {tuple(upper_hsv)}'
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--camera',
        default='/dev/video0',
        help='카메라 장치 경로(기본 /dev/video0)',
    )
    parser.add_argument(
        '--color',
        choices=('white', 'yellow'),
        default='white',
        help='초기 HSV 값(기본 white)',
    )
    return parser


class HsvTuner:
    """Own the OpenCV camera and HSV trackbar windows."""

    def __init__(self, camera_path, color):
        self.camera_path = camera_path
        self.color = color
        self.capture = None

    def _create_controls(self):
        lower, upper = default_hsv_range(self.color)
        initial_values = (*lower, *upper)

        cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROL_WINDOW, 720, 360)
        for (name, maximum), value in zip(TRACKBARS, initial_values):
            cv2.createTrackbar(
                name,
                CONTROL_WINDOW,
                int(value),
                maximum,
                lambda _value: None,
            )

    def hsv_range(self):
        values = tuple(
            cv2.getTrackbarPos(name, CONTROL_WINDOW)
            for name, _maximum in TRACKBARS
        )
        return values[:3], values[3:]

    @staticmethod
    def _control_panel(lower_hsv, upper_hsv):
        panel = np.zeros((150, 720, 3), dtype=np.uint8)
        lines = (
            f'LOWER HSV = {lower_hsv}',
            f'UPPER HSV = {upper_hsv}',
            'p: print values    q or ESC: quit',
        )
        colors = ((255, 255, 255), (255, 255, 255), (0, 255, 255))
        for index, (line, color) in enumerate(zip(lines, colors)):
            cv2.putText(
                panel,
                line,
                (20, 40 + index * 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
            )
        return panel

    def run(self):
        self.capture = cv2.VideoCapture(self.camera_path, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            print(f'카메라를 열 수 없습니다: {self.camera_path}')
            print('camera_node 등 카메라를 사용 중인 프로그램을 먼저 종료하세요.')
            return 1

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        lower_hsv, upper_hsv = default_hsv_range(self.color)
        print(f'{self.color} HSV 조절을 시작합니다.')
        print(format_hsv_values(lower_hsv, upper_hsv))

        try:
            self._create_controls()
            while True:
                ok, image = self.capture.read()
                if not ok:
                    print('카메라 프레임을 읽을 수 없습니다.')
                    return 1

                bev = bird_eye_view(image)
                lower_hsv, upper_hsv = self.hsv_range()
                hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
                raw_mask = cv2.inRange(
                    hsv,
                    np.array(lower_hsv, dtype=np.uint8),
                    np.array(upper_hsv, dtype=np.uint8),
                )
                lane_mask = lane_color_mask(bev, lower_hsv, upper_hsv)
                selected = cv2.bitwise_and(bev, bev, mask=lane_mask)

                cv2.imshow(SOURCE_WINDOW, bev)
                cv2.imshow(RAW_MASK_WINDOW, raw_mask)
                cv2.imshow(LANE_MASK_WINDOW, lane_mask)
                cv2.imshow(RESULT_WINDOW, selected)
                cv2.imshow(
                    CONTROL_WINDOW,
                    self._control_panel(lower_hsv, upper_hsv),
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('p'):
                    print('\n현재 HSV 값')
                    print(format_hsv_values(lower_hsv, upper_hsv))
        except KeyboardInterrupt:
            print('\n종료 요청')
        finally:
            if self.capture is not None:
                self.capture.release()
            cv2.destroyAllWindows()

        print('\n최종 HSV 값')
        print(format_hsv_values(lower_hsv, upper_hsv))
        return 0


def main(argv=None):
    options = build_parser().parse_args(argv)
    return HsvTuner(options.camera, options.color).run()


if __name__ == '__main__':
    raise SystemExit(main())
