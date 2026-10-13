#!/usr/bin/env python3
"""Homography-based bird's-eye-view (BEV) transform for the vehicle camera.

Vehicle frame (all lengths in metres):
    origin  rear-axle centre on the floor
    +X      vehicle right
    +Y      vehicle forward

BEV image frame (pixels):
    u = (x - x_min) * pixels_per_meter          (right is +u)
    v = (y_max - y) * pixels_per_meter          (forward is up, -v)

The calibration stores the metric homography ``image -> floor`` so the BEV
range and scale can be changed later without clicking the points again.
This module does not import ROS and never touches motor, ESC, or PWM outputs.
"""

import datetime
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml


CALIBRATION_FORMAT_VERSION = 1
DEFAULT_CALIBRATION_FILE = 'bev_calibration.yaml'

# Reference rectangle (phone) dimensions.
PHONE_WIDTH_M = 0.0716
PHONE_LENGTH_M = 0.1476
PHONE_THICKNESS_M = 0.0078

# Vehicle geometry measured from the rear-axle centre.
WHEELBASE_M = 0.257
CAMERA_FORWARD_M = 0.320
CAMERA_LATERAL_M = 0.0
CAMERA_HEIGHT_M = 0.116

POINT_LABELS = ('A', 'B', 'C', 'D')
POINT_DESCRIPTIONS = (
    'A: 차량 쪽 왼쪽',
    'B: 차량 쪽 오른쪽',
    'C: 먼 쪽 오른쪽',
    'D: 먼 쪽 왼쪽',
)


class CalibrationError(ValueError):
    """Raised when clicked points or a calibration file cannot be used."""


@dataclass
class BevConfig:
    """Output scale and visible floor range of the BEV image."""

    pixels_per_meter: float = 1000.0  # 10 px/cm
    x_min: float = -0.40
    x_max: float = 0.40
    y_min: float = 0.00
    y_max: float = 0.90

    def __post_init__(self):
        if self.pixels_per_meter <= 0:
            raise CalibrationError('pixels_per_meter는 0보다 커야 합니다')
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise CalibrationError('BEV 표시 범위의 max가 min보다 커야 합니다')

    @property
    def size(self):
        """BEV output size as (width, height) in pixels."""
        width = int(round((self.x_max - self.x_min) * self.pixels_per_meter))
        height = int(round((self.y_max - self.y_min) * self.pixels_per_meter))
        return width, height

    @property
    def origin_pixel(self):
        """BEV pixel of the rear-axle centre (may lie outside the image)."""
        return self.vehicle_to_pixel(0.0, 0.0)

    def ground_to_bev_matrix(self):
        """3x3 affine matrix that maps vehicle metres to BEV pixels."""
        ppm = self.pixels_per_meter
        return np.array([
            [ppm, 0.0, -self.x_min * ppm],
            [0.0, -ppm, self.y_max * ppm],
            [0.0, 0.0, 1.0],
        ])

    def pixel_to_vehicle(self, u, v):
        """BEV pixel (u, v) -> vehicle coordinates (x, y) in metres."""
        x = self.x_min + float(u) / self.pixels_per_meter
        y = self.y_max - float(v) / self.pixels_per_meter
        return x, y

    def vehicle_to_pixel(self, x, y):
        """Vehicle coordinates (x, y) in metres -> BEV pixel (u, v)."""
        u = (float(x) - self.x_min) * self.pixels_per_meter
        v = (self.y_max - float(y)) * self.pixels_per_meter
        return u, v

    def to_dict(self):
        width, height = self.size
        origin_u, origin_v = self.origin_pixel
        return {
            'pixels_per_meter': float(self.pixels_per_meter),
            'pixels_per_cm': float(self.pixels_per_meter) / 100.0,
            'x_range_m': [float(self.x_min), float(self.x_max)],
            'y_range_m': [float(self.y_min), float(self.y_max)],
            'output_size': {'width': width, 'height': height},
            'origin_pixel': {'u': float(origin_u), 'v': float(origin_v)},
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            pixels_per_meter=float(data['pixels_per_meter']),
            x_min=float(data['x_range_m'][0]),
            x_max=float(data['x_range_m'][1]),
            y_min=float(data['y_range_m'][0]),
            y_max=float(data['y_range_m'][1]),
        )


def reference_rectangle(y_near, x_center=0.0,
                        width=PHONE_WIDTH_M, length=PHONE_LENGTH_M):
    """Floor corners A, B, C, D of the reference rectangle in metres.

    A = near-left, B = near-right, C = far-right, D = far-left.
    """
    half = width / 2.0
    return np.array([
        [x_center - half, y_near],
        [x_center + half, y_near],
        [x_center + half, y_near + length],
        [x_center - half, y_near + length],
    ], dtype=np.float64)


def correct_raised_points(points, height,
                          camera_xyz=(CAMERA_LATERAL_M, CAMERA_FORWARD_M,
                                      CAMERA_HEIGHT_M)):
    """Floor point seen along the same camera ray as raised points.

    A corner lifted ``height`` metres (e.g. the phone's top face) appears in
    the image where the camera ray hits the floor further away.  Using that
    floor point keeps the homography a pure floor-plane mapping.  Only the
    camera position is needed, not its tilt.
    """
    points = np.asarray(points, dtype=np.float64)
    if height <= 0:
        return points.copy()
    cam_x, cam_y, cam_h = camera_xyz
    if height >= cam_h:
        raise CalibrationError('기준점 높이가 카메라 높이 이상입니다')
    scale = cam_h / (cam_h - height)
    camera = np.array([cam_x, cam_y])
    return camera + (points - camera) * scale


def _transform(matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)


def validate_image_points(image_points, image_size):
    """Raise CalibrationError if the A, B, C, D clicks are unusable."""
    points = np.asarray(image_points, dtype=np.float64)
    if points.shape != (4, 2):
        raise CalibrationError('기준점 4개가 모두 필요합니다')
    if not np.all(np.isfinite(points)):
        raise CalibrationError('기준점 좌표에 숫자가 아닌 값이 있습니다')

    width, height = image_size
    for label, (u, v) in zip(POINT_LABELS, points):
        if not (0 <= u <= width - 1 and 0 <= v <= height - 1):
            raise CalibrationError(
                f'{label}점 ({u:.1f}, {v:.1f})이 영상 범위 '
                f'{width}x{height} 밖에 있습니다'
            )

    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(points[i] - points[j]) < 5.0:
                raise CalibrationError(
                    f'{POINT_LABELS[i]}점과 {POINT_LABELS[j]}점이 너무 가깝습니다'
                )

    # In image coordinates (v down), A->B->C->D going near-left, near-right,
    # far-right, far-left is a negative-area loop.  Every turn must have the
    # same sign, otherwise the quad is concave, crossed, or in the wrong order.
    crosses = []
    for i in range(4):
        p0, p1, p2 = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        e1, e2 = p1 - p0, p2 - p1
        crosses.append(e1[0] * e2[1] - e1[1] * e2[0])
    crosses = np.array(crosses)
    if np.any(crosses >= 0):
        raise CalibrationError(
            '네 점이 볼록 사각형을 이루지 않거나 순서가 틀렸습니다. '
            'A(가까운 왼쪽) → B(가까운 오른쪽) → C(먼 오른쪽) → D(먼 왼쪽) '
            '순서로 다시 클릭하세요'
        )

    area = 0.5 * abs(sum(
        points[i][0] * points[(i + 1) % 4][1]
        - points[(i + 1) % 4][0] * points[i][1]
        for i in range(4)
    ))
    if area < 200.0:
        raise CalibrationError(f'사각형 면적이 너무 작습니다 ({area:.0f} px²)')

    near_v = (points[0][1] + points[1][1]) / 2.0
    far_v = (points[2][1] + points[3][1]) / 2.0
    if near_v <= far_v:
        raise CalibrationError(
            '가까운 쪽 A, B가 먼 쪽 C, D보다 영상 아래에 있어야 합니다'
        )


def validate_ground_points(ground_points):
    """Raise CalibrationError if the floor rectangle is not usable."""
    points = np.asarray(ground_points, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise CalibrationError('실제 바닥 좌표 4개가 올바르지 않습니다')
    if np.any(points[:, 1] <= 0):
        raise CalibrationError('기준점의 전방 거리(Y)는 0보다 커야 합니다')


def compute_homography(image_points, ground_points):
    """Metric homography mapping image pixels to floor metres."""
    src = np.asarray(image_points, dtype=np.float32)
    dst = np.asarray(ground_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-15:
        raise CalibrationError('Homography 계산 결과가 특이 행렬입니다')
    return matrix / matrix[2, 2]


def reprojection_errors(image_to_ground, image_points, ground_points):
    """Per-point errors in floor metres and image pixels."""
    image_points = np.asarray(image_points, dtype=np.float64)
    ground_points = np.asarray(ground_points, dtype=np.float64)
    projected_ground = _transform(image_to_ground, image_points)
    projected_image = _transform(np.linalg.inv(image_to_ground), ground_points)
    ground_err = np.linalg.norm(projected_ground - ground_points, axis=1)
    image_err = np.linalg.norm(projected_image - image_points, axis=1)
    return ground_err, image_err


@dataclass
class BevCalibration:
    """Floor homography plus BEV output settings."""

    image_size: tuple
    image_points: np.ndarray
    ground_points: np.ndarray
    image_to_ground: np.ndarray
    config: BevConfig = field(default_factory=BevConfig)
    reference: dict = field(default_factory=dict)
    undistortion: dict = None
    created_at: str = ''

    def __post_init__(self):
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.image_points = np.asarray(self.image_points, dtype=np.float64)
        self.ground_points = np.asarray(self.ground_points, dtype=np.float64)
        self.image_to_ground = np.asarray(self.image_to_ground,
                                          dtype=np.float64)
        self.ground_to_image = np.linalg.inv(self.image_to_ground)
        self._undistorter = Undistorter(self.undistortion)
        self._set_bev_matrices()

    def _set_bev_matrices(self):
        self.image_to_bev = (
            self.config.ground_to_bev_matrix() @ self.image_to_ground
        )
        self.image_to_bev /= self.image_to_bev[2, 2]
        self.bev_to_image = np.linalg.inv(self.image_to_bev)
        # BEV pixels behind the camera plane map to mirrored image content;
        # keep only those on the same side of the horizon as the clicks.
        width, height = self.config.size
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        row = self.bev_to_image[2]
        w = row[0] * u + row[1] * v + row[2]
        ref = row @ np.append(_transform(
            self.image_to_bev, self.image_points[:1])[0], 1.0)
        self.bev_valid_mask = w * ref > 0

    @classmethod
    def from_points(cls, image_points, ground_points, image_size,
                    config=None, reference=None, undistortion=None):
        """Validate the clicks and build a calibration from A, B, C, D."""
        validate_image_points(image_points, image_size)
        validate_ground_points(ground_points)
        matrix = compute_homography(image_points, ground_points)
        calibration = cls(
            image_size=image_size,
            image_points=image_points,
            ground_points=ground_points,
            image_to_ground=matrix,
            config=config or BevConfig(),
            reference=dict(reference or {}),
            undistortion=undistortion,
            created_at=datetime.datetime.now().astimezone().isoformat(
                timespec='seconds'),
        )
        calibration.check_sanity()
        return calibration

    def set_config(self, config):
        """Change BEV scale/range; the floor homography is kept."""
        self.config = config
        self._set_bev_matrices()

    @property
    def bev_size(self):
        return self.config.size

    # ----- checks -------------------------------------------------------
    def check_sanity(self):
        """Reject homographies whose horizon falls inside the clicked area."""
        # The denominator of a homography changes sign on the horizon line.
        # All clicked points must be on the same (floor) side of it.
        h = self.image_to_ground
        pts = np.hstack([self.image_points, np.ones((4, 1))])
        w = pts @ h[2]
        if not (np.all(w > 0) or np.all(w < 0)):
            raise CalibrationError('Homography의 지평선이 기준점 사이를 지납니다')
        ground_err, _ = reprojection_errors(
            h, self.image_points, self.ground_points)
        if np.max(ground_err) > 1e-3:
            raise CalibrationError(
                f'기준점 재투영 오차가 비정상적으로 큽니다 '
                f'({np.max(ground_err) * 1000:.2f} mm)'
            )

    def check_image_size(self, width, height):
        """Raise CalibrationError when the live image size differs."""
        if (int(width), int(height)) != tuple(self.image_size):
            raise CalibrationError(
                f'카메라 해상도 {width}x{height}가 캘리브레이션 해상도 '
                f'{self.image_size[0]}x{self.image_size[1]}와 다릅니다. '
                '같은 해상도로 다시 캘리브레이션하세요'
            )

    # ----- coordinate conversion ---------------------------------------
    def pixel_to_vehicle(self, u, v):
        """BEV pixel (u, v) -> vehicle coordinates (x, y) in metres."""
        return self.config.pixel_to_vehicle(u, v)

    def vehicle_to_pixel(self, x, y):
        """Vehicle coordinates (x, y) in metres -> BEV pixel (u, v)."""
        return self.config.vehicle_to_pixel(x, y)

    def pixels_to_vehicle(self, pixels):
        """Vectorised BEV pixels (N, 2) -> vehicle metres (N, 2)."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        ppm = self.config.pixels_per_meter
        x = self.config.x_min + pixels[:, 0] / ppm
        y = self.config.y_max - pixels[:, 1] / ppm
        return np.column_stack([x, y])

    def image_to_vehicle(self, u, v):
        """Camera image pixel -> vehicle floor coordinates in metres.

        ``u, v`` are pixels of the (undistorted, if enabled) camera image.
        """
        x, y = _transform(self.image_to_ground, [[u, v]])[0]
        return float(x), float(y)

    def vehicle_to_image(self, x, y):
        """Vehicle floor coordinates in metres -> camera image pixel."""
        u, v = _transform(self.ground_to_image, [[x, y]])[0]
        return float(u), float(v)

    def is_in_front_of_camera(self, x, y):
        """True if a floor point lies on the visible side of the horizon."""
        w = self.ground_to_image[2] @ np.array([x, y, 1.0])
        ref = self.ground_to_image[2] @ np.append(self.ground_points[0], 1.0)
        return bool(w * ref > 0)

    # ----- image warping -----------------------------------------------
    def undistort(self, image):
        """Apply the optional lens undistortion stored in the calibration."""
        return self._undistorter.apply(image)

    def warp(self, image, undistort=True):
        """Camera image -> BEV image (check_image_size first)."""
        height, width = image.shape[:2]
        self.check_image_size(width, height)
        if undistort:
            image = self.undistort(image)
        bev = cv2.warpPerspective(
            image, self.image_to_bev, self.bev_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        bev[~self.bev_valid_mask] = 0
        return bev

    # ----- persistence -------------------------------------------------
    def to_dict(self):
        ground_err, image_err = reprojection_errors(
            self.image_to_ground, self.image_points, self.ground_points)
        points = []
        for i, label in enumerate(POINT_LABELS):
            points.append({
                'label': label,
                'description': POINT_DESCRIPTIONS[i],
                'image_px': [float(v) for v in self.image_points[i]],
                'ground_m': [float(v) for v in self.ground_points[i]],
                'reprojection_error_mm': float(ground_err[i] * 1000.0),
                'reprojection_error_px': float(image_err[i]),
            })
        return {
            'format_version': CALIBRATION_FORMAT_VERSION,
            'calibrated': True,
            'calibration_date': self.created_at,
            'coordinate_frame': {
                'origin': 'rear axle centre on the floor',
                'x': 'vehicle right (+)',
                'y': 'vehicle forward (+)',
                'unit': 'm',
            },
            'image_size': {
                'width': self.image_size[0],
                'height': self.image_size[1],
            },
            'reference': self.reference,
            'points': points,
            'homography_image_to_ground': self.image_to_ground.tolist(),
            'homography_image_to_bev': self.image_to_bev.tolist(),
            'bev': self.config.to_dict(),
            'undistortion': self.undistortion or {'enabled': False},
            'note': (
                '4점 Homography는 같은 4점의 재투영 오차가 거의 0입니다. '
                '정확도는 추가 기준점이나 줄자 측정으로 확인하세요.'
            ),
        }

    @classmethod
    def from_dict(cls, data):
        if not data or not data.get('calibrated'):
            raise CalibrationError(
                '캘리브레이션되지 않은 파일입니다 (calibrated: false)')
        version = data.get('format_version')
        if version != CALIBRATION_FORMAT_VERSION:
            raise CalibrationError(f'지원하지 않는 format_version: {version}')
        points = data['points']
        if [p['label'] for p in points] != list(POINT_LABELS):
            raise CalibrationError('points는 A, B, C, D 순서여야 합니다')
        image_size = (int(data['image_size']['width']),
                      int(data['image_size']['height']))
        image_points = np.array([p['image_px'] for p in points], dtype=float)
        ground_points = np.array([p['ground_m'] for p in points], dtype=float)
        validate_image_points(image_points, image_size)
        validate_ground_points(ground_points)
        undistortion = data.get('undistortion')
        if undistortion and not undistortion.get('enabled'):
            undistortion = None
        calibration = cls(
            image_size=image_size,
            image_points=image_points,
            ground_points=ground_points,
            image_to_ground=np.array(data['homography_image_to_ground'],
                                     dtype=float),
            config=BevConfig.from_dict(data['bev']),
            reference=data.get('reference', {}),
            undistortion=undistortion,
            created_at=str(data.get('calibration_date', '')),
        )
        calibration.check_sanity()
        return calibration

    def save(self, path=DEFAULT_CALIBRATION_FILE):
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('# BEV calibration generated by bev_calibrate.py\n')
            yaml.safe_dump(self.to_dict(), stream, allow_unicode=True,
                           sort_keys=False, default_flow_style=None)

    @classmethod
    def load(cls, path=DEFAULT_CALIBRATION_FILE, image_size=None):
        """Load a calibration; optionally require a matching image size."""
        try:
            with open(path, encoding='utf-8') as stream:
                data = yaml.safe_load(stream)
        except FileNotFoundError as exc:
            raise CalibrationError(f'파일이 없습니다: {path}') from exc
        except yaml.YAMLError as exc:
            raise CalibrationError(f'YAML 형식 오류: {exc}') from exc
        try:
            calibration = cls.from_dict(data)
        except (KeyError, TypeError, IndexError) as exc:
            raise CalibrationError(f'필수 항목이 없거나 형식이 틀림: {exc}') \
                from exc
        if image_size is not None:
            calibration.check_image_size(*image_size)
        return calibration


class Undistorter:
    """Optional lens undistortion; a no-op when disabled."""

    def __init__(self, undistortion=None):
        self.undistortion = undistortion
        self._maps = None

    @property
    def enabled(self):
        return bool(self.undistortion and self.undistortion.get('enabled'))

    def apply(self, image):
        if not self.enabled:
            return image
        height, width = image.shape[:2]
        if self._maps is None or self._maps[0] != (width, height):
            camera_matrix = np.array(self.undistortion['camera_matrix'],
                                     dtype=np.float64)
            dist = np.array(self.undistortion['dist_coeffs'], dtype=np.float64)
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist, None, camera_matrix,
                (width, height), cv2.CV_16SC2)
            self._maps = ((width, height), map1, map2)
        _, map1, map2 = self._maps
        return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def load_calibration(path=DEFAULT_CALIBRATION_FILE, image_size=None):
    """Convenience wrapper for lane-following code."""
    return BevCalibration.load(path, image_size)


def load_undistortion(path):
    """Read camera_matrix/dist_coeffs from an OpenCV or ROS camera YAML."""
    if path.endswith(('.yaml', '.yml')):
        with open(path, encoding='utf-8') as stream:
            text = stream.read()
        if text.lstrip().startswith('%YAML'):
            data = None
        else:
            data = yaml.safe_load(text)
        if isinstance(data, dict) and 'camera_matrix' in data and isinstance(
                data['camera_matrix'], dict) and 'data' in data[
                    'camera_matrix']:
            # ROS camera_info YAML (camera_calibration package output).
            camera_matrix = np.array(data['camera_matrix']['data'],
                                     dtype=float).reshape(3, 3)
            dist = np.array(data['distortion_coefficients']['data'],
                            dtype=float).ravel()
            return _undistortion_dict(camera_matrix, dist)
    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise CalibrationError(f'렌즈 보정 파일을 열 수 없습니다: {path}')
    camera_matrix = storage.getNode('camera_matrix').mat()
    dist = storage.getNode('dist_coeffs').mat()
    if dist is None:
        dist = storage.getNode('distortion_coefficients').mat()
    storage.release()
    if camera_matrix is None or dist is None:
        raise CalibrationError(
            '렌즈 보정 파일에 camera_matrix와 dist_coeffs가 필요합니다')
    return _undistortion_dict(camera_matrix, dist.ravel())


def _undistortion_dict(camera_matrix, dist):
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise CalibrationError('camera_matrix 형식이 올바르지 않습니다')
    return {
        'enabled': True,
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': [float(v) for v in dist],
    }


def ground_distance(p, q):
    """Euclidean distance between two floor points in metres."""
    return math.hypot(p[0] - q[0], p[1] - q[1])
