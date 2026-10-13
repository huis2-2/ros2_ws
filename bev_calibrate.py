#!/usr/bin/env python3
"""Interactive BEV calibration from a ROS 2 camera topic.

Click the floor corners A, B, C, D of a known rectangle (phone outline) in the
"Camera Calibration" window and check the live result in "BEV Preview".
This tool only subscribes to the camera; it never drives motor, ESC, or PWM.

Keys: r reset, u undo, c clear check points, s save, l load, q/ESC quit.
"""

import argparse
import math
import os
import threading
import time

import cv2
import numpy as np
import yaml

from bev_transform import (
    CAMERA_FORWARD_M,
    CAMERA_HEIGHT_M,
    CAMERA_LATERAL_M,
    DEFAULT_CALIBRATION_FILE,
    PHONE_LENGTH_M,
    PHONE_WIDTH_M,
    POINT_DESCRIPTIONS,
    POINT_LABELS,
    WHEELBASE_M,
    BevCalibration,
    BevConfig,
    CalibrationError,
    Undistorter,
    correct_raised_points,
    ground_distance,
    load_undistortion,
    reference_rectangle,
)


CAMERA_WINDOW = 'Camera Calibration'
BEV_WINDOW = 'BEV Preview'
CAMERA_TOPIC = '/camera/image_raw'
IMAGE_TIMEOUT_SEC = 1.0

# Trackbar ranges (mm).  Lateral offset is shifted so the slider stays >= 0.
NEAR_Y_MAX_MM = 2000
CENTER_X_LIMIT_MM = 500

POINT_COLORS = (
    (0, 0, 255),    # A red
    (0, 165, 255),  # B orange
    (0, 255, 0),    # C green
    (255, 0, 0),    # D blue
)
CHECK_COLOR = (255, 0, 255)
MARKER_COLOR = (0, 255, 255)


def parse_point_mm(text):
    """Parse 'x_mm,y_mm' for --check-point."""
    try:
        x_text, y_text = text.split(',')
        return float(x_text) / 1000.0, float(y_text) / 1000.0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'"{text}"는 x_mm,y_mm 형식이어야 합니다 (예: 0,600)') from exc


def parse_range_m(text):
    try:
        low, high = (float(v) for v in text.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'"{text}"는 min,max 형식이어야 합니다 (예: -0.4,0.4)') from exc
    if high <= low:
        raise argparse.ArgumentTypeError(f'"{text}": max가 min보다 커야 합니다')
    return low, high


def build_parser():
    parser = argparse.ArgumentParser(
        description='ROS 2 카메라 영상으로 BEV Homography를 캘리브레이션합니다 '
                    '(모터/ESC/PWM은 사용하지 않음).')
    parser.add_argument('--config', help='실행 인자 기본값을 담은 YAML 파일')
    parser.add_argument('--topic', default=CAMERA_TOPIC, help='카메라 토픽')
    parser.add_argument('--calibration', default=DEFAULT_CALIBRATION_FILE,
                        help='저장/불러오기 파일 (기본: %(default)s)')
    parser.add_argument('--no-autoload', action='store_true',
                        help='시작할 때 기존 캘리브레이션을 불러오지 않음')
    parser.add_argument('--near-y-mm', type=float, default=500.0,
                        help='뒷바퀴 차축 중심 → 가까운 기준선(A-B) 전방 거리')
    parser.add_argument('--center-x-mm', type=float, default=0.0,
                        help='기준 사각형 중심의 좌우 위치 (오른쪽 +)')
    parser.add_argument('--rect-width-mm', type=float,
                        default=PHONE_WIDTH_M * 1000.0,
                        help='기준 사각형 가로(좌우) 길이')
    parser.add_argument('--rect-length-mm', type=float,
                        default=PHONE_LENGTH_M * 1000.0,
                        help='기준 사각형 세로(전방) 길이')
    parser.add_argument('--reference-height-mm', type=float, default=0.0,
                        help='클릭한 점의 바닥 위 높이. 바닥 표시점은 0, '
                             '휴대폰 윗면 모서리를 직접 쓰면 7.8')
    parser.add_argument('--camera-forward-mm', type=float,
                        default=CAMERA_FORWARD_M * 1000.0)
    parser.add_argument('--camera-lateral-mm', type=float,
                        default=CAMERA_LATERAL_M * 1000.0)
    parser.add_argument('--camera-height-mm', type=float,
                        default=CAMERA_HEIGHT_M * 1000.0)
    parser.add_argument('--px-per-cm', type=float, default=10.0,
                        help='BEV 축척 (기본 10 px/cm)')
    parser.add_argument('--x-range-m', type=parse_range_m, default=(-0.40, 0.40),
                        help='BEV 좌우 표시 범위 min,max (m)')
    parser.add_argument('--y-range-m', type=parse_range_m, default=(-0.05, 0.90),
                        help='BEV 전방 표시 범위 min,max (m)')
    parser.add_argument('--undistort',
                        help='선택 사항: 렌즈 왜곡 보정 파일 '
                             '(ROS camera_info YAML 또는 OpenCV YAML)')
    parser.add_argument('--check-point', type=parse_point_mm, action='append',
                        default=[], metavar='X_MM,Y_MM',
                        help='줄자로 측정한 검증 지점. 두 창에 노란 + 로 표시 '
                             '(여러 번 지정 가능)')
    return parser


def parse_args(argv=None):
    """Parse arguments; values in --config become the defaults."""
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        try:
            with open(pre.config, encoding='utf-8') as stream:
                values = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError) as exc:
            parser.error(f'설정 파일을 읽을 수 없습니다: {exc}')
        if not isinstance(values, dict):
            parser.error('설정 파일은 key: value 형식이어야 합니다')
        known = {action.dest for action in parser._actions}
        defaults = {}
        for key, value in values.items():
            dest = str(key).replace('-', '_')
            if dest not in known:
                parser.error(f'설정 파일에 알 수 없는 항목: {key}')
            if dest in ('x_range_m', 'y_range_m'):
                value = parse_range_m(','.join(str(v) for v in value))
            elif dest == 'check_point':
                value = [parse_point_mm(f'{p[0]},{p[1]}') for p in value]
            defaults[dest] = value
        parser.set_defaults(**defaults)
    return parser.parse_known_args(argv)


class FrameBuffer:
    """Latest camera frame shared between the ROS thread and the UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = None

    def put(self, frame):
        with self._lock:
            self._frame = frame
            self._stamp = time.monotonic()

    def get(self):
        with self._lock:
            return self._frame, self._stamp


class BevCalibrationApp:
    """OpenCV UI state: clicks, live homography, drawing, and keys."""

    def __init__(self, options, frames):
        self.options = options
        self.frames = frames
        self.config = BevConfig(
            pixels_per_meter=options.px_per_cm * 100.0,
            x_min=options.x_range_m[0], x_max=options.x_range_m[1],
            y_min=options.y_range_m[0], y_max=options.y_range_m[1],
        )
        self.undistortion = None
        if options.undistort:
            self.undistortion = load_undistortion(options.undistort)
            print(f'렌즈 왜곡 보정 사용: {options.undistort}')
        self.undistorter = Undistorter(self.undistortion)

        self.near_y_mm = float(options.near_y_mm)
        self.center_x_mm = float(options.center_x_mm)
        self.points = []
        self.check_points = []      # (u, v) clicks in the camera image
        self.bev_probe = None       # (u, v) click in the BEV image
        self.mouse = None
        self.calibration = None
        self.status = '네 점을 클릭하세요'
        self.status_error = False
        self.image_size = None
        self.pending_load = not options.no_autoload
        self.last_error = None

    # ----- setup --------------------------------------------------------
    def create_windows(self):
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(BEV_WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('near Y mm', CAMERA_WINDOW,
                           int(round(np.clip(self.near_y_mm, 0,
                                             NEAR_Y_MAX_MM))),
                           NEAR_Y_MAX_MM, lambda _value: None)
        cv2.createTrackbar(f'center X mm +{CENTER_X_LIMIT_MM}', CAMERA_WINDOW,
                           int(round(np.clip(self.center_x_mm,
                                             -CENTER_X_LIMIT_MM,
                                             CENTER_X_LIMIT_MM)
                                     + CENTER_X_LIMIT_MM)),
                           2 * CENTER_X_LIMIT_MM, lambda _value: None)
        cv2.setMouseCallback(CAMERA_WINDOW, self.on_camera_mouse)
        cv2.setMouseCallback(BEV_WINDOW, self.on_bev_mouse)

    def set_trackbars(self):
        cv2.setTrackbarPos('near Y mm', CAMERA_WINDOW,
                           int(round(np.clip(self.near_y_mm, 0,
                                             NEAR_Y_MAX_MM))))
        cv2.setTrackbarPos(f'center X mm +{CENTER_X_LIMIT_MM}', CAMERA_WINDOW,
                           int(round(np.clip(self.center_x_mm,
                                             -CENTER_X_LIMIT_MM,
                                             CENTER_X_LIMIT_MM)
                                     + CENTER_X_LIMIT_MM)))

    def poll_trackbars(self):
        """Recompute when the distance sliders move."""
        near = float(cv2.getTrackbarPos('near Y mm', CAMERA_WINDOW))
        center = float(cv2.getTrackbarPos(
            f'center X mm +{CENTER_X_LIMIT_MM}', CAMERA_WINDOW)
            - CENTER_X_LIMIT_MM)
        # Slider positions are whole mm; keep fractional CLI values until
        # the user actually moves a slider.
        if round(self.near_y_mm) != near or round(self.center_x_mm) != center:
            self.near_y_mm = near
            self.center_x_mm = center
            print(f'입력 거리 변경: near Y={near:.0f} mm, '
                  f'center X={center:+.0f} mm')
            self.recompute()

    # ----- reference geometry ------------------------------------------
    def camera_xyz(self):
        o = self.options
        return (o.camera_lateral_mm / 1000.0, o.camera_forward_mm / 1000.0,
                o.camera_height_mm / 1000.0)

    def nominal_ground_points(self):
        o = self.options
        return reference_rectangle(
            self.near_y_mm / 1000.0, self.center_x_mm / 1000.0,
            o.rect_width_mm / 1000.0, o.rect_length_mm / 1000.0)

    def ground_points(self):
        return correct_raised_points(
            self.nominal_ground_points(),
            self.options.reference_height_mm / 1000.0, self.camera_xyz())

    def reference_info(self):
        o = self.options
        nominal = self.nominal_ground_points()
        return {
            'near_y_mm': float(self.near_y_mm),
            'center_x_mm': float(self.center_x_mm),
            'rect_width_mm': round(float(o.rect_width_mm), 3),
            'rect_length_mm': round(float(o.rect_length_mm), 3),
            'reference_height_mm': float(o.reference_height_mm),
            'camera_position_mm': [float(o.camera_lateral_mm),
                                   float(o.camera_forward_mm),
                                   float(o.camera_height_mm)],
            'nominal_corners_m': [[float(v) for v in p] for p in nominal],
            'height_corrected': o.reference_height_mm > 0,
        }

    # ----- calibration --------------------------------------------------
    def set_status(self, text, error=False):
        self.status = text
        self.status_error = error
        if error:
            if text != self.last_error:
                print(f'[오류] {text}')
            self.last_error = text
        else:
            self.last_error = None
            print(text)

    def recompute(self):
        """Build a calibration from the clicked points, if all four exist."""
        self.calibration = None
        if len(self.points) < 4:
            nxt = len(self.points)
            self.set_status(f'다음 클릭: {POINT_DESCRIPTIONS[nxt]}')
            return
        if self.image_size is None:
            return
        try:
            self.calibration = BevCalibration.from_points(
                self.points, self.ground_points(), self.image_size,
                config=self.config, reference=self.reference_info(),
                undistortion=self.undistortion)
        except (CalibrationError, cv2.error, np.linalg.LinAlgError) as exc:
            self.set_status(f'Homography 계산 실패: {exc}', error=True)
            return
        self.set_status('Homography 계산 완료. BEV 확인 후 s로 저장')
        self.print_summary()

    def print_summary(self):
        cal = self.calibration
        print('기준점  픽셀(u,v)          실제(x,y) m')
        for label, px, gm in zip(POINT_LABELS, cal.image_points,
                                 cal.ground_points):
            print(f'  {label}   ({px[0]:7.1f},{px[1]:7.1f})  '
                  f'({gm[0]:+.4f}, {gm[1]:.4f})')
        print('image->ground Homography:')
        print(np.array2string(cal.image_to_ground, precision=6,
                              suppress_small=False))

    def reset(self):
        self.points = []
        self.check_points = []
        self.bev_probe = None
        self.calibration = None
        self.set_status('기준점 초기화. ' + POINT_DESCRIPTIONS[0] + ' 클릭')

    def save(self):
        if self.calibration is None:
            self.set_status('저장 안 함: 유효한 캘리브레이션이 없습니다 '
                            '(네 점을 올바르게 클릭하세요)', error=True)
            return
        try:
            # Re-validate against the current camera image before writing.
            self.calibration.check_image_size(*self.image_size)
            self.calibration.check_sanity()
            self.calibration.save(self.options.calibration)
        except (CalibrationError, OSError, cv2.error) as exc:
            self.set_status(f'저장 실패: {exc}', error=True)
            return
        self.set_status(
            f'저장 완료: {os.path.abspath(self.options.calibration)}')

    def load(self, quiet_missing=False):
        path = self.options.calibration
        if quiet_missing and not os.path.exists(path):
            self.set_status(f'저장된 캘리브레이션 없음 ({path}). '
                            + POINT_DESCRIPTIONS[0] + ' 클릭')
            return
        try:
            loaded = BevCalibration.load(path)
        except CalibrationError as exc:
            self.set_status(f'불러오기 실패 ({path}): {exc}', error=True)
            return
        if self.image_size is not None:
            try:
                loaded.check_image_size(*self.image_size)
            except CalibrationError as exc:
                self.set_status(f'불러온 값을 사용하지 않습니다: {exc}',
                                error=True)
                return
        # Keep the BEV view settings of this run, use the saved homography.
        loaded.set_config(self.config)
        reference = loaded.reference or {}
        self.near_y_mm = float(reference.get('near_y_mm', self.near_y_mm))
        self.center_x_mm = float(reference.get('center_x_mm',
                                               self.center_x_mm))
        # Recomputing from these clicks must reproduce the saved geometry.
        o = self.options
        o.rect_width_mm = float(reference.get('rect_width_mm',
                                              o.rect_width_mm))
        o.rect_length_mm = float(reference.get('rect_length_mm',
                                               o.rect_length_mm))
        o.reference_height_mm = float(reference.get('reference_height_mm',
                                                    o.reference_height_mm))
        camera_mm = reference.get('camera_position_mm')
        if camera_mm:
            (o.camera_lateral_mm, o.camera_forward_mm,
             o.camera_height_mm) = (float(v) for v in camera_mm)
        self.set_trackbars()
        self.points = [tuple(p) for p in loaded.image_points]
        self.check_points = []
        if loaded.undistortion != self.undistortion:
            state = '사용' if loaded.undistortion else '사용 안 함'
            print(f'불러온 파일의 렌즈 왜곡 보정 설정({state})을 적용합니다')
            self.undistortion = loaded.undistortion
            self.undistorter = Undistorter(self.undistortion)
        self.calibration = loaded
        self.set_status(f'불러오기 완료: {path} ({loaded.created_at})')
        self.print_summary()

    # ----- mouse --------------------------------------------------------
    def on_camera_mouse(self, event, u, v, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (u, v)
            return
        if event != cv2.EVENT_LBUTTONDOWN or self.image_size is None:
            return
        width, height = self.image_size
        if not (0 <= u < width and 0 <= v < height):
            self.set_status(f'영상 범위 밖 클릭 ({u}, {v}) 무시', error=True)
            return
        if len(self.points) < 4:
            index = len(self.points)
            self.points.append((float(u), float(v)))
            print(f'{index + 1}. {POINT_DESCRIPTIONS[index]} = ({u}, {v})')
            self.recompute()
            return
        if self.calibration is None:
            self.set_status('4점이 유효하지 않습니다. r로 초기화 후 다시 클릭',
                            error=True)
            return
        # Extra clicks are verification points.
        self.check_points.append((float(u), float(v)))
        x, y = self.calibration.image_to_vehicle(u, v)
        text = f'검증점 {len(self.check_points)}: x={x * 100:+.1f} cm, ' \
               f'y={y * 100:.1f} cm'
        if len(self.check_points) >= 2:
            pu, pv = self.check_points[-2]
            prev = self.calibration.image_to_vehicle(pu, pv)
            text += f', 이전 검증점과 거리 {ground_distance(prev, (x, y)) * 100:.1f} cm'
        self.set_status(text)

    def on_bev_mouse(self, event, u, v, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and self.calibration is not None:
            self.bev_probe = (u, v)
            x, y = self.calibration.pixel_to_vehicle(u, v)
            print(f'BEV 클릭 ({u}, {v}) px → x={x * 100:+.1f} cm, '
                  f'y={y * 100:.1f} cm')

    # ----- drawing ------------------------------------------------------
    def draw_camera(self, image):
        display = image.copy()
        cal = self.calibration
        if cal is not None:
            self.draw_floor_grid_on_camera(display, cal)
        pts = [tuple(int(round(c)) for c in p) for p in self.points]
        for i in range(len(pts) - 1):
            cv2.line(display, pts[i], pts[i + 1], (0, 255, 255), 1,
                     cv2.LINE_AA)
        if len(pts) == 4:
            cv2.line(display, pts[3], pts[0], (0, 255, 255), 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(display, p, 4, POINT_COLORS[i], -1, cv2.LINE_AA)
            cv2.circle(display, p, 9, POINT_COLORS[i], 1, cv2.LINE_AA)
            cv2.putText(display, f'{i + 1}:{POINT_LABELS[i]}',
                        (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, f'{i + 1}:{POINT_LABELS[i]}',
                        (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        POINT_COLORS[i], 1, cv2.LINE_AA)
        if cal is not None:
            for i, (u, v) in enumerate(self.check_points):
                x, y = cal.image_to_vehicle(u, v)
                p = (int(round(u)), int(round(v)))
                cv2.drawMarker(display, p, CHECK_COLOR, cv2.MARKER_TILTED_CROSS,
                               12, 2)
                put_label(display, f'{x * 100:+.1f},{y * 100:.1f}cm',
                          (p[0] + 6, p[1] + 16), CHECK_COLOR)
            for x, y in self.options.check_point:
                if not cal.is_in_front_of_camera(x, y):
                    continue
                u, v = cal.vehicle_to_image(x, y)
                if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
                    p = (int(round(u)), int(round(v)))
                    cv2.drawMarker(display, p, MARKER_COLOR, cv2.MARKER_CROSS,
                                   14, 1)
        self.draw_magnifier(display, image)

        lines = []
        if len(self.points) < 4:
            lines.append(f'Click {len(self.points) + 1}/4: '
                         + ('A near-left', 'B near-right', 'C far-right',
                            'D far-left')[len(self.points)])
        else:
            lines.append('Extra clicks = check points (x,y cm)')
        lines.append(f'near Y={self.near_y_mm:.0f}mm  center X='
                     f'{self.center_x_mm:+.0f}mm  ref h='
                     f'{self.options.reference_height_mm:.1f}mm')
        lines.append('r reset  u undo  c clear checks  s save  l load  q quit')
        state = 'OK' if cal is not None else (
            'ERROR - see terminal' if self.status_error else 'not calibrated')
        lines.append(f'Homography: {state}')
        for i, text in enumerate(lines):
            color = (0, 0, 255) if (i == 3 and cal is None
                                    and self.status_error) else (255, 255, 255)
            put_label(display, text, (8, 20 + 20 * i), color, 0.5)
        return display

    def draw_floor_grid_on_camera(self, display, cal):
        """Project 10 cm floor lines into the camera image for a quick check."""
        cfg = self.config
        height, width = display.shape[:2]
        samples = np.linspace(0, 1, 40)
        ys = np.arange(math.ceil(max(cfg.y_min, 0.0) / 0.1) * 0.1,
                       cfg.y_max + 1e-9, 0.1)
        xs = np.arange(math.ceil(cfg.x_min / 0.1) * 0.1, cfg.x_max + 1e-9, 0.1)
        polylines = []
        for y in ys:
            polylines.append([(cfg.x_min + (cfg.x_max - cfg.x_min) * s, y)
                              for s in samples])
        for x in xs:
            polylines.append([(x, max(cfg.y_min, 0.0)
                               + (cfg.y_max - max(cfg.y_min, 0.0)) * s)
                              for s in samples])
        for line in polylines:
            pts = []
            for x, y in line:
                if not cal.is_in_front_of_camera(x, y):
                    continue
                u, v = cal.vehicle_to_image(x, y)
                if -width <= u <= 2 * width and -height <= v <= 2 * height:
                    pts.append((int(round(u)), int(round(v))))
            if len(pts) >= 2:
                cv2.polylines(display, [np.array(pts, dtype=np.int32)], False,
                              (200, 200, 0), 1, cv2.LINE_AA)
        # Vehicle centre line (x = 0).
        pts = []
        for s in samples:
            y = max(cfg.y_min, 0.0) + (cfg.y_max - max(cfg.y_min, 0.0)) * s
            if cal.is_in_front_of_camera(0.0, y):
                u, v = cal.vehicle_to_image(0.0, y)
                pts.append((int(round(u)), int(round(v))))
        if len(pts) >= 2:
            cv2.polylines(display, [np.array(pts, dtype=np.int32)], False,
                          (0, 0, 255), 1, cv2.LINE_AA)

    def draw_magnifier(self, display, image, radius=20, zoom=4):
        """Zoomed inset around the cursor for precise corner clicks."""
        if self.mouse is None:
            return
        u, v = self.mouse
        height, width = image.shape[:2]
        if not (0 <= u < width and 0 <= v < height):
            return
        padded = cv2.copyMakeBorder(image, radius, radius, radius, radius,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = padded[v:v + 2 * radius + 1, u:u + 2 * radius + 1]
        size = (2 * radius + 1) * zoom
        inset = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
        c = radius * zoom + zoom // 2
        cv2.line(inset, (c, 0), (c, size - 1), (0, 255, 0), 1)
        cv2.line(inset, (0, c), (size - 1, c), (0, 255, 0), 1)
        if size + 10 > width or size + 10 > height:
            return
        # Put the inset in the corner away from the cursor.
        x0 = width - size - 5 if u < width / 2 else 5
        y0 = height - size - 5
        display[y0:y0 + size, x0:x0 + size] = inset
        cv2.rectangle(display, (x0, y0), (x0 + size - 1, y0 + size - 1),
                      (0, 255, 0), 1)
        put_label(display, f'({u},{v})', (x0 + 4, y0 + 16), (0, 255, 0), 0.45)

    def draw_bev(self, image):
        cal = self.calibration
        cfg = self.config
        width, height = cfg.size
        if cal is None:
            bev = np.zeros((height, width, 3), dtype=np.uint8)
            put_label(bev, 'No valid calibration - click A,B,C,D', (20, 40),
                      (0, 0, 255), 0.7)
        else:
            bev = cv2.warpPerspective(image, cal.image_to_bev, (width, height),
                                      flags=cv2.INTER_LINEAR)
            bev[~cal.bev_valid_mask] = 0
        draw_bev_overlay(bev, cfg)

        if cal is not None:
            corners = np.array([cal.vehicle_to_pixel(x, y)
                                for x, y in self.nominal_ground_points()])
            cv2.polylines(bev, [np.round(corners).astype(np.int32)], True,
                          (0, 255, 0), 1, cv2.LINE_AA)
            for i, (u, v) in enumerate(corners):
                put_label(bev, POINT_LABELS[i], (int(u) + 4, int(v) - 4),
                          POINT_COLORS[i], 0.5)
            for i, (u, v) in enumerate(self.check_points):
                x, y = cal.image_to_vehicle(u, v)
                pu, pv = cal.vehicle_to_pixel(x, y)
                p = (int(round(pu)), int(round(pv)))
                cv2.drawMarker(bev, p, CHECK_COLOR, cv2.MARKER_TILTED_CROSS,
                               14, 2)
                put_label(bev, f'{x * 100:+.1f},{y * 100:.1f}cm',
                          (p[0] + 8, p[1] + 16), CHECK_COLOR, 0.5)
            if self.bev_probe is not None:
                x, y = cal.pixel_to_vehicle(*self.bev_probe)
                cv2.circle(bev, self.bev_probe, 5, (255, 255, 0), 1)
                put_label(bev, f'x={x * 100:+.1f} y={y * 100:.1f} cm',
                          (self.bev_probe[0] + 8, self.bev_probe[1] - 8),
                          (255, 255, 0), 0.5)
        for x, y in self.options.check_point:
            u, v = cfg.vehicle_to_pixel(x, y)
            p = (int(round(u)), int(round(v)))
            cv2.drawMarker(bev, p, MARKER_COLOR, cv2.MARKER_CROSS, 16, 1)
            put_label(bev, f'ref {x * 100:+.0f},{y * 100:.0f}',
                      (p[0] + 8, p[1] - 8), MARKER_COLOR, 0.45)
        put_label(bev, f'{cfg.pixels_per_meter / 100:.1f} px/cm  '
                       f'{width}x{height}px  grid 10 cm', (10, 20),
                  (255, 255, 255), 0.5)
        return bev

    # ----- main loop ----------------------------------------------------
    def handle_key(self, key):
        if key in (ord('q'), 27):
            return False
        if key == ord('r'):
            self.reset()
        elif key == ord('u'):
            if self.check_points:
                self.check_points.pop()
            elif self.points:
                self.points.pop()
                self.recompute()
        elif key == ord('c'):
            self.check_points = []
            self.bev_probe = None
        elif key == ord('s'):
            self.save()
        elif key == ord('l'):
            self.load()
        return True

    def run(self):
        self.create_windows()
        waiting_reported = False
        stale_reported = False
        started = time.monotonic()
        while True:
            frame, stamp = self.frames.get()
            now = time.monotonic()
            if frame is None:
                if not waiting_reported and now - started > 2.0:
                    print(f'[대기] {self.options.topic} 영상을 기다리는 중입니다. '
                          'camera_node가 실행 중인지 확인하세요')
                    waiting_reported = True
                blank = np.zeros((240, 640, 3), dtype=np.uint8)
                put_label(blank, f'Waiting for {self.options.topic} ...',
                          (20, 120), (0, 255, 255), 0.7)
                cv2.imshow(CAMERA_WINDOW, blank)
                if not self.handle_key(cv2.waitKey(30) & 0xFF):
                    break
                continue

            size = (frame.shape[1], frame.shape[0])
            if self.image_size != size:
                if self.image_size is not None:
                    print(f'[경고] 카메라 해상도 변경 {self.image_size} → {size}. '
                          '기준점을 초기화합니다')
                    self.reset()
                self.image_size = size
                print(f'카메라 해상도: {size[0]}x{size[1]}')
                if self.pending_load:
                    self.pending_load = False
                    self.load(quiet_missing=True)
                else:
                    self.recompute()

            if now - stamp > IMAGE_TIMEOUT_SEC:
                if not stale_reported:
                    print(f'[경고] {IMAGE_TIMEOUT_SEC:.0f}초 이상 새 영상이 '
                          '없습니다')
                    stale_reported = True
            else:
                stale_reported = False

            image = self.undistorter.apply(frame)
            self.poll_trackbars()
            camera_view = self.draw_camera(image)
            if now - stamp > IMAGE_TIMEOUT_SEC:
                put_label(camera_view, 'NO NEW FRAMES', (8, image.shape[0] - 12),
                          (0, 0, 255), 0.6)
            cv2.imshow(CAMERA_WINDOW, camera_view)
            cv2.imshow(BEV_WINDOW, self.draw_bev(image))
            if not self.handle_key(cv2.waitKey(15) & 0xFF):
                break


def put_label(image, text, origin, color, scale=0.55):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_bev_overlay(bev, cfg):
    """10 cm grid, distance labels, centre line, origin, and axes."""
    width, height = cfg.size
    step = 0.10
    for x in np.arange(math.ceil(cfg.x_min / step) * step, cfg.x_max + 1e-9,
                       step):
        u = int(round(cfg.vehicle_to_pixel(x, 0)[0]))
        major = abs(round(x * 100) % 50) == 0
        cv2.line(bev, (u, 0), (u, height - 1),
                 (110, 110, 110) if major else (60, 60, 60), 1)
        if abs(x) > 1e-6:
            put_label(bev, f'{x * 100:+.0f}', (u + 2, height - 8),
                      (200, 200, 200), 0.4)
    for y in np.arange(math.ceil(cfg.y_min / step) * step, cfg.y_max + 1e-9,
                       step):
        v = int(round(cfg.vehicle_to_pixel(0, y)[1]))
        major = abs(round(y * 100) % 50) == 0
        cv2.line(bev, (0, v), (width - 1, v),
                 (110, 110, 110) if major else (60, 60, 60), 1)
        put_label(bev, f'{y * 100:.0f}cm', (4, v - 3), (200, 200, 200), 0.4)

    # Vehicle reference lines.
    for y, label, color in ((WHEELBASE_M, 'front axle', (255, 200, 0)),
                            (CAMERA_FORWARD_M, 'camera', (255, 0, 255))):
        v = int(round(cfg.vehicle_to_pixel(0, y)[1]))
        if 0 <= v < height:
            u0 = int(round(cfg.vehicle_to_pixel(-0.08, y)[0]))
            u1 = int(round(cfg.vehicle_to_pixel(0.08, y)[0]))
            cv2.line(bev, (u0, v), (u1, v), color, 2)
            put_label(bev, label, (u1 + 4, v + 4), color, 0.4)

    # Vehicle centre line x = 0.
    cu = int(round(cfg.vehicle_to_pixel(0, 0)[0]))
    cv2.line(bev, (cu, 0), (cu, height - 1), (0, 0, 255), 1)

    # Origin (rear-axle centre) and axes.
    ou, ov = (int(round(c)) for c in cfg.origin_pixel)
    axis = int(round(0.10 * cfg.pixels_per_meter))
    cv2.arrowedLine(bev, (ou, ov), (ou + axis, ov), (0, 0, 255), 2,
                    tipLength=0.15)
    cv2.arrowedLine(bev, (ou, ov), (ou, ov - axis), (0, 255, 0), 2,
                    tipLength=0.15)
    cv2.circle(bev, (ou, ov), 5, (255, 255, 255), -1)
    put_label(bev, '+X right', (ou + axis + 4, ov + 5), (0, 0, 255), 0.45)
    put_label(bev, '+Y fwd', (ou + 6, ov - axis - 4), (0, 255, 0), 0.45)
    put_label(bev, 'O rear axle', (ou + 8, ov - 8), (255, 255, 255), 0.45)


def main(argv=None):
    options, ros_args = parse_args(argv)
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
    except ModuleNotFoundError as exc:
        print(f'ROS 2 모듈을 불러올 수 없습니다: {exc}')
        print('source /opt/ros/jazzy/setup.bash 후 다시 실행하세요.')
        return 2

    frames = FrameBuffer()

    class CameraSubscriber(Node):
        """Only subscribes to the camera; no actuator outputs exist here."""

        def __init__(self):
            super().__init__('bev_calibrate')
            self.bridge = CvBridge()
            self.last_error_at = 0.0
            self.create_subscription(Image, options.topic, self.on_image,
                                     qos_profile_sensor_data)
            self.get_logger().info(f'{options.topic} 구독 시작')

        def on_image(self, message):
            try:
                frames.put(self.bridge.imgmsg_to_cv2(
                    message, desired_encoding='bgr8'))
            except Exception as exc:  # cv_bridge raises several types
                now = time.monotonic()
                if now - self.last_error_at > 1.0:
                    self.get_logger().error(f'영상 변환 실패: {exc}')
                    self.last_error_at = now

    try:
        app = BevCalibrationApp(options, frames)
    except (CalibrationError, OSError, cv2.error) as exc:
        print(f'초기화 오류: {exc}')
        return 1

    rclpy.init(args=ros_args)
    node = CameraSubscriber()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    print('BEV 캘리브레이션 시작. 클릭 순서: ' + ', '.join(POINT_DESCRIPTIONS))
    try:
        app.run()
    except KeyboardInterrupt:
        print('\n종료 요청')
    except Exception as exc:
        print(f'실행 오류: {type(exc).__name__}: {exc}')
        return 1
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
