#!/usr/bin/env python3
"""Pure LiDAR sector analysis and obstacle-avoidance state machine."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AvoidanceConfig:
    avoid_distance_m: float = 0.70
    clear_distance_m: float = 0.90
    emergency_distance_m: float = 0.25
    minimum_side_clearance_m: float = 0.45


@dataclass(frozen=True)
class ScanSectors:
    front_m: float | None
    left_m: float | None
    right_m: float | None


@dataclass(frozen=True)
class DriveCommand:
    mode: str
    steering: str
    moving: bool
    reason: str


def sector_minimum(scan, center_deg, half_width_deg):
    """Return a sector's nearest valid range, infinity if clear, or None."""
    if (
        not scan.ranges
        or not math.isfinite(scan.angle_min)
        or not math.isfinite(scan.angle_increment)
        or scan.angle_increment == 0.0
        or not math.isfinite(scan.range_min)
        or not math.isfinite(scan.range_max)
        or scan.range_min < 0.0
        or scan.range_max <= scan.range_min
    ):
        return None

    center = math.radians(center_deg)
    half_width = math.radians(half_width_deg)
    nearest = math.inf
    has_measurement = False

    for index, distance in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        offset = math.atan2(
            math.sin(angle - center),
            math.cos(angle - center),
        )
        if abs(offset) > half_width:
            continue

        if math.isinf(distance) and distance > 0.0:
            has_measurement = True
        elif math.isfinite(distance) and scan.range_min <= distance <= scan.range_max:
            has_measurement = True
            nearest = min(nearest, distance)

    return nearest if has_measurement else None


def analyze_scan(
    scan,
    front_center_deg=180.0,
    front_half_width_deg=25.0,
    side_center_offset_deg=50.0,
    side_half_width_deg=25.0,
):
    """Measure the front and the two possible avoidance corridors."""
    return ScanSectors(
        front_m=sector_minimum(
            scan,
            front_center_deg,
            front_half_width_deg,
        ),
        left_m=sector_minimum(
            scan,
            front_center_deg + side_center_offset_deg,
            side_half_width_deg,
        ),
        right_m=sector_minimum(
            scan,
            front_center_deg - side_center_offset_deg,
            side_half_width_deg,
        ),
    )


class AvoidanceController:
    """Convert sector distances into stable, fail-safe drive commands."""

    def __init__(self, config=None):
        self.config = config or AvoidanceConfig()
        self.mode = 'waiting'

    @staticmethod
    def _is_open(distance, minimum):
        return distance is not None and distance >= minimum

    def _command(self, mode, steering, moving, reason):
        self.mode = mode
        return DriveCommand(mode, steering, moving, reason)

    def update(self, sectors):
        cfg = self.config
        if (
            sectors.front_m is None
            or sectors.left_m is None
            or sectors.right_m is None
        ):
            return self._command(
                'sensor_fault',
                'center',
                False,
                '회피 구간에 유효한 LiDAR 거리값이 없습니다',
            )

        if sectors.front_m <= cfg.emergency_distance_m:
            return self._command(
                'emergency_stop',
                'center',
                False,
                f'전방 장애물이 {sectors.front_m:.2f} m까지 접근했습니다',
            )

        avoiding = self.mode in ('avoiding_left', 'avoiding_right')
        needs_avoidance = sectors.front_m <= cfg.avoid_distance_m
        if avoiding and sectors.front_m < cfg.clear_distance_m:
            needs_avoidance = True

        if not needs_avoidance:
            return self._command(
                'straight',
                'center',
                True,
                '전방 통로가 확보되었습니다',
            )

        left_open = self._is_open(
            sectors.left_m,
            cfg.minimum_side_clearance_m,
        )
        right_open = self._is_open(
            sectors.right_m,
            cfg.minimum_side_clearance_m,
        )

        # Keep the chosen direction to prevent left/right oscillation. If that
        # corridor closes, stop for one update before considering a new turn.
        if self.mode == 'avoiding_left':
            if left_open:
                return self._command(
                    'avoiding_left',
                    'left',
                    True,
                    f'좌측 회피 유지 (여유 {sectors.left_m:.2f} m)',
                )
            return self._command(
                'blocked',
                'center',
                False,
                '선택한 좌측 회피 통로가 막혔습니다',
            )

        if self.mode == 'avoiding_right':
            if right_open:
                return self._command(
                    'avoiding_right',
                    'right',
                    True,
                    f'우측 회피 유지 (여유 {sectors.right_m:.2f} m)',
                )
            return self._command(
                'blocked',
                'center',
                False,
                '선택한 우측 회피 통로가 막혔습니다',
            )

        if not left_open and not right_open:
            return self._command(
                'blocked',
                'center',
                False,
                '좌우 회피 통로가 모두 막혔습니다',
            )

        if left_open and (not right_open or sectors.left_m >= sectors.right_m):
            return self._command(
                'avoiding_left',
                'left',
                True,
                f'좌측 통로 선택 (좌 {sectors.left_m:.2f} m, '
                f'우 {sectors.right_m:.2f} m)',
            )

        return self._command(
            'avoiding_right',
            'right',
            True,
            f'우측 통로 선택 (좌 {sectors.left_m:.2f} m, '
            f'우 {sectors.right_m:.2f} m)',
        )
