#!/usr/bin/env python3

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float):
    half = 0.5 * yaw
    return (
        0.0,
        0.0,
        math.sin(half),
        math.cos(half),
    )


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class ScanPoint:
    x: float
    y: float


@dataclass
class LineFit:
    valid: bool = False
    a: float = 0.0
    b: float = 0.0
    inliers: int = 0
    visible_length: float = 0.0


@dataclass
class RowDetection:
    left_valid: bool = False
    right_valid: bool = False

    left_a: float = 0.0
    left_b: float = 0.0
    right_a: float = 0.0
    right_b: float = 0.0

    center_a: float = 0.0
    center_b: float = 0.0

    lane_width: float = 0.0
    confidence: float = 0.0
    end_probability: float = 0.0

    points_left: List[ScanPoint] = field(default_factory=list)
    points_right: List[ScanPoint] = field(default_factory=list)
    points_all: List[ScanPoint] = field(default_factory=list)


@dataclass
class RowModel:
    valid: bool = False
    confidence: float = 0.0

    center_a: float = 0.0
    center_b: float = 0.0

    row_yaw_base: float = 0.0
    row_width: float = 0.75

    end_probability: float = 0.0
    end_detected: bool = False

    missing_frames: int = 0
    last_detection: Optional[RowDetection] = None


@dataclass
class PathPoint:
    x: float
    y: float
    yaw: float
    v: float


@dataclass
class LocalPath:
    points: List[PathPoint]
    valid: bool
    frame_id: str = "base_link"
    reason: str = ""


@dataclass(frozen=True)
class PatternStep:
    row_shift_count: int
    row_shift_direction: str


@dataclass
class MapRowBand:
    valid: bool = False
    lateral_v: float = 0.0
    u_min: float = 0.0
    u_max: float = 0.0
    points: int = 0


@dataclass
class MapLane:
    valid: bool = False
    center_v: float = 0.0
    left_row_v: float = 0.0
    right_row_v: float = 0.0
    width: float = 0.0
    confidence: float = 0.0
    source: str = ""


def parse_pattern(pattern: str) -> List[PatternStep]:
    """Parse patterns like "1L 2R 3 L" or "1L-1R-2L"."""
    normalized = pattern.replace("-", " ").replace(",", " ").upper()
    raw_tokens = [token for token in normalized.split() if token]

    steps: List[PatternStep] = []
    pending_count: Optional[int] = None

    for token in raw_tokens:
        if token.isdigit():
            pending_count = int(token)
            continue

        if token in ("L", "R") and pending_count is not None:
            steps.append(PatternStep(max(1, pending_count), token))
            pending_count = None
            continue

        if len(token) >= 2 and token[:-1].isdigit() and token[-1] in ("L", "R"):
            steps.append(PatternStep(max(1, int(token[:-1])), token[-1]))
            pending_count = None
            continue

        raise ValueError(
            f"Invalid pattern token '{token}'. Use e.g. '1L 2R 3L' or '1 L 2 R'."
        )

    if pending_count is not None:
        raise ValueError("Pattern ends with a number but no direction L/R.")

    return steps


class MissionState(Enum):
    IDLE = 0
    FOLLOW_ROW = 1
    EXIT_ROW = 2
    PLAN_TURN = 3
    EXECUTE_TURN = 4
    ACQUIRE_ROW = 5
    ENTER_ROW = 6
    FINISHED = 7


@dataclass
class NavigatorParams:
    scan_topic: str = "/sensors/merged_scan"
    cmd_vel_topic: str = "/cmd_vel"
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    map_topic: str = "/map"
    map_frame: str = "map"
    use_slam_map: bool = True
    require_map_for_turns: bool = True

    control_frequency: float = 30.0

    expected_row_width: float = 0.75
    min_lane_width: float = 0.55
    max_lane_width: float = 1.20

    roi_x_min: float = 0.25
    roi_x_max: float = 2.0
    roi_y_abs_min: float = 0.18
    roi_y_abs_max: float = 0.90

    acquire_roi_x_min: float = -0.30
    acquire_roi_x_max: float = 2.80
    acquire_roi_y_abs_min: float = 0.05
    acquire_roi_y_abs_max: float = 1.20

    ransac_iterations: int = 80
    ransac_distance: float = 0.08
    min_inliers: int = 5
    min_visible_length: float = 0.35
    max_abs_line_slope: float = 0.9

    centerline_max_abs_slope: float = 0.7

    tracker_alpha: float = 0.18
    confidence_decay: float = 0.95

    front_density_x_min: float = 0.60
    front_density_x_max: float = 2.00
    front_density_y_abs: float = 0.45
    front_density_threshold: int = 1
    end_probability_threshold: float = 0.88
    end_stable_frames_required: int = 35

    min_follow_confidence: float = 0.10
    min_enter_confidence: float = 0.12
    enter_stable_frames_required: int = 3
    acquire_timeout_sec: float = 8.0

    follow_speed: float = 0.35
    slow_speed: float = 0.12
    enter_speed: float = 0.22
    turn_speed: float = 0.28
    max_linear_speed: float = 0.45
    max_angular_speed: float = 1.20
    follow_max_angular_speed: float = 0.90
    turn_max_angular_speed: float = 1.00
    angular_rate_limit: float = 1.8

    lookahead_distance: float = 0.75
    path_goal_xy_tolerance: float = 0.20
    path_goal_yaw_tolerance: float = 0.40

    exit_distance: float = 0.70
    turn_forward_distance: float = 2.20
    min_turn_radius: float = 0.38
    enter_distance: float = 0.90
    pattern: str = "1L 2R 2L 3R"
    row_shift_count: int = 1
    row_shift_direction: str = "L"
    turn_180: bool = True

    map_row_detection_enabled: bool = True
    map_row_occupancy_threshold: int = 50
    map_row_search_x_forward: float = 4.0
    map_row_search_x_backward: float = 1.0
    map_row_search_y_side: float = 4.0
    map_row_lateral_bin: float = 0.10
    map_row_min_band_points: int = 12
    map_row_min_band_length: float = 1.2
    map_row_max_extrapolated_lanes: int = 3

    turn_replan_enabled: bool = True
    turn_replan_period_frames: int = 5
    turn_replan_max_attempts: int = 60

    enable_safety: bool = False

    obstacle_stop_distance: float = 0.25
    obstacle_slow_distance: float = 0.45

    publish_debug: bool = True


class MapRowDetector:
    def __init__(self, params: NavigatorParams):
        self.p = params

    def detect_lanes(self, grid: Optional[OccupancyGrid], pose: Pose2D) -> Tuple[List[MapLane], List[MapRowBand], str]:
        if grid is None:
            return [], [], "no OccupancyGrid"

        if not self.p.map_row_detection_enabled:
            return [], [], "map row detection disabled"

        width = int(grid.info.width)
        height = int(grid.info.height)
        resolution = float(grid.info.resolution)

        if width <= 0 or height <= 0 or resolution <= 0.0:
            return [], [], "invalid OccupancyGrid metadata"

        data = np.asarray(grid.data, dtype=np.int16).reshape((height, width))
        occ_r, occ_c = np.where(data >= int(self.p.map_row_occupancy_threshold))

        if len(occ_r) == 0:
            return [], [], "no occupied map cells"

        origin = grid.info.origin
        origin_yaw = yaw_from_quaternion(origin.orientation)
        co = math.cos(origin_yaw)
        so = math.sin(origin_yaw)

        # Begrenze die Punktzahl deterministisch, damit die Erkennung auch bei grossen Maps
        # in Echtzeit bleibt. Die Auswahl ist gleichmaessig ueber die belegten Zellen verteilt.
        max_cells = 25000
        if len(occ_r) > max_cells:
            idx = np.linspace(0, len(occ_r) - 1, max_cells).astype(int)
            occ_r = occ_r[idx]
            occ_c = occ_c[idx]

        gx = (occ_c.astype(float) + 0.5) * resolution
        gy = (occ_r.astype(float) + 0.5) * resolution

        mx = origin.position.x + co * gx - so * gy
        my = origin.position.y + so * gx + co * gy

        dx = mx - pose.x
        dy = my - pose.y

        cp = math.cos(pose.yaw)
        sp = math.sin(pose.yaw)

        # Lokales Koordinatensystem am Roboter:
        # u = vorwaerts entlang aktueller Reihenrichtung
        # v = links quer zur Reihenrichtung
        u = cp * dx + sp * dy
        v = -sp * dx + cp * dy

        mask = (
            (u >= -self.p.map_row_search_x_backward)
            & (u <= self.p.map_row_search_x_forward)
            & (np.abs(v) <= self.p.map_row_search_y_side)
        )

        u = u[mask]
        v = v[mask]

        if len(u) < self.p.map_row_min_band_points:
            return [], [], f"not enough occupied cells in local map window: {len(u)}"

        order = np.argsort(v)
        u_sorted = u[order]
        v_sorted = v[order]

        raw_bands: List[MapRowBand] = []
        start = 0
        merge_gap = max(self.p.map_row_lateral_bin, 0.03)

        for i in range(1, len(v_sorted)):
            if abs(float(v_sorted[i] - v_sorted[i - 1])) > merge_gap:
                raw_bands.append(self._make_band(u_sorted[start:i], v_sorted[start:i]))
                start = i

        raw_bands.append(self._make_band(u_sorted[start:], v_sorted[start:]))

        bands = [
            b for b in raw_bands
            if b.valid
            and b.points >= self.p.map_row_min_band_points
            and (b.u_max - b.u_min) >= self.p.map_row_min_band_length
        ]
        bands.sort(key=lambda b: b.lateral_v)

        if len(bands) < 2:
            return [], bands, f"not enough row bands detected: {len(bands)}"

        lanes: List[MapLane] = []

        for right, left in zip(bands[:-1], bands[1:]):
            width_v = left.lateral_v - right.lateral_v
            if self.p.min_lane_width <= width_v <= self.p.max_lane_width:
                overlap = max(0.0, min(right.u_max, left.u_max) - max(right.u_min, left.u_min))
                length_score = clamp(overlap / max(self.p.map_row_min_band_length, 1e-3), 0.0, 1.0)
                width_error = abs(width_v - self.p.expected_row_width)
                width_score = clamp(1.0 - width_error / max(self.p.expected_row_width, 1e-3), 0.0, 1.0)
                confidence = clamp(0.55 * length_score + 0.45 * width_score, 0.0, 1.0)
                lanes.append(
                    MapLane(
                        valid=True,
                        center_v=0.5 * (right.lateral_v + left.lateral_v),
                        left_row_v=left.lateral_v,
                        right_row_v=right.lateral_v,
                        width=width_v,
                        confidence=confidence,
                        source="detected",
                    )
                )

        if len(lanes) == 0:
            return [], bands, "row bands detected, but no valid lane gap"

        lanes.sort(key=lambda lane: lane.center_v)
        return lanes, bands, "ok"

    def _make_band(self, u_values: np.ndarray, v_values: np.ndarray) -> MapRowBand:
        if len(u_values) == 0:
            return MapRowBand(valid=False)

        return MapRowBand(
            valid=True,
            lateral_v=float(np.median(v_values)),
            u_min=float(np.min(u_values)),
            u_max=float(np.max(u_values)),
            points=int(len(u_values)),
        )

    def select_target_lane(
        self,
        lanes: List[MapLane],
        direction: str,
        count: int,
    ) -> Optional[MapLane]:
        if len(lanes) == 0:
            return None

        lanes_sorted = sorted(lanes, key=lambda lane: lane.center_v)
        current_idx = min(range(len(lanes_sorted)), key=lambda i: abs(lanes_sorted[i].center_v))

        step = max(1, int(count))
        sign = 1 if direction.upper() == "L" else -1
        target_idx = current_idx + sign * step

        if 0 <= target_idx < len(lanes_sorted):
            lane = lanes_sorted[target_idx]
            lane.source = "detected"
            return lane

        # Wenn die Zielgasse im aktuellen Kartenfenster nicht komplett sichtbar ist,
        # extrapolieren wir aus den in der SLAM-Map erkannten Gassenabstaenden.
        if len(lanes_sorted) >= 2:
            spacings = [
                lanes_sorted[i + 1].center_v - lanes_sorted[i].center_v
                for i in range(len(lanes_sorted) - 1)
            ]
            spacing = float(np.median(spacings))
        else:
            spacing = self.p.expected_row_width

        requested_shift = sign * step * spacing
        center_v = lanes_sorted[current_idx].center_v + requested_shift

        if abs(target_idx - current_idx) > self.p.map_row_max_extrapolated_lanes:
            return None

        return MapLane(
            valid=True,
            center_v=float(center_v),
            left_row_v=float(center_v + 0.5 * spacing),
            right_row_v=float(center_v - 0.5 * spacing),
            width=float(spacing),
            confidence=0.45,
            source="extrapolated_from_map_lanes",
        )


class RowPerception:
    def __init__(self, params: NavigatorParams):
        self.p = params
        self.mode = MissionState.FOLLOW_ROW

    def set_mode(self, mode: MissionState) -> None:
        self.mode = mode

    def process_scan(self, scan: LaserScan) -> RowDetection:
        points = self.scan_to_points(scan)
        points = self.filter_roi(points)

        det = RowDetection()
        det.points_all = points
        det.points_left = self.select_nearest_side_points(points, "left")
        det.points_right = self.select_nearest_side_points(points, "right")

        left_line = self.fit_line_ransac(det.points_left)
        right_line = self.fit_line_ransac(det.points_right)

        if left_line.valid:
            det.left_valid = True
            det.left_a = left_line.a
            det.left_b = left_line.b

        if right_line.valid:
            det.right_valid = True
            det.right_a = right_line.a
            det.right_b = right_line.b

        self.compute_centerline(det)
        self.compute_confidence(det, left_line, right_line)
        self.compute_end_probability(det)

        return det

    def current_roi(self) -> Tuple[float, float, float, float]:
        if self.mode in (MissionState.ACQUIRE_ROW, MissionState.ENTER_ROW):
            return (
                self.p.acquire_roi_x_min,
                self.p.acquire_roi_x_max,
                self.p.acquire_roi_y_abs_min,
                self.p.acquire_roi_y_abs_max,
            )

        return (
            self.p.roi_x_min,
            self.p.roi_x_max,
            self.p.roi_y_abs_min,
            self.p.roi_y_abs_max,
        )

    def current_y_abs_min(self) -> float:
        return self.current_roi()[2]

    def scan_to_points(self, scan: LaserScan) -> List[ScanPoint]:
        points: List[ScanPoint] = []
        angle = scan.angle_min

        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append(ScanPoint(x, y))

            angle += scan.angle_increment

        return points

    def filter_roi(self, points: List[ScanPoint]) -> List[ScanPoint]:
        x_min, x_max, y_abs_min, y_abs_max = self.current_roi()

        out = []
        for pt in points:
            if x_min <= pt.x <= x_max and y_abs_min <= abs(pt.y) <= y_abs_max:
                out.append(pt)

        return out

    def select_nearest_side_points(
        self,
        points: List[ScanPoint],
        side: str,
    ) -> List[ScanPoint]:
        if side == "left":
            side_points = [p for p in points if p.y > self.current_y_abs_min()]
            side_points.sort(key=lambda p: p.y)
        else:
            side_points = [p for p in points if p.y < -self.current_y_abs_min()]
            side_points.sort(key=lambda p: abs(p.y))

        if len(side_points) < self.p.min_inliers:
            return side_points

        nearest_y_abs = abs(side_points[0].y)
        band_width = 0.35

        selected = [
            p for p in side_points
            if abs(abs(p.y) - nearest_y_abs) <= band_width
        ]

        return selected

    def fit_line_ransac(self, points: List[ScanPoint]) -> LineFit:
        if len(points) < self.p.min_inliers:
            return LineFit(valid=False)

        best_a = 0.0
        best_b = 0.0
        best_inliers: List[ScanPoint] = []

        rng = random.Random(42)

        for _ in range(self.p.ransac_iterations):
            p1, p2 = rng.sample(points, 2)
            dx = p2.x - p1.x

            if abs(dx) < 1e-3:
                continue

            a = (p2.y - p1.y) / dx

            if abs(a) > self.p.max_abs_line_slope:
                continue

            b = p1.y - a * p1.x

            inliers = []
            denom = math.sqrt(a * a + 1.0)

            for p in points:
                dist = abs(a * p.x - p.y + b) / denom
                if dist < self.p.ransac_distance:
                    inliers.append(p)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_a = a
                best_b = b

        if len(best_inliers) < self.p.min_inliers:
            return LineFit(valid=False)

        xs = np.array([p.x for p in best_inliers], dtype=float)
        ys = np.array([p.y for p in best_inliers], dtype=float)

        try:
            a, b = np.polyfit(xs, ys, 1)
        except Exception:
            a, b = best_a, best_b

        visible_length = float(max(xs) - min(xs)) if len(xs) else 0.0
        valid = visible_length >= self.p.min_visible_length

        return LineFit(
            valid=valid,
            a=float(a),
            b=float(b),
            inliers=len(best_inliers),
            visible_length=visible_length,
        )

    def compute_centerline(self, det: RowDetection) -> None:
        if det.left_valid and det.right_valid:
            x_ref = 1.0

            left_y = det.left_a * x_ref + det.left_b
            right_y = det.right_a * x_ref + det.right_b

            det.lane_width = left_y - right_y

            if det.lane_width < self.p.min_lane_width or det.lane_width > self.p.max_lane_width:
                det.left_valid = False
                det.right_valid = False
                det.center_a = 0.0
                det.center_b = 0.0
                det.lane_width = 0.0
                return

            det.center_a = 0.5 * (det.left_a + det.right_a)
            det.center_b = 0.5 * (det.left_b + det.right_b)
            return

        if det.left_valid:
            det.center_a = det.left_a
            det.center_b = det.left_b - self.p.expected_row_width / 2.0
            det.lane_width = self.p.expected_row_width
            return

        if det.right_valid:
            det.center_a = det.right_a
            det.center_b = det.right_b + self.p.expected_row_width / 2.0
            det.lane_width = self.p.expected_row_width
            return

        det.center_a = 0.0
        det.center_b = 0.0
        det.lane_width = 0.0

    def compute_confidence(self, det: RowDetection, left: LineFit, right: LineFit) -> None:
        score = 0.0

        if self.mode in (MissionState.ACQUIRE_ROW, MissionState.ENTER_ROW):
            if det.left_valid or det.right_valid:
                score += 0.40

            if det.left_valid and det.right_valid:
                score += 0.20

            if left.visible_length >= self.p.min_visible_length or right.visible_length >= self.p.min_visible_length:
                score += 0.20

            if self.p.min_lane_width <= det.lane_width <= self.p.max_lane_width:
                score += 0.10

            if abs(det.center_a) < 1.2:
                score += 0.10

            det.confidence = clamp(score, 0.0, 1.0)
            return

        if det.left_valid:
            score += 0.25

        if det.right_valid:
            score += 0.25

        if det.left_valid and det.right_valid:
            if self.p.min_lane_width <= det.lane_width <= self.p.max_lane_width:
                score += 0.25

            if abs(det.left_a - det.right_a) < 0.35:
                score += 0.15
        else:
            score += 0.10

        if abs(det.center_a) < 1.2:
            score += 0.10

        det.confidence = clamp(score, 0.0, 1.0)

    def compute_end_probability(self, det: RowDetection) -> None:
        # Konservative Reihenende-Erkennung.
        #
        # Ziel:
        # - Keine Reihenenden bei kurzen Luecken, duennen Pflanzen oder
        #   kurzzeitig schlechter Linienerkennung.
        # - Ein Reihenende nur dann erkennen, wenn ueber mehrere Frames
        #   beide Seitenstrukturen fehlen und die Detektionskonfidenz niedrig ist.

        forward_side_points = [
            p for p in det.points_all
            if self.p.front_density_x_min <= p.x <= self.p.front_density_x_max
            and self.current_y_abs_min() <= abs(p.y) <= self.current_roi()[3]
        ]

        side_density = len(forward_side_points)

        both_missing = not det.left_valid and not det.right_valid
        one_missing = det.left_valid != det.right_valid

        low_confidence = det.confidence < 0.20
        very_low_confidence = det.confidence < 0.08

        side_window_empty = side_density < self.p.front_density_threshold

        score = 0.0

        # Sicheres Reihenende:
        # Beide Seiten fehlen, im Vorwaerts-Seitenfenster sind keine Punkte,
        # und die Konfidenz ist sehr niedrig.
        if both_missing and side_window_empty and very_low_confidence:
            score = 1.0

        # Wahrscheinliches Reihenende:
        # Beide Seiten fehlen und das Seitenfenster ist leer.
        elif both_missing and side_window_empty and low_confidence:
            score = 0.85

        # Moegliches Reihenende:
        # Beide Seiten fehlen, aber es gibt noch einzelne Seitenpunkte.
        # Das kann auch eine Luecke in der Reihe sein, deshalb nur schwach werten.
        elif both_missing and low_confidence:
            score = 0.35

        # Eine einzelne fehlende Seite ist kein Reihenende.
        # Das passiert haeufig bei schiefen Pflanzen, Luecken oder asymmetrischer Sicht.
        elif one_missing and side_window_empty and very_low_confidence:
            score = 0.15

        else:
            score = 0.0

        det.end_probability = clamp(score, 0.0, 1.0)


class RowTracker:
    def __init__(self, params: NavigatorParams):
        self.p = params
        self.model = RowModel(row_width=params.expected_row_width)

    def update(self, det: RowDetection) -> RowModel:
        if det.confidence > 0.15 and (det.left_valid or det.right_valid):
            if not self.model.valid:
                self.model.center_a = det.center_a
                self.model.center_b = det.center_b
                self.model.valid = True
            else:
                a = self.p.tracker_alpha
                self.model.center_a = (1.0 - a) * self.model.center_a + a * det.center_a
                self.model.center_b = (1.0 - a) * self.model.center_b + a * det.center_b

            self.model.confidence = clamp(
                0.80 * self.model.confidence + 0.20 * det.confidence,
                0.0,
                1.0,
            )

            self.model.row_yaw_base = math.atan(self.model.center_a)

            if det.lane_width > 0.1:
                self.model.row_width = (
                    0.90 * self.model.row_width + 0.10 * det.lane_width
                )

            self.model.missing_frames = 0
            self.model.last_detection = det
        else:
            self.model.missing_frames += 1
            self.model.confidence *= self.p.confidence_decay

            if self.model.confidence < 0.01 and self.model.missing_frames > 20:
                self.model.valid = False

        # Reihenende asymmetrisch filtern:
        # - langsamer Anstieg, damit kurze Luecken nicht sofort ein Ende ausloesen
        # - schneller Abfall, damit falsche Ende-Hypothesen rasch verschwinden
        if det.end_probability > self.model.end_probability:
            self.model.end_probability = (
                0.90 * self.model.end_probability + 0.10 * det.end_probability
            )
        else:
            self.model.end_probability = (
                0.55 * self.model.end_probability + 0.45 * det.end_probability
            )

        self.model.end_detected = (
            self.model.end_probability >= self.p.end_probability_threshold
        )

        return self.model


class LocalPlanner:
    def __init__(self, params: NavigatorParams):
        self.p = params

    def plan_follow_row(self, row: RowModel, speed: Optional[float] = None) -> LocalPath:
        if not row.valid:
            return LocalPath([], False, "base_link", "row model invalid")

        v = self.p.follow_speed if speed is None else speed

        if row.confidence < self.p.min_follow_confidence:
            v = min(v, self.p.slow_speed)

        points: List[PathPoint] = []

        center_a = clamp(
            row.center_a,
            -self.p.centerline_max_abs_slope,
            self.p.centerline_max_abs_slope,
        )

        # Groessere seitliche Fehler duerfen korrigiert werden.
        # Bei grosser Querabweichung wird die Geschwindigkeit reduziert, damit
        # der Roboter bei breiten/kurvigen Reihen nicht aggressiv ausschert.
        center_b = clamp(row.center_b, -0.45, 0.45)
        lateral_error = abs(center_b)

        if lateral_error > 0.35:
            v = min(v, self.p.slow_speed)
        elif lateral_error > 0.25:
            v *= 0.70

        path_x_max = max(1.2, min(3.0, self.p.roi_x_max))

        for x in np.linspace(0.25, path_x_max, 40):
            y = center_a * float(x) + center_b
            yaw = math.atan(center_a)
            points.append(PathPoint(float(x), float(y), float(yaw), float(v)))

        return LocalPath(points, True, "base_link")

    def plan_exit_row(self) -> LocalPath:
        points: List[PathPoint] = []

        if self.p.exit_distance <= 0.01:
            points.append(PathPoint(0.0, 0.0, 0.0, self.p.slow_speed))
            return LocalPath(points, True, "base_link")

        for x in np.linspace(0.05, self.p.exit_distance, 12):
            points.append(PathPoint(float(x), 0.0, 0.0, self.p.slow_speed))

        return LocalPath(points, True, "base_link")

    def plan_acquire_row(self, row: RowModel) -> LocalPath:
        if row.valid and row.confidence > 0.12:
            return self.plan_follow_row(row, speed=self.p.enter_speed)

        points: List[PathPoint] = []

        for x in np.linspace(0.20, 1.40, 18):
            xf = float(x)
            y = 0.18 * math.sin(2.6 * xf)
            yaw = 0.20 * math.sin(2.6 * xf)
            points.append(PathPoint(xf, y, yaw, self.p.enter_speed * 0.65))

        return LocalPath(points, True, "base_link")

    def plan_enter_row(self, row: RowModel) -> LocalPath:
        if row.valid:
            return self.plan_follow_row(row, speed=self.p.enter_speed)

        points: List[PathPoint] = []

        for x in np.linspace(0.2, self.p.enter_distance, 18):
            points.append(PathPoint(float(x), 0.0, 0.0, self.p.enter_speed))

        return LocalPath(points, True, "base_link")

    def plan_turn_path_odom(self, start: Pose2D) -> LocalPath:
        return self.plan_turn_path_global(start, self.p.odom_frame)

    def plan_turn_path_map(self, start: Pose2D) -> LocalPath:
        return self.plan_turn_path_global(start, self.p.map_frame)

    def plan_turn_path_global(self, start: Pose2D, frame_id: str) -> LocalPath:
        direction = 1.0 if self.p.row_shift_direction.upper() == "L" else -1.0

        row_shift = self.p.row_shift_count * self.p.expected_row_width
        radius = max(row_shift / 2.0, self.p.min_turn_radius)

        if radius <= 0.05:
            return LocalPath([], False, frame_id, "invalid turn radius")

        local: List[PathPoint] = []

        if self.p.exit_distance > 0.01:
            for x in np.linspace(0.0, self.p.exit_distance, 12):
                local.append(PathPoint(float(x), 0.0, 0.0, self.p.slow_speed))

        x_offset = self.p.exit_distance

        for phi in np.linspace(-math.pi / 2.0, math.pi / 2.0, 90):
            x = x_offset + radius * math.cos(phi)
            y = direction * (radius + radius * math.sin(phi))

            dx_dphi = -radius * math.sin(phi)
            dy_dphi = direction * radius * math.cos(phi)

            yaw = math.atan2(dy_dphi, dx_dphi)

            local.append(
                PathPoint(
                    float(x),
                    float(y),
                    float(yaw),
                    self.p.turn_speed,
                )
            )

        end_y = direction * (2.0 * radius)

        if self.p.enter_distance > 0.01:
            for s in np.linspace(0.0, self.p.enter_distance, 20):
                x = x_offset - float(s)
                y = end_y
                yaw = math.pi

                local.append(
                    PathPoint(
                        float(x),
                        float(y),
                        float(yaw),
                        self.p.enter_speed,
                    )
                )

        odom_points: List[PathPoint] = []

        c = math.cos(start.yaw)
        s = math.sin(start.yaw)

        for p in local:
            ox = start.x + c * p.x - s * p.y
            oy = start.y + s * p.x + c * p.y
            oyaw = wrap_to_pi(start.yaw + p.yaw)

            odom_points.append(
                PathPoint(
                    float(ox),
                    float(oy),
                    float(oyaw),
                    float(p.v),
                )
            )

        return LocalPath(odom_points, True, frame_id)


    def plan_turn_path_to_map_lane(self, start: Pose2D, target_lane: MapLane) -> LocalPath:
        if not target_lane.valid:
            return LocalPath([], False, self.p.map_frame, "invalid target map lane")

        row_shift = float(target_lane.center_v)

        if abs(row_shift) < 0.10:
            return LocalPath([], False, self.p.map_frame, "target map lane too close to current lane")

        direction = 1.0 if row_shift > 0.0 else -1.0
        radius = max(abs(row_shift) / 2.0, self.p.min_turn_radius)

        if radius <= 0.05:
            return LocalPath([], False, self.p.map_frame, "invalid map lane turn radius")

        local: List[PathPoint] = []

        if self.p.exit_distance > 0.01:
            for x in np.linspace(0.0, self.p.exit_distance, 12):
                local.append(PathPoint(float(x), 0.0, 0.0, self.p.slow_speed))

        x_offset = self.p.exit_distance

        for phi in np.linspace(-math.pi / 2.0, math.pi / 2.0, 90):
            x = x_offset + radius * math.cos(phi)
            y = direction * (radius + radius * math.sin(phi))

            dx_dphi = -radius * math.sin(phi)
            dy_dphi = direction * radius * math.cos(phi)
            yaw = math.atan2(dy_dphi, dx_dphi)

            local.append(PathPoint(float(x), float(y), float(yaw), self.p.turn_speed))

        end_y = row_shift

        if self.p.enter_distance > 0.01:
            for s in np.linspace(0.0, self.p.enter_distance, 20):
                x = x_offset - float(s)
                y = end_y
                yaw = math.pi
                local.append(PathPoint(float(x), float(y), float(yaw), self.p.enter_speed))

        map_points: List[PathPoint] = []
        c = math.cos(start.yaw)
        s = math.sin(start.yaw)

        for p in local:
            mx = start.x + c * p.x - s * p.y
            my = start.y + s * p.x + c * p.y
            myaw = wrap_to_pi(start.yaw + p.yaw)
            map_points.append(PathPoint(float(mx), float(my), float(myaw), float(p.v)))

        return LocalPath(map_points, True, self.p.map_frame)


class PathFollower:
    def __init__(self, params: NavigatorParams):
        self.p = params
        self.last_w_base = 0.0
        self.last_w_odom = 0.0

    def compute_cmd(self, path: LocalPath, pose_odom: Optional[Pose2D]) -> Twist:
        cmd = Twist()

        if not path.valid or len(path.points) == 0:
            self.last_w_base = 0.0
            self.last_w_odom = 0.0
            return cmd

        if path.frame_id == "base_link":
            return self.compute_cmd_base_link(path)

        if path.frame_id in (self.p.odom_frame, self.p.map_frame):
            if pose_odom is None:
                return cmd

            return self.compute_cmd_odom(path, pose_odom)

        return cmd

    def compute_cmd_base_link(self, path: LocalPath) -> Twist:
        cmd = Twist()
        target = self.find_lookahead_base_link(path.points)

        if target is None:
            return cmd

        alpha = math.atan2(target.y, target.x)
        curvature = 2.0 * math.sin(alpha) / max(self.p.lookahead_distance, 1e-3)

        v = clamp(target.v, 0.0, self.p.max_linear_speed)

        abs_curvature = abs(curvature)

        # Bei engen Kurven innerhalb der Reihe wird die Geschwindigkeit reduziert.
        # Dadurch bleibt die benoetigte Winkelgeschwindigkeit erreichbar und der
        # Roboter schneidet Kurven weniger stark.
        if abs_curvature > 1.6:
            v *= 0.40
        elif abs_curvature > 1.1:
            v *= 0.55
        elif abs_curvature > 0.7:
            v *= 0.75

        w = clamp(
            v * curvature,
            -self.p.follow_max_angular_speed,
            self.p.follow_max_angular_speed,
        )
        w = self.rate_limit(w, self.last_w_base)
        self.last_w_base = w

        cmd.linear.x = v
        cmd.angular.z = w

        return cmd

    def compute_cmd_odom(self, path: LocalPath, pose: Pose2D) -> Twist:
        cmd = Twist()
        target = self.find_lookahead_odom(path.points, pose)

        if target is None:
            return cmd

        goal = path.points[-1]
        goal_dist = math.hypot(goal.x - pose.x, goal.y - pose.y)
        goal_yaw_err = wrap_to_pi(goal.yaw - pose.yaw)

        if goal_dist < self.p.path_goal_xy_tolerance:
            if abs(goal_yaw_err) > self.p.path_goal_yaw_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = clamp(
                    1.5 * goal_yaw_err,
                    -self.p.turn_max_angular_speed,
                    self.p.turn_max_angular_speed,
                )
                return cmd

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        dx = target.x - pose.x
        dy = target.y - pose.y

        c = math.cos(-pose.yaw)
        s = math.sin(-pose.yaw)

        bx = c * dx - s * dy
        by = s * dx + c * dy

        alpha = math.atan2(by, bx)

        if bx < -0.05:
            cmd.linear.x = 0.0
            cmd.angular.z = clamp(
                1.2 * alpha,
                -self.p.turn_max_angular_speed,
                self.p.turn_max_angular_speed,
            )
            return cmd

        curvature = 2.0 * math.sin(alpha) / max(self.p.lookahead_distance, 1e-3)

        v = clamp(target.v, 0.0, self.p.max_linear_speed)

        abs_curvature = abs(curvature)

        if abs_curvature > 1.5:
            v *= 0.45
        elif abs_curvature > 0.9:
            v *= 0.65

        w = clamp(v * curvature, -self.p.turn_max_angular_speed, self.p.turn_max_angular_speed)
        w = self.rate_limit(w, self.last_w_odom)
        self.last_w_odom = w

        cmd.linear.x = v
        cmd.angular.z = w

        return cmd

    def find_lookahead_base_link(self, points: List[PathPoint]) -> Optional[PathPoint]:
        follow_lookahead = self.p.lookahead_distance

        for p in points:
            d = math.hypot(p.x, p.y)
            if d >= follow_lookahead and p.x > 0.0:
                return p

        return points[-1] if points else None

    def rate_limit(self, target_w: float, last_w: float) -> float:
        if self.p.control_frequency <= 0.0 or self.p.angular_rate_limit <= 0.0:
            return target_w

        max_delta = self.p.angular_rate_limit / self.p.control_frequency
        return clamp(target_w, last_w - max_delta, last_w + max_delta)

    def find_lookahead_odom(self, points: List[PathPoint], pose: Pose2D) -> Optional[PathPoint]:
        if not points:
            return None

        closest_idx = 0
        closest_dist = float("inf")

        for i, p in enumerate(points):
            d = math.hypot(p.x - pose.x, p.y - pose.y)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        acc = 0.0

        for i in range(closest_idx, len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            segment = math.hypot(p1.x - p0.x, p1.y - p0.y)
            acc += segment

            if acc >= self.p.lookahead_distance:
                return p1

        return points[-1]

    def path_goal_distance(self, path: LocalPath, pose: Pose2D) -> float:
        if not path.points:
            return float("inf")

        goal = path.points[-1]

        return math.hypot(goal.x - pose.x, goal.y - pose.y)

    def path_goal_reached(self, path: LocalPath, pose: Pose2D) -> bool:
        if not path.valid or not path.points:
            return False

        goal = path.points[-1]
        d = math.hypot(goal.x - pose.x, goal.y - pose.y)
        yaw_err = abs(wrap_to_pi(goal.yaw - pose.yaw))

        return d < self.p.path_goal_xy_tolerance and yaw_err < self.p.path_goal_yaw_tolerance


class SafetySupervisor:
    def __init__(self, params: NavigatorParams):
        self.p = params

    def filter_cmd(
        self,
        cmd: Twist,
        scan: Optional[LaserScan],
        row: RowModel,
        state: MissionState,
    ) -> Twist:
        # Safety ist per Parameter abschaltbar.
        # Bei enable_safety=false werden Hindernisse nicht mehr fuer Stop/Slowdown
        # verwendet. Die globalen Geschwindigkeitslimits bleiben aktiv.
        if not self.p.enable_safety:
            cmd.linear.x = clamp(
                cmd.linear.x,
                -self.p.max_linear_speed,
                self.p.max_linear_speed,
            )
            cmd.angular.z = clamp(
                cmd.angular.z,
                -self.p.max_angular_speed,
                self.p.max_angular_speed,
            )
            return cmd

        if scan is None:
            return self.stop()

        front_min = self.front_min_distance(scan, state)

        if front_min < self.p.obstacle_stop_distance:
            return self.stop()

        if front_min < self.p.obstacle_slow_distance:
            cmd.linear.x *= 0.35
            cmd.angular.z *= 0.7

        if state == MissionState.FOLLOW_ROW:
            if row.confidence < 0.05:
                cmd.linear.x *= 0.3

            if row.confidence < self.p.min_follow_confidence:
                cmd.linear.x *= 0.6

        cmd.linear.x = clamp(
            cmd.linear.x,
            -self.p.max_linear_speed,
            self.p.max_linear_speed,
        )
        cmd.angular.z = clamp(
            cmd.angular.z,
            -self.p.max_angular_speed,
            self.p.max_angular_speed,
        )

        return cmd

    def front_min_distance(self, scan: LaserScan, state: MissionState) -> float:
        angle = scan.angle_min
        min_r = float("inf")

        if state == MissionState.FOLLOW_ROW:
            front_angle = math.radians(8.0)
        elif state in (
            MissionState.EXECUTE_TURN,
            MissionState.ACQUIRE_ROW,
            MissionState.ENTER_ROW,
            MissionState.EXIT_ROW,
            MissionState.PLAN_TURN,
        ):
            front_angle = math.radians(35.0)
        else:
            front_angle = math.radians(15.0)

        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                if abs(angle) < front_angle:
                    min_r = min(min_r, r)

            angle += scan.angle_increment

        return min_r

    def stop(self) -> Twist:
        return Twist()


class MissionManager:
    def __init__(self, node: Node, params: NavigatorParams):
        self.node = node
        self.p = params

        self.state = MissionState.IDLE
        self.active_turn_path: Optional[LocalPath] = None
        self.active_turn_uses_map_lane = False
        self.turn_replan_attempts = 0
        self.turn_replan_frame_counter = 0
        self.map_lanes: List[MapLane] = []
        self.map_row_bands: List[MapRowBand] = []
        self.target_map_lane: Optional[MapLane] = None
        self.last_map_row_reason: str = ""
        self.pattern_steps: List[PatternStep] = []
        self.pattern_index = 0

        self.end_stable_frames = 0
        self.enter_stable_frames = 0

        self.acquire_start_time = None

        self.started = False

    def start(self) -> None:
        self.pattern_steps = self.load_pattern()
        self.pattern_index = 0
        self.apply_current_pattern_step()
        self.started = True
        self.transition(MissionState.FOLLOW_ROW, "start requested")

    def stop(self) -> None:
        self.started = False
        self.active_turn_path = None
        self.active_turn_uses_map_lane = False
        self.turn_replan_attempts = 0
        self.turn_replan_frame_counter = 0
        self.target_map_lane = None
        self.map_lanes = []
        self.map_row_bands = []
        self.transition(MissionState.IDLE, "stop requested")

    def load_pattern(self) -> List[PatternStep]:
        try:
            steps = parse_pattern(self.p.pattern)
        except ValueError as exc:
            self.node.get_logger().error(str(exc))
            steps = [PatternStep(self.p.row_shift_count, self.p.row_shift_direction.upper())]

        if len(steps) == 0:
            steps = [PatternStep(self.p.row_shift_count, self.p.row_shift_direction.upper())]

        text = " ".join(f"{step.row_shift_count}{step.row_shift_direction}" for step in steps)
        self.node.get_logger().info(f"Loaded row pattern: {text}")
        return steps

    def apply_current_pattern_step(self) -> None:
        if not self.pattern_steps:
            return

        step = self.pattern_steps[self.pattern_index]
        self.p.row_shift_count = step.row_shift_count
        self.p.row_shift_direction = step.row_shift_direction
        self.node.get_logger().info(
            f"Pattern step {self.pattern_index + 1}/{len(self.pattern_steps)}: "
            f"{self.p.row_shift_count}{self.p.row_shift_direction}"
        )

    def advance_pattern(self) -> bool:
        if self.pattern_index + 1 >= len(self.pattern_steps):
            return False

        self.pattern_index += 1
        self.apply_current_pattern_step()
        return True

    def transition(self, new_state: MissionState, reason: str = "") -> None:
        if new_state == self.state:
            return

        self.node.get_logger().info(
            f"State transition: {self.state.name} -> {new_state.name}"
            + (f" ({reason})" if reason else "")
        )

        self.state = new_state

        if new_state == MissionState.PLAN_TURN:
            self.active_turn_uses_map_lane = False
            self.turn_replan_attempts = 0
            self.turn_replan_frame_counter = 0

        if new_state == MissionState.ACQUIRE_ROW:
            self.acquire_start_time = self.node.get_clock().now()
            self.enter_stable_frames = 0
            self.active_turn_uses_map_lane = False
            self.turn_replan_attempts = 0
            self.turn_replan_frame_counter = 0

        if new_state == MissionState.FOLLOW_ROW:
            self.end_stable_frames = 0
            self.active_turn_uses_map_lane = False
            self.turn_replan_attempts = 0
            self.turn_replan_frame_counter = 0

    def update(
        self,
        row: RowModel,
        pose_odom: Optional[Pose2D],
        pose_map: Optional[Pose2D],
        planner: LocalPlanner,
        controller: PathFollower,
        map_detector: Optional[MapRowDetector] = None,
        latest_map: Optional[OccupancyGrid] = None,
    ) -> LocalPath:
        if not self.started or self.state == MissionState.IDLE:
            return LocalPath([], False, "base_link", "idle")

        if self.state == MissionState.FOLLOW_ROW:
            # Normale Reihenende-Erkennung ueber end_probability.
            if row.end_detected:
                self.end_stable_frames += 1
            else:
                self.end_stable_frames = 0

            if self.end_stable_frames >= self.p.end_stable_frames_required:
                self.transition(MissionState.EXIT_ROW, "row end detected by end_probability")
                return planner.plan_exit_row()

            # Fallback fuer das echte Reihenende:
            # Wenn am Reihenende beide Reihenstrukturen aus dem LiDAR verschwinden,
            # wird das RowModel ungueltig. Ohne diesen Fallback bleibt der Roboter
            # in FOLLOW_ROW stehen, weil plan_follow_row() dann keinen Pfad erzeugt.
            # Die Bedingung ist absichtlich ueber mehrere Frames stabilisiert, damit
            # kurze Luecken in der Reihe nicht sofort als Reihenende interpretiert werden.
            if (
                not row.valid
                and row.missing_frames >= 18
                and row.confidence < 0.03
            ):
                self.node.get_logger().warn(
                    "FOLLOW_ROW fallback: row lost for several frames, "
                    "interpreting as row end and switching to EXIT_ROW"
                )
                self.transition(MissionState.EXIT_ROW, "row lost fallback")
                return planner.plan_exit_row()

            return planner.plan_follow_row(row)

        if self.state == MissionState.EXIT_ROW:
            self.transition(MissionState.PLAN_TURN, "exit row complete")
            return planner.plan_exit_row()

        if self.state == MissionState.PLAN_TURN:
            if self.p.use_slam_map and self.p.require_map_for_turns:
                if pose_map is None:
                    self.node.get_logger().warn(
                        "PLAN_TURN blocked: no map pose available. "
                        "Check TF map -> base_link and SLAM/localization."
                    )
                    return LocalPath([], False, self.p.map_frame, "PLAN_TURN blocked: no map pose")

                if self.p.map_row_detection_enabled:
                    if latest_map is None:
                        return LocalPath([], False, self.p.map_frame, "PLAN_TURN blocked: no SLAM map")

                    if map_detector is None:
                        return LocalPath([], False, self.p.map_frame, "PLAN_TURN blocked: no MapRowDetector")

                    self.map_lanes, self.map_row_bands, self.last_map_row_reason = map_detector.detect_lanes(
                        latest_map,
                        pose_map,
                    )
                    self.target_map_lane = map_detector.select_target_lane(
                        self.map_lanes,
                        self.p.row_shift_direction,
                        self.p.row_shift_count,
                    )

                    if self.target_map_lane is not None:
                        self.active_turn_path = planner.plan_turn_path_to_map_lane(
                            pose_map,
                            self.target_map_lane,
                        )

                        if not self.active_turn_path.valid or len(self.active_turn_path.points) == 0:
                            self.node.get_logger().warn(
                                f"PLAN_TURN failed: invalid SLAM-map lane turn path. "
                                f"reason='{self.active_turn_path.reason}', "
                                f"target_v={self.target_map_lane.center_v:.3f}, "
                                f"lanes={len(self.map_lanes)}, bands={len(self.map_row_bands)}. "
                                f"Falling back to geometric map-frame turn."
                            )
                            self.active_turn_path = planner.plan_turn_path_map(pose_map)
                            self.target_map_lane = None
                        else:
                            self.active_turn_uses_map_lane = True
                            self.turn_replan_attempts = 0
                            self.turn_replan_frame_counter = 0
                            self.node.get_logger().info(
                                f"PLAN_TURN ok: target SLAM-map lane v={self.target_map_lane.center_v:.3f} m, "
                                f"width={self.target_map_lane.width:.3f} m, "
                                f"source={self.target_map_lane.source}, "
                                f"path_points={len(self.active_turn_path.points)}"
                            )
                            self.transition(MissionState.EXECUTE_TURN, "turn path planned to SLAM map lane")
                            return self.active_turn_path
                    else:
                        # Die Zielgasse kann erst dann aus der SLAM-Map erkannt werden,
                        # wenn sie bereits gemappt wurde. Bei einer noch unbekannten
                        # Zielgasse wird deshalb NICHT blockiert. Stattdessen wird im
                        # map-Frame geometrisch anhand expected_row_width gewendet.
                        # Danach faengt ACQUIRE_ROW die neue Gasse lokal mit LiDAR ein,
                        # waehrend SLAM die Zielgasse weiter aufbaut.
                        self.node.get_logger().warn(
                            f"PLAN_TURN fallback: no target lane in SLAM map "
                            f"({self.last_map_row_reason}, lanes={len(self.map_lanes)}, "
                            f"bands={len(self.map_row_bands)}). "
                            f"Using geometric map-frame turn."
                        )
                        self.active_turn_path = planner.plan_turn_path_map(pose_map)

                    if not self.active_turn_path.valid or len(self.active_turn_path.points) == 0:
                        self.node.get_logger().warn(
                            f"PLAN_TURN failed: invalid geometric map-frame fallback path. "
                            f"reason='{self.active_turn_path.reason}', "
                            f"points={len(self.active_turn_path.points)}"
                        )
                        return self.active_turn_path

                    self.active_turn_uses_map_lane = False
                    self.turn_replan_attempts = 0
                    self.turn_replan_frame_counter = 0
                    self.transition(MissionState.EXECUTE_TURN, "turn path planned in map frame with geometric fallback")
                    return self.active_turn_path

                self.active_turn_path = planner.plan_turn_path_map(pose_map)
                self.active_turn_uses_map_lane = False
                self.turn_replan_attempts = 0
                self.turn_replan_frame_counter = 0
                self.transition(MissionState.EXECUTE_TURN, "turn path planned in map frame")
                return self.active_turn_path

            if self.p.use_slam_map and pose_map is not None:
                self.active_turn_path = planner.plan_turn_path_map(pose_map)
                self.active_turn_uses_map_lane = False
                self.turn_replan_attempts = 0
                self.turn_replan_frame_counter = 0
                self.transition(MissionState.EXECUTE_TURN, "turn path planned in map frame")
                return self.active_turn_path

            if pose_odom is not None:
                self.active_turn_path = planner.plan_turn_path_odom(pose_odom)
                self.active_turn_uses_map_lane = False
                self.turn_replan_attempts = 0
                self.turn_replan_frame_counter = 0
                self.transition(MissionState.EXECUTE_TURN, "turn path planned in odom frame")
                return self.active_turn_path

            return LocalPath([], False, self.p.map_frame if self.p.use_slam_map else self.p.odom_frame, "no pose for turn planning")

        if self.state == MissionState.EXECUTE_TURN:
            if self.active_turn_path is None:
                self.transition(MissionState.PLAN_TURN, "missing active turn path")
                return LocalPath([], False, self.p.map_frame if self.p.use_slam_map else self.p.odom_frame, "missing path")

            # Adaptive Replanung waehrend des Wendens:
            # Wenn der Turn nur geometrisch im map-Frame gestartet wurde, die SLAM-Map
            # aber waehrend des Wendens eine Zielgasse sichtbar macht, wird der Rest
            # des Turns ab der aktuellen map-Pose neu zur erkannten Zielgasse geplant.
            if (
                self.p.turn_replan_enabled
                and self.p.use_slam_map
                and self.p.require_map_for_turns
                and self.p.map_row_detection_enabled
                and not self.active_turn_uses_map_lane
                and self.active_turn_path.frame_id == self.p.map_frame
                and pose_map is not None
                and latest_map is not None
                and map_detector is not None
                and self.turn_replan_attempts < max(0, self.p.turn_replan_max_attempts)
            ):
                self.turn_replan_frame_counter += 1

                if self.turn_replan_frame_counter >= max(1, self.p.turn_replan_period_frames):
                    self.turn_replan_frame_counter = 0
                    self.turn_replan_attempts += 1

                    self.map_lanes, self.map_row_bands, self.last_map_row_reason = map_detector.detect_lanes(
                        latest_map,
                        pose_map,
                    )
                    replanning_target_lane = map_detector.select_target_lane(
                        self.map_lanes,
                        self.p.row_shift_direction,
                        self.p.row_shift_count,
                    )

                    if replanning_target_lane is not None:
                        replanned_path = planner.plan_turn_path_to_map_lane(
                            pose_map,
                            replanning_target_lane,
                        )

                        if replanned_path.valid and len(replanned_path.points) > 0:
                            self.active_turn_path = replanned_path
                            self.target_map_lane = replanning_target_lane
                            self.active_turn_uses_map_lane = True

                            self.node.get_logger().warn(
                                "EXECUTE_TURN replan: SLAM target lane became available. "
                                f"Replanned map turn from current pose. "
                                f"target_v={replanning_target_lane.center_v:.3f}, "
                                f"width={replanning_target_lane.width:.3f}, "
                                f"source={replanning_target_lane.source}, "
                                f"lanes={len(self.map_lanes)}, bands={len(self.map_row_bands)}, "
                                f"attempt={self.turn_replan_attempts}"
                            )
                            return self.active_turn_path

                        self.node.get_logger().warn(
                            "EXECUTE_TURN replan candidate rejected: invalid replanned path. "
                            f"reason='{replanned_path.reason}', "
                            f"points={len(replanned_path.points)}, "
                            f"target_v={replanning_target_lane.center_v:.3f}"
                        )

            active_pose = pose_map if self.active_turn_path.frame_id == self.p.map_frame else pose_odom
            if active_pose is not None and controller.path_goal_reached(self.active_turn_path, active_pose):
                self.active_turn_path = None
                self.active_turn_uses_map_lane = False
                self.transition(MissionState.ACQUIRE_ROW, "turn path reached")
                return planner.plan_acquire_row(row)

            return self.active_turn_path

        if self.state == MissionState.ACQUIRE_ROW:
            if row.valid and row.confidence >= self.p.min_enter_confidence:
                self.enter_stable_frames += 1
            else:
                self.enter_stable_frames = 0

            if self.enter_stable_frames >= self.p.enter_stable_frames_required:
                self.transition(MissionState.ENTER_ROW, "target row acquired")
                return planner.plan_enter_row(row)

            if self.acquire_start_time is not None:
                elapsed = (
                    self.node.get_clock().now() - self.acquire_start_time
                ).nanoseconds * 1e-9

                if elapsed > self.p.acquire_timeout_sec:
                    self.node.get_logger().warn(
                        "ACQUIRE_ROW timeout: still searching. Check ROI, turn yaw, row_shift_direction and row markers."
                    )
                    self.acquire_start_time = self.node.get_clock().now()

            return planner.plan_acquire_row(row)

        if self.state == MissionState.ENTER_ROW:
            if row.valid and row.confidence >= self.p.min_follow_confidence:
                self.enter_stable_frames += 1
            else:
                self.enter_stable_frames = 0

            if self.enter_stable_frames >= self.p.enter_stable_frames_required:
                if self.advance_pattern():
                    self.transition(MissionState.FOLLOW_ROW, "stable row following")
                    return planner.plan_follow_row(row)

                self.transition(MissionState.FINISHED, "pattern complete")
                return LocalPath([], False, "base_link", "pattern complete")

            return planner.plan_enter_row(row)

        if self.state == MissionState.FINISHED:
            return LocalPath([], False, "base_link", "finished")

        return LocalPath([], False, "base_link", "unknown state")


class MaizeNavigator(Node):
    def __init__(self):
        super().__init__("maize_navigator")

        self.p = self.load_params()

        self.perception = RowPerception(self.p)
        self.tracker = RowTracker(self.p)
        self.planner = LocalPlanner(self.p)
        self.map_row_detector = MapRowDetector(self.p)
        self.controller = PathFollower(self.p)
        self.safety = SafetySupervisor(self.p)
        self.mission = MissionManager(self, self.p)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_path: Optional[LocalPath] = None
        self.latest_row: RowModel = self.tracker.model

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.p.scan_topic,
            self.scan_callback,
            10,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.p.map_topic,
            self.map_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)

        self.state_pub = self.create_publisher(String, "/debug/state", 10)
        self.conf_pub = self.create_publisher(Float32, "/debug/row_confidence", 10)
        self.end_pub = self.create_publisher(Float32, "/debug/end_probability", 10)
        self.path_pub = self.create_publisher(Path, "/debug/local_path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/debug/row_markers", 10)

        self.diag_pub = self.create_publisher(
            String,
            "/debug/navigation_diagnostics",
            10,
        )
        self.last_diag_text = ""
        self.last_diag_log_time = self.get_clock().now()

        self.start_srv = self.create_service(Trigger, "/start_navigation", self.start_cb)
        self.stop_srv = self.create_service(Trigger, "/stop_navigation", self.stop_cb)

        period = 1.0 / max(self.p.control_frequency, 1.0)
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info("maize_navigator started")
        self.get_logger().info(
            f"scan_topic={self.p.scan_topic}, cmd_vel_topic={self.p.cmd_vel_topic}, "
            f"map_topic={self.p.map_topic}, map_frame={self.p.map_frame}"
        )

    def load_params(self) -> NavigatorParams:
        p = NavigatorParams()

        def declare(name: str, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        p.scan_topic = declare("scan_topic", p.scan_topic)
        p.cmd_vel_topic = declare("cmd_vel_topic", p.cmd_vel_topic)
        p.base_frame = declare("base_frame", p.base_frame)
        p.odom_frame = declare("odom_frame", p.odom_frame)
        p.map_topic = declare("map_topic", p.map_topic)
        p.map_frame = declare("map_frame", p.map_frame)
        p.use_slam_map = bool(declare("use_slam_map", p.use_slam_map))
        p.require_map_for_turns = bool(declare("require_map_for_turns", p.require_map_for_turns))

        p.control_frequency = float(declare("control_frequency", p.control_frequency))

        p.expected_row_width = float(declare("expected_row_width", p.expected_row_width))
        p.min_lane_width = float(declare("min_lane_width", p.min_lane_width))
        p.max_lane_width = float(declare("max_lane_width", p.max_lane_width))

        p.roi_x_min = float(declare("roi_x_min", p.roi_x_min))
        p.roi_x_max = float(declare("roi_x_max", p.roi_x_max))
        p.roi_y_abs_min = float(declare("roi_y_abs_min", p.roi_y_abs_min))
        p.roi_y_abs_max = float(declare("roi_y_abs_max", p.roi_y_abs_max))

        p.acquire_roi_x_min = float(declare("acquire_roi_x_min", p.acquire_roi_x_min))
        p.acquire_roi_x_max = float(declare("acquire_roi_x_max", p.acquire_roi_x_max))
        p.acquire_roi_y_abs_min = float(declare("acquire_roi_y_abs_min", p.acquire_roi_y_abs_min))
        p.acquire_roi_y_abs_max = float(declare("acquire_roi_y_abs_max", p.acquire_roi_y_abs_max))

        p.ransac_iterations = int(declare("ransac_iterations", p.ransac_iterations))
        p.ransac_distance = float(declare("ransac_distance", p.ransac_distance))
        p.min_inliers = int(declare("min_inliers", p.min_inliers))
        p.min_visible_length = float(declare("min_visible_length", p.min_visible_length))
        p.max_abs_line_slope = float(declare("max_abs_line_slope", p.max_abs_line_slope))
        p.centerline_max_abs_slope = float(declare("centerline_max_abs_slope", p.centerline_max_abs_slope))

        p.tracker_alpha = float(declare("tracker_alpha", p.tracker_alpha))
        p.confidence_decay = float(declare("confidence_decay", p.confidence_decay))

        p.front_density_x_min = float(declare("front_density_x_min", p.front_density_x_min))
        p.front_density_x_max = float(declare("front_density_x_max", p.front_density_x_max))
        p.front_density_y_abs = float(declare("front_density_y_abs", p.front_density_y_abs))
        p.front_density_threshold = int(declare("front_density_threshold", p.front_density_threshold))
        p.end_probability_threshold = float(declare("end_probability_threshold", p.end_probability_threshold))
        p.end_stable_frames_required = int(declare("end_stable_frames_required", p.end_stable_frames_required))

        p.min_follow_confidence = float(declare("min_follow_confidence", p.min_follow_confidence))
        p.min_enter_confidence = float(declare("min_enter_confidence", p.min_enter_confidence))
        p.enter_stable_frames_required = int(declare("enter_stable_frames_required", p.enter_stable_frames_required))
        p.acquire_timeout_sec = float(declare("acquire_timeout_sec", p.acquire_timeout_sec))

        p.follow_speed = float(declare("follow_speed", p.follow_speed))
        p.slow_speed = float(declare("slow_speed", p.slow_speed))
        p.enter_speed = float(declare("enter_speed", p.enter_speed))
        p.turn_speed = float(declare("turn_speed", p.turn_speed))
        p.max_linear_speed = float(declare("max_linear_speed", p.max_linear_speed))
        p.max_angular_speed = float(declare("max_angular_speed", p.max_angular_speed))
        p.follow_max_angular_speed = float(declare("follow_max_angular_speed", p.follow_max_angular_speed))
        p.turn_max_angular_speed = float(declare("turn_max_angular_speed", p.turn_max_angular_speed))
        p.angular_rate_limit = float(declare("angular_rate_limit", p.angular_rate_limit))

        p.lookahead_distance = float(declare("lookahead_distance", p.lookahead_distance))
        p.path_goal_xy_tolerance = float(declare("path_goal_xy_tolerance", p.path_goal_xy_tolerance))
        p.path_goal_yaw_tolerance = float(declare("path_goal_yaw_tolerance", p.path_goal_yaw_tolerance))

        p.exit_distance = float(declare("exit_distance", p.exit_distance))
        p.turn_forward_distance = float(declare("turn_forward_distance", p.turn_forward_distance))
        p.min_turn_radius = float(declare("min_turn_radius", p.min_turn_radius))
        p.enter_distance = float(declare("enter_distance", p.enter_distance))
        p.pattern = str(declare("pattern", p.pattern))
        p.row_shift_count = int(declare("row_shift_count", p.row_shift_count))
        p.row_shift_direction = str(declare("row_shift_direction", p.row_shift_direction))
        p.turn_180 = bool(declare("turn_180", p.turn_180))

        p.map_row_detection_enabled = bool(declare("map_row_detection_enabled", p.map_row_detection_enabled))
        p.map_row_occupancy_threshold = int(declare("map_row_occupancy_threshold", p.map_row_occupancy_threshold))
        p.map_row_search_x_forward = float(declare("map_row_search_x_forward", p.map_row_search_x_forward))
        p.map_row_search_x_backward = float(declare("map_row_search_x_backward", p.map_row_search_x_backward))
        p.map_row_search_y_side = float(declare("map_row_search_y_side", p.map_row_search_y_side))
        p.map_row_lateral_bin = float(declare("map_row_lateral_bin", p.map_row_lateral_bin))
        p.map_row_min_band_points = int(declare("map_row_min_band_points", p.map_row_min_band_points))
        p.map_row_min_band_length = float(declare("map_row_min_band_length", p.map_row_min_band_length))
        p.map_row_max_extrapolated_lanes = int(declare("map_row_max_extrapolated_lanes", p.map_row_max_extrapolated_lanes))

        p.turn_replan_enabled = bool(declare("turn_replan_enabled", p.turn_replan_enabled))
        p.turn_replan_period_frames = int(declare("turn_replan_period_frames", p.turn_replan_period_frames))
        p.turn_replan_max_attempts = int(declare("turn_replan_max_attempts", p.turn_replan_max_attempts))

        p.enable_safety = bool(declare("enable_safety", p.enable_safety))
        p.obstacle_stop_distance = float(declare("obstacle_stop_distance", p.obstacle_stop_distance))
        p.obstacle_slow_distance = float(declare("obstacle_slow_distance", p.obstacle_slow_distance))

        p.publish_debug = bool(declare("publish_debug", p.publish_debug))

        return p

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def start_cb(self, request, response):
        self.mission.start()
        response.success = True
        response.message = "navigation started"
        return response

    def stop_cb(self, request, response):
        self.mission.stop()
        self.publish_stop()
        response.success = True
        response.message = "navigation stopped"
        return response

    def control_loop(self) -> None:
        pose_odom = self.lookup_pose(self.p.odom_frame)
        pose_map = self.lookup_pose(self.p.map_frame) if self.p.use_slam_map else None

        if self.latest_scan is None:
            stop_cmd = Twist()
            self.publish_navigation_diagnostics(
                row=self.latest_row,
                path=LocalPath([], False, "base_link", "no LaserScan received"),
                cmd=stop_cmd,
                safe_cmd=stop_cmd,
                pose_odom=pose_odom,
                pose_map=pose_map,
                active_pose=None,
            )
            self.publish_stop()
            return

        self.perception.set_mode(self.mission.state)

        det = self.perception.process_scan(self.latest_scan)
        row = self.tracker.update(det)

        self.latest_row = row

        path = self.mission.update(
            row,
            pose_odom,
            pose_map,
            self.planner,
            self.controller,
            self.map_row_detector,
            self.latest_map,
        )
        self.latest_path = path

        active_pose = pose_map if path.frame_id == self.p.map_frame else pose_odom
        cmd = self.controller.compute_cmd(path, active_pose)
        safe_cmd = self.safety.filter_cmd(
            cmd,
            self.latest_scan,
            row,
            self.mission.state,
        )

        if self.mission.state in (MissionState.IDLE, MissionState.FINISHED):
            safe_cmd = Twist()

        self.publish_navigation_diagnostics(
            row=row,
            path=path,
            cmd=cmd,
            safe_cmd=safe_cmd,
            pose_odom=pose_odom,
            pose_map=pose_map,
            active_pose=active_pose,
        )

        self.cmd_pub.publish(safe_cmd)

        if self.p.publish_debug:
            self.publish_debug(det, row, path)

    def lookup_pose(self, target_frame: str) -> Optional[Pose2D]:
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                self.p.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.02),
            )
        except Exception:
            return None

        q = tf.transform.rotation
        yaw = yaw_from_quaternion(q)

        t = tf.transform.translation

        return Pose2D(float(t.x), float(t.y), float(yaw))

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def publish_navigation_diagnostics(
        self,
        row: RowModel,
        path: LocalPath,
        cmd: Twist,
        safe_cmd: Twist,
        pose_odom: Optional[Pose2D],
        pose_map: Optional[Pose2D],
        active_pose: Optional[Pose2D],
    ) -> None:
        state = self.mission.state

        pose_odom_text = "ok" if pose_odom is not None else "None"
        pose_map_text = "ok" if pose_map is not None else "None"
        active_pose_text = "ok" if active_pose is not None else "None"

        path_points = len(path.points) if path.points is not None else 0

        raw_cmd_zero = (
            abs(cmd.linear.x) < 1e-4
            and abs(cmd.angular.z) < 1e-4
        )

        final_cmd_zero = (
            abs(safe_cmd.linear.x) < 1e-4
            and abs(safe_cmd.angular.z) < 1e-4
        )

        command_changed = (
            abs(cmd.linear.x - safe_cmd.linear.x) > 1e-4
            or abs(cmd.angular.z - safe_cmd.angular.z) > 1e-4
        )

        reasons: List[str] = []

        if self.latest_scan is None:
            reasons.append("no LaserScan received")

        if self.p.use_slam_map and self.p.require_map_for_turns:
            if state in (
                MissionState.EXIT_ROW,
                MissionState.PLAN_TURN,
                MissionState.EXECUTE_TURN,
            ) and pose_map is None:
                reasons.append("map pose missing while map turn is required")

        if path.frame_id == self.p.map_frame and pose_map is None:
            reasons.append("path is in map frame but pose_map is missing")

        if path.frame_id == self.p.odom_frame and pose_odom is None:
            reasons.append("path is in odom frame but pose_odom is missing")

        if not path.valid:
            if path.reason:
                reasons.append(f"path invalid: {path.reason}")
            else:
                reasons.append("path invalid")

        if path.valid and path_points == 0:
            reasons.append("path valid but contains zero points")

        if active_pose is None and path.valid and path.frame_id not in ("base_link", ""):
            reasons.append(f"active pose missing for path frame '{path.frame_id}'")

        if not row.valid and state in (
            MissionState.FOLLOW_ROW,
            MissionState.ACQUIRE_ROW,
            MissionState.ENTER_ROW,
        ):
            reasons.append("row model invalid")

        if row.confidence < self.p.min_follow_confidence and state == MissionState.FOLLOW_ROW:
            reasons.append(
                f"low row confidence: {row.confidence:.3f} < {self.p.min_follow_confidence:.3f}"
            )

        if state == MissionState.PLAN_TURN:
            if pose_map is not None:
                reasons.append("planning map turn")
            else:
                reasons.append("cannot plan map turn without map pose")

        if state == MissionState.EXECUTE_TURN and self.mission.active_turn_path is None:
            reasons.append("EXECUTE_TURN but active_turn_path is None")

        if raw_cmd_zero and state not in (MissionState.IDLE, MissionState.FINISHED):
            reasons.append("controller output is zero")

        if command_changed:
            if self.p.enable_safety:
                reasons.append("safety modified command")
            else:
                reasons.append("command changed although safety is disabled")

        if final_cmd_zero and not raw_cmd_zero and state not in (
            MissionState.IDLE,
            MissionState.FINISHED,
        ):
            reasons.append("final command is zero although controller command was nonzero")

        if self.p.map_row_detection_enabled:
            if state in (MissionState.PLAN_TURN, MissionState.EXECUTE_TURN):
                reasons.append(
                    f"map_lanes={len(self.mission.map_lanes)}, "
                    f"map_bands={len(self.mission.map_row_bands)}, "
                    f"map_reason={self.mission.last_map_row_reason}"
                )

        if state == MissionState.EXECUTE_TURN:
            if self.mission.active_turn_uses_map_lane:
                reasons.append("turn_source=SLAM_map_lane")
            else:
                reasons.append("turn_source=geometric_map_fallback")

        target_lane = self.mission.target_map_lane
        if target_lane is not None:
            target_lane_text = (
                f"valid={target_lane.valid},"
                f"v={target_lane.center_v:.3f},"
                f"width={target_lane.width:.3f},"
                f"conf={target_lane.confidence:.3f},"
                f"source={target_lane.source}"
            )
        else:
            target_lane_text = "None"

        if len(reasons) == 0:
            reasons.append("ok")

        diag_text = (
            f"STATE={state.name}"
            f" | reasons={'; '.join(reasons)}"
            f" | pose_map={pose_map_text}"
            f" | pose_odom={pose_odom_text}"
            f" | active_pose={active_pose_text}"
            f" | path_valid={path.valid}"
            f" | path_frame={path.frame_id}"
            f" | path_points={path_points}"
            f" | path_reason='{path.reason}'"
            f" | row_valid={row.valid}"
            f" | row_missing_frames={row.missing_frames}"
            f" | row_conf={row.confidence:.3f}"
            f" | row_end_prob={row.end_probability:.3f}"
            f" | raw_cmd=({cmd.linear.x:.3f}, {cmd.angular.z:.3f})"
            f" | final_cmd=({safe_cmd.linear.x:.3f}, {safe_cmd.angular.z:.3f})"
            f" | safety_enabled={self.p.enable_safety}"
            f" | map_rows_enabled={self.p.map_row_detection_enabled}"
            f" | map_lanes={len(self.mission.map_lanes)}"
            f" | map_bands={len(self.mission.map_row_bands)}"
            f" | target_map_lane={target_lane_text}"
            f" | active_turn_uses_map_lane={self.mission.active_turn_uses_map_lane}"
            f" | turn_replan_attempts={self.mission.turn_replan_attempts}"
            f" | turn_replan_enabled={self.p.turn_replan_enabled}"
        )

        msg = String()
        msg.data = diag_text
        self.diag_pub.publish(msg)

        self.log_diagnostic_throttled(diag_text)

    def log_diagnostic_throttled(self, text: str) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_diag_log_time).nanoseconds * 1e-9

        should_log = text != self.last_diag_text or dt > 2.0

        if not should_log:
            return

        self.last_diag_text = text
        self.last_diag_log_time = now

        if "reasons=ok" in text:
            self.get_logger().info(text)
        else:
            self.get_logger().warn(text)

    def publish_debug(self, det: RowDetection, row: RowModel, path: LocalPath) -> None:
        state_msg = String()
        state_msg.data = self.mission.state.name
        self.state_pub.publish(state_msg)

        conf_msg = Float32()
        conf_msg.data = float(row.confidence)
        self.conf_pub.publish(conf_msg)

        end_msg = Float32()
        end_msg.data = float(row.end_probability)
        self.end_pub.publish(end_msg)

        self.publish_path(path)
        self.publish_markers(det, row)

    def publish_path(self, path: LocalPath) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = path.frame_id

        for p in path.points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(p.x)
            pose.pose.position.y = float(p.y)
            pose.pose.position.z = 0.0

            q = quaternion_from_yaw(p.yaw)

            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def publish_markers(self, det: RowDetection, row: RowModel) -> None:
        markers = MarkerArray()

        def line_marker(
            marker_id: int,
            a: float,
            b: float,
            color: Tuple[float, float, float],
            valid: bool,
        ):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.p.base_frame
            marker.ns = "rows"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD if valid else Marker.DELETE
            marker.scale.x = 0.035

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            for x in [0.2, 3.0]:
                pt = Point()
                pt.x = x
                pt.y = a * x + b
                pt.z = 0.05
                marker.points.append(pt)

            return marker

        markers.markers.append(
            line_marker(1, det.left_a, det.left_b, (0.0, 1.0, 0.0), det.left_valid)
        )
        markers.markers.append(
            line_marker(2, det.right_a, det.right_b, (0.0, 1.0, 0.0), det.right_valid)
        )
        markers.markers.append(
            line_marker(3, row.center_a, row.center_b, (1.0, 0.5, 0.0), row.valid)
        )

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = MaizeNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
