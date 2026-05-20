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
class MapRowLine:
    valid: bool = False
    a: float = 0.0
    b: float = 0.0
    yaw: float = 0.0
    inliers: int = 0
    length: float = 0.0
    lateral_v: float = 0.0
    confidence: float = 0.0


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

    # Vorgewende-Manoever fuer robusten Gassenwechsel:
    # gesteuerter Bogen raus -> relativer Versatz -> gesteuerter Bogen rein.
    EXIT_CURVE = 8
    HEADLAND_SHIFT = 9
    ENTRY_CURVE = 10


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
    turn_lookahead_distance: float = 0.32
    turn_min_angular_speed: float = 0.30
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

    # Robuster Gassenwechsel im Vorgewende.
    # Die Reihenfahrt selbst bleibt davon unberuehrt.
    headland_maneuver_enabled: bool = True
    # Vor dem eigentlichen Ausfahrbogen wird erst gerade aus der Gasse herausgefahren.
    # Das verhindert, dass der Roboter beim Rechts-/Linksbogen noch an den letzten Pflanzen haengen bleibt.
    headland_exit_straight_distance: float = 0.45
    headland_exit_straight_speed: float = 0.18
    exit_curve_speed: float = 0.18
    exit_curve_angular_speed: float = 0.48
    exit_curve_yaw_change: float = 1.35

    headland_shift_speed: float = 0.22
    headland_shift_tolerance: float = 0.04
    # Schutztoleranz: Wenn der seitliche Versatz beim Einfahrbogen darueber hinausgeht,
    # wird der Bogen abgebrochen. Das verhindert, dass 1L/1R eine weitere Reihe ueberspringt.
    headland_shift_overshoot_tolerance: float = 0.08
    headland_yaw_tolerance: float = 0.25
    # HEADLAND_SHIFT wird in der Map parallel zum Reihenende ausgerichtet.
    # Die Reihenrichtung wird aus der SLAM-Map per PCA auf belegten Pflanzenpunkten geschaetzt.
    headland_use_map_row_heading: bool = True
    headland_heading_kp: float = 1.4
    headland_heading_max_yaw_error: float = 0.75
    # Wenn die Reihenrichtung aus der SLAM-Map vorliegt, wird die Vorgewende-
    # Querfahrt hart auf die dazu senkrechte Richtung gelegt. Der Wert ist nur
    # noch ein Fallback-Grenzwert fuer sehr schlechte Map-Yaw-Schaetzungen.
    headland_map_heading_min_confidence: float = 0.20

    entry_curve_speed: float = 0.16
    entry_curve_angular_speed: float = 0.427
    # Gesamt-Drehwinkel des Vorgewende-Manoevers. Fuer U-Turns muss die
    # Endausrichtung der Gegenrichtung entsprechen.
    headland_total_yaw_change: float = math.pi
    # Falls kleiner/gleich 0, wird der benoetigte Einfahrbogen automatisch als
    # headland_total_yaw_change - exit_curve_yaw_change berechnet.
    entry_curve_yaw_change: float = -1.0
    # ENTRY_CURVE darf erst lokal uebergeben, wenn Yaw und Versatz plausibel sind.
    entry_yaw_accept_tolerance: float = 0.35
    entry_shift_accept_tolerance: float = 0.12
    entry_row_min_confidence: float = 0.32
    entry_row_stable_frames: int = 10
    # Eine lokal erkannte Zielgasse wird erst akzeptiert, wenn sie geometrisch
    # plausibel ist: beide Begrenzungsreihen sichtbar, Gassenmitte nahe y=0,
    # Breite passend und Reihenwinkel klein. Das verhindert Einfaedeln gegen
    # eine einzelne Pflanzenreihe und hilft, mittiger in die neue Gasse zu fahren.
    entry_require_full_lane: bool = True
    entry_center_b_tolerance: float = 0.14
    entry_lane_width_tolerance: float = 0.22
    entry_row_yaw_tolerance: float = 0.35
    # Nach dem U-Turn darf die Umschaltung nicht dauerhaft blockieren, nur weil
    # am Reihenanfang kurzzeitig nur eine Begrenzungsreihe sichtbar ist. Sobald
    # Querposition und Gesamt-Gierwinkel erreicht sind, reicht fuer ACQUIRE/ENTER
    # eine stabile, mittige und nahezu gerade lokale Reihenhypothese aus.
    entry_relaxed_geometry_after_yaw_shift: bool = True
    entry_relaxed_center_b_tolerance: float = 0.28
    entry_relaxed_row_yaw_tolerance: float = 0.45
    # Mehrreihige Einfahrt im Vorgewende: ein Teil des seitlichen Versatzes
    # entsteht rechnerisch im ENTRY_CURVE. In asymmetrischen Reihenenden kann
    # dieser Bogen aber zu frueh in die vordere Begrenzungsreihe schneiden.
    # Fuer 2R/2L/... wird deshalb vor dem Einfahrbogen eine kleine zusaetzliche
    # Querreserve aufgebaut. 1L/1R bleiben unveraendert, damit sie nicht die
    # Nachbargasse ueberspringen.
    multirow_entry_extra_shift: float = 0.18
    # Wenn die per Reihenzaehlung erkannte Zielgasse bei 2R/2L nach innen
    # gegenueber dem Pattern-Sollzentrum verschoben ist, wird sie nicht als
    # Zielzentrum akzeptiert. Das tritt am Reihenende auf, wenn die vordere
    # Pflanzenreihe in die Gasse hineinragt; dann bleibt das pattern-relative
    # Zielzentrum massgebend.
    map_counted_lane_inward_bias_tolerance: float = 0.08
    # Bei Mehrreihensprüngen ist die Map-Querposition am Einfahren weniger
    # belastbar als die lokale Reihenhypothese: durch U-Turn-Geometrie,
    # Schlupf und Reihenanfangs-Asymmetrie kann der gemessene Versatz um
    # ungefähr eine Gassenbreite überschießen. Das darf ACQUIRE_ROW nicht
    # blockieren, sobald die lokale Reihe stabil und gerade ist.
    multirow_entry_shift_overshoot_rows: float = 1.50
    multirow_local_takeover_min_yaw_progress: float = 2.00
    multirow_local_takeover_min_confidence: float = 0.55

    # Direkter Nachbargassenwechsel 1L/1R:
    # Beim Einfahren wird die lokale Zielreihe erst akzeptiert, wenn der
    # pattern-relative Versatz wirklich erreicht ist und die Maisreihe auf
    # der erwarteten Seite sichtbar ist. Dadurch wird verhindert, dass 1L/1R
    # eine Reihe zu weit springt.
    neighbor_reference_turn_enabled: bool = True
    neighbor_reference_entry_requires_shift: bool = True
    neighbor_reference_requires_same_side_row: bool = True

    map_row_detection_enabled: bool = True
    map_row_occupancy_threshold: int = 50
    map_row_search_x_forward: float = 4.0
    map_row_search_x_backward: float = 4.0
    map_row_search_y_side: float = 4.0
    # Reihenrichtung aus belegten SLAM-Map-Punkten schaetzen.
    # Wichtig im Vorgewende: dort ist die Roboter-Yaw nicht mehr parallel zur Reihenrichtung.
    map_row_use_pca_orientation: bool = True
    map_row_pca_radius: float = 5.0
    map_row_pca_min_points: int = 80
    map_row_lateral_bin: float = 0.10
    map_row_min_band_points: int = 12
    map_row_min_band_length: float = 1.2
    map_row_max_extrapolated_lanes: int = 3

    # SLAM-Reihenerkennung ueber Linienfit durch belegte Map-Zellen.
    map_row_line_ransac_iterations: int = 180
    map_row_line_distance: float = 0.12
    map_row_min_line_inliers: int = 18
    map_row_min_line_length: float = 1.20
    map_row_max_abs_line_slope: float = 0.70
    map_row_max_lines: int = 12
    map_row_line_merge_distance: float = 0.22

    # SLAM-Zielgassen duerfen das Pattern nur korrigieren, nicht beliebig weit ueberschreiben.
    # Beispiel: 1L erwartet ca. +expected_row_width. Eine erkannte Lane bei +3 m wird verworfen.
    map_lane_accept_tolerance: float = 0.45

    turn_replan_enabled: bool = True
    turn_replan_period_frames: int = 5
    turn_replan_max_attempts: int = 60

    # Wenn die neue Gasse lokal mit LiDAR stabil erkannt wird, wird EXECUTE_TURN beendet.
    # Dadurch faehrt der Roboter nicht an der richtigen Gasse vorbei.
    turn_exit_on_local_row: bool = True
    turn_exit_min_confidence: float = 0.32
    turn_exit_stable_frames: int = 10

    enable_safety: bool = False

    obstacle_stop_distance: float = 0.25
    obstacle_slow_distance: float = 0.45

    publish_debug: bool = True


class MapRowDetector:
    def __init__(self, params: NavigatorParams):
        self.p = params
        self.last_target_reason: str = "not evaluated"
        self.last_expected_target_offset: float = 0.0
        self.last_detected_target_offset: float = 0.0
        self.last_target_offset_error: float = 0.0
        self.last_reference_row_v: float = 0.0
        self.last_target_row_v: float = 0.0
        self.last_candidate_rows_text: str = ""
        self.last_row_yaw_map: Optional[float] = None
        self.last_row_yaw_confidence: float = 0.0

    def detect_lanes(self, grid: Optional[OccupancyGrid], pose: Pose2D) -> Tuple[List[MapLane], List[MapRowBand], str]:
        """Detect plant row lines from occupied SLAM-map cells and derive lane centerlines.

        The map is transformed into the robot-local frame:
        u = forward along the currently driven row direction,
        v = left/right across the rows.

        We do not search for free space. We fit lines through occupied plant cells.
        Neighboring plant-row lines form a lane. The lane center_v is therefore
        the center between two fitted plant rows.
        """
        if grid is None:
            return [], [], "no OccupancyGrid"

        if not self.p.map_row_detection_enabled:
            return [], [], "map row detection disabled"

        u, v, reason = self._occupied_points_local(grid, pose)
        if u is None or v is None:
            return [], [], reason

        if len(u) < self.p.map_row_min_line_inliers:
            return [], [], f"not enough occupied cells in local map window: {len(u)}"

        row_lines = self._fit_row_lines(u, v)
        row_lines = self._merge_row_lines(row_lines)
        row_lines.sort(key=lambda line: line.lateral_v)

        # Keep MapRowBand for existing diagnostics/visual language. Each band now
        # represents one fitted plant-row line, not a pure lateral histogram band.
        bands = [
            MapRowBand(
                valid=True,
                lateral_v=float(line.lateral_v),
                u_min=-0.5 * float(line.length),
                u_max=0.5 * float(line.length),
                points=int(line.inliers),
            )
            for line in row_lines
        ]

        if len(row_lines) < 2:
            return [], bands, f"not enough row lines fitted: {len(row_lines)}"

        lanes: List[MapLane] = []

        for right, left in zip(row_lines[:-1], row_lines[1:]):
            width_v = left.lateral_v - right.lateral_v

            if not (self.p.min_lane_width <= width_v <= self.p.max_lane_width):
                continue

            slope_error = abs(left.a - right.a)
            parallel_score = clamp(1.0 - slope_error / 0.35, 0.0, 1.0)

            width_error = abs(width_v - self.p.expected_row_width)
            width_score = clamp(1.0 - width_error / max(self.p.expected_row_width, 1e-3), 0.0, 1.0)

            length_score = clamp(
                min(left.length, right.length) / max(self.p.map_row_min_line_length, 1e-3),
                0.0,
                1.0,
            )

            inlier_score = clamp(
                min(left.inliers, right.inliers) / max(float(self.p.map_row_min_line_inliers), 1.0),
                0.0,
                1.0,
            )

            confidence = clamp(
                0.30 * parallel_score
                + 0.30 * width_score
                + 0.25 * length_score
                + 0.15 * inlier_score,
                0.0,
                1.0,
            )

            lanes.append(
                MapLane(
                    valid=True,
                    center_v=0.5 * (right.lateral_v + left.lateral_v),
                    left_row_v=left.lateral_v,
                    right_row_v=right.lateral_v,
                    width=width_v,
                    confidence=confidence,
                    source="detected_linefit",
                )
            )

        lanes.sort(key=lambda lane: lane.center_v)

        if len(lanes) == 0:
            return [], bands, f"row lines fitted, but no valid lane gap: lines={len(row_lines)}"

        return lanes, bands, f"ok: linefit rows={len(row_lines)}, lanes={len(lanes)}"

    def _occupied_points_local(
        self,
        grid: OccupancyGrid,
        pose: Pose2D,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        width = int(grid.info.width)
        height = int(grid.info.height)
        resolution = float(grid.info.resolution)

        if width <= 0 or height <= 0 or resolution <= 0.0:
            return None, None, "invalid OccupancyGrid metadata"

        data = np.asarray(grid.data, dtype=np.int16).reshape((height, width))
        occ_r, occ_c = np.where(data >= int(self.p.map_row_occupancy_threshold))

        if len(occ_r) == 0:
            return None, None, "no occupied map cells"

        # Deterministic downsampling for real-time behavior on large maps.
        max_cells = 35000
        if len(occ_r) > max_cells:
            idx = np.linspace(0, len(occ_r) - 1, max_cells).astype(int)
            occ_r = occ_r[idx]
            occ_c = occ_c[idx]

        origin = grid.info.origin
        origin_yaw = yaw_from_quaternion(origin.orientation)
        co = math.cos(origin_yaw)
        so = math.sin(origin_yaw)

        gx = (occ_c.astype(float) + 0.5) * resolution
        gy = (occ_r.astype(float) + 0.5) * resolution

        mx = origin.position.x + co * gx - so * gy
        my = origin.position.y + so * gx + co * gy

        dx = mx - pose.x
        dy = my - pose.y

        # Die Linien der Maisreihen sollen in der SLAM-Map erkannt werden,
        # auch wenn der Roboter im Vorgewende quer zu den Reihen steht.
        # Deshalb wird die Reihenrichtung nicht aus pose.yaw abgeleitet,
        # sondern aus den belegten Kartenpunkten per PCA geschaetzt.
        row_yaw = pose.yaw
        self.last_row_yaw_map = None
        self.last_row_yaw_confidence = 0.0

        if self.p.map_row_use_pca_orientation:
            radius = max(1.0, float(self.p.map_row_pca_radius))
            near = (dx * dx + dy * dy) <= radius * radius
            near_dx = dx[near]
            near_dy = dy[near]

            if len(near_dx) >= max(10, int(self.p.map_row_pca_min_points)):
                x0 = near_dx - float(np.mean(near_dx))
                y0 = near_dy - float(np.mean(near_dy))
                cov_xx = float(np.mean(x0 * x0))
                cov_yy = float(np.mean(y0 * y0))
                cov_xy = float(np.mean(x0 * y0))
                angle = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)

                # PCA liefert eine Achse ohne Richtung. Fuer konsistente u-Koordinaten
                # wird die Achse so gewaehlt, dass sie moeglichst zur aktuellen oder
                # entgegengesetzten Fahrtrichtung passt.
                if abs(wrap_to_pi(angle + math.pi - pose.yaw)) < abs(wrap_to_pi(angle - pose.yaw)):
                    angle = wrap_to_pi(angle + math.pi)

                lambda_sum = cov_xx + cov_yy
                lambda_diff = math.sqrt((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy * cov_xy)
                conf = clamp(lambda_diff / max(lambda_sum, 1e-6), 0.0, 1.0)

                if conf >= 0.20:
                    row_yaw = angle
                    self.last_row_yaw_map = float(row_yaw)
                    self.last_row_yaw_confidence = float(conf)

        cp = math.cos(row_yaw)
        sp = math.sin(row_yaw)

        # Row-local map coordinates:
        # u: entlang der Maisreihen, v: quer ueber die Reihen.
        u = cp * dx + sp * dy
        v = -sp * dx + cp * dy

        mask = (
            (u >= -self.p.map_row_search_x_backward)
            & (u <= self.p.map_row_search_x_forward)
            & (np.abs(v) <= self.p.map_row_search_y_side)
        )

        u = u[mask]
        v = v[mask]

        if len(u) == 0:
            return None, None, "no occupied cells inside row-oriented map window"

        return u.astype(float), v.astype(float), "ok"

    def _fit_row_lines(self, u: np.ndarray, v: np.ndarray) -> List[MapRowLine]:
        lines: List[MapRowLine] = []

        remaining_u = np.asarray(u, dtype=float)
        remaining_v = np.asarray(v, dtype=float)

        rng = random.Random(42)
        max_lines = max(1, int(self.p.map_row_max_lines))

        for _ in range(max_lines):
            if len(remaining_u) < self.p.map_row_min_line_inliers:
                break

            line, inlier_mask = self._fit_one_line_ransac(remaining_u, remaining_v, rng)

            if not line.valid or inlier_mask is None:
                break

            lines.append(line)

            keep = ~inlier_mask
            remaining_u = remaining_u[keep]
            remaining_v = remaining_v[keep]

        return lines

    def _fit_one_line_ransac(
        self,
        u: np.ndarray,
        v: np.ndarray,
        rng: random.Random,
    ) -> Tuple[MapRowLine, Optional[np.ndarray]]:
        n = len(u)
        if n < self.p.map_row_min_line_inliers:
            return MapRowLine(valid=False), None

        best_mask: Optional[np.ndarray] = None
        best_count = 0
        best_a = 0.0
        best_b = 0.0

        iterations = max(1, int(self.p.map_row_line_ransac_iterations))
        distance_threshold = max(0.02, float(self.p.map_row_line_distance))
        max_slope = max(0.05, float(self.p.map_row_max_abs_line_slope))

        indices = list(range(n))

        for _ in range(iterations):
            i1, i2 = rng.sample(indices, 2)
            du = float(u[i2] - u[i1])
            if abs(du) < 1e-3:
                continue

            a = float((v[i2] - v[i1]) / du)
            if abs(a) > max_slope:
                continue

            b = float(v[i1] - a * u[i1])
            denom = math.sqrt(a * a + 1.0)
            distances = np.abs(a * u - v + b) / denom
            mask = distances <= distance_threshold
            count = int(np.count_nonzero(mask))

            if count > best_count:
                best_count = count
                best_mask = mask
                best_a = a
                best_b = b

        if best_mask is None or best_count < self.p.map_row_min_line_inliers:
            return MapRowLine(valid=False), None

        inlier_u = u[best_mask]
        inlier_v = v[best_mask]

        if len(inlier_u) < self.p.map_row_min_line_inliers:
            return MapRowLine(valid=False), None

        try:
            a, b = np.polyfit(inlier_u, inlier_v, 1)
            a = float(a)
            b = float(b)
        except Exception:
            a = best_a
            b = best_b

        if abs(a) > max_slope:
            return MapRowLine(valid=False), None

        length = float(max(inlier_u) - min(inlier_u)) if len(inlier_u) else 0.0
        if length < self.p.map_row_min_line_length:
            return MapRowLine(valid=False), None

        yaw = math.atan(a)
        confidence = clamp(
            0.55 * (best_count / max(float(self.p.map_row_min_line_inliers), 1.0))
            + 0.45 * (length / max(float(self.p.map_row_min_line_length), 1e-3)),
            0.0,
            1.0,
        )

        return (
            MapRowLine(
                valid=True,
                a=float(a),
                b=float(b),
                yaw=float(yaw),
                inliers=int(best_count),
                length=float(length),
                lateral_v=float(b),
                confidence=float(confidence),
            ),
            best_mask,
        )

    def _merge_row_lines(self, lines: List[MapRowLine]) -> List[MapRowLine]:
        valid_lines = [line for line in lines if line.valid]
        valid_lines.sort(key=lambda line: line.lateral_v)

        if not valid_lines:
            return []

        merge_distance = max(0.05, float(self.p.map_row_line_merge_distance))
        merged: List[MapRowLine] = []

        for line in valid_lines:
            if not merged:
                merged.append(line)
                continue

            prev = merged[-1]
            if abs(line.lateral_v - prev.lateral_v) <= merge_distance:
                total = max(prev.inliers + line.inliers, 1)
                w_prev = prev.inliers / total
                w_line = line.inliers / total

                merged[-1] = MapRowLine(
                    valid=True,
                    a=w_prev * prev.a + w_line * line.a,
                    b=w_prev * prev.b + w_line * line.b,
                    yaw=w_prev * prev.yaw + w_line * line.yaw,
                    inliers=prev.inliers + line.inliers,
                    length=max(prev.length, line.length),
                    lateral_v=w_prev * prev.lateral_v + w_line * line.lateral_v,
                    confidence=max(prev.confidence, line.confidence),
                )
            else:
                merged.append(line)

        return merged

    def _make_band(self, u_values: np.ndarray, v_values: np.ndarray) -> MapRowBand:
        # Kept for compatibility with older debug semantics.
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
        row_bands: List[MapRowBand],
        direction: str,
        count: int,
    ) -> Optional[MapLane]:
        """Select the target lane by counting fitted plant-row lines from the current side row.

        This is deliberately row-line-relative, not absolute-lane-relative:
        - nL counts plant rows to the left of the currently driven lane.
        - nR counts plant rows to the right of the currently driven lane.

        For 1L, the current left plant row and the next left plant row form the
        target lane. For 2R, the second and third right-side row lines form the
        target lane. If the SLAM map does not yet contain enough row lines, the
        target is extrapolated from the nearest visible reference row instead of
        choosing an arbitrary far-away lane.
        """
        step = max(1, int(count))
        sign = 1 if direction.upper() == "L" else -1
        expected_offset = sign * step * self.p.expected_row_width
        tolerance = max(0.05, float(self.p.map_lane_accept_tolerance))

        self.last_expected_target_offset = float(expected_offset)
        self.last_detected_target_offset = 0.0
        self.last_target_offset_error = 0.0
        self.last_reference_row_v = 0.0
        self.last_target_row_v = 0.0
        self.last_candidate_rows_text = ""

        width = clamp(
            float(self.p.expected_row_width),
            self.p.min_lane_width,
            self.p.max_lane_width,
        )

        row_positions = sorted(
            float(band.lateral_v)
            for band in row_bands
            if band.valid and math.isfinite(band.lateral_v)
        )

        if sign > 0:
            side_rows = [v for v in row_positions if v > 0.0]
            side_rows.sort(key=lambda value: value)
        else:
            side_rows = [v for v in row_positions if v < 0.0]
            side_rows.sort(key=lambda value: -value)

        self.last_candidate_rows_text = ",".join(f"{v:.3f}" for v in side_rows[:8])

        # Primary mode: count fitted plant-row lines on the requested side.
        # side_rows[0] is the current boundary row. The target lane for step k lies
        # between side_rows[k-1] and side_rows[k].
        if len(side_rows) >= step + 1:
            inner_row_v = float(side_rows[step - 1])
            outer_row_v = float(side_rows[step])
            center_v = 0.5 * (inner_row_v + outer_row_v)
            detected_offset = float(center_v)
            error = abs(detected_offset - expected_offset)

            self.last_reference_row_v = float(side_rows[0])
            self.last_target_row_v = outer_row_v
            self.last_detected_target_offset = detected_offset
            self.last_target_offset_error = float(error)

            # A counted target is only valid if both the pattern-relative center
            # and the row spacing are plausible. The previous version accepted
            # row pairs with only ~0.25...0.43 m separation for a 0.75 m crop row.
            # That creates a false "lane" on a plant row and makes 2R drive into
            # the crop. Narrow row pairs are rejected and the deterministic
            # pattern-limited center is used instead.
            measured_width = abs(outer_row_v - inner_row_v)
            width_min = max(0.05, float(self.p.min_lane_width))
            width_max = max(width_min, float(self.p.max_lane_width))
            width_ok = width_min <= measured_width <= width_max
            # signed_outward_error > 0: counted lane is farther outward than the
            # pattern target. signed_outward_error < 0: counted lane is shifted
            # inward, i.e. back toward the currently driven lane. For multi-row
            # turns this inward bias is dangerous at the headland because a
            # protruding front row can pull the target center into the crop.
            signed_outward_error = sign * (detected_offset - expected_offset)
            inward_bias_limit = max(0.0, float(self.p.map_counted_lane_inward_bias_tolerance))
            inward_bias_ok = not (step >= 2 and signed_outward_error < -inward_bias_limit)

            if error <= tolerance and width_ok and inward_bias_ok:
                width = float(measured_width)
                self.last_target_reason = (
                    f"row-line counted target accepted: direction={direction.upper()}, "
                    f"step={step}, reference_row={side_rows[0]:.3f}, "
                    f"inner_row={inner_row_v:.3f}, outer_row={outer_row_v:.3f}, "
                    f"center={center_v:.3f}, expected={expected_offset:.3f}, "
                    f"error={error:.3f}, width={measured_width:.3f}, "
                    f"signed_outward_error={signed_outward_error:.3f}, "
                    f"width_range=[{width_min:.3f},{width_max:.3f}], tolerance={tolerance:.3f}, "
                    f"side_rows=[{self.last_candidate_rows_text}]"
                )
                return MapLane(
                    valid=True,
                    center_v=float(center_v),
                    left_row_v=float(max(inner_row_v, outer_row_v)),
                    right_row_v=float(min(inner_row_v, outer_row_v)),
                    width=float(width),
                    confidence=0.98,
                    source="detected_reference_row_linefit",
                )

            self.last_target_reason = (
                f"row-line counted target rejected: direction={direction.upper()}, "
                f"step={step}, reference_row={side_rows[0]:.3f}, "
                f"inner_row={inner_row_v:.3f}, outer_row={outer_row_v:.3f}, "
                f"center={center_v:.3f}, expected={expected_offset:.3f}, "
                f"error={error:.3f}, width={measured_width:.3f}, "
                f"width_ok={width_ok}, inward_bias_ok={inward_bias_ok}, "
                f"signed_outward_error={signed_outward_error:.3f}, "
                f"width_range=[{width_min:.3f},{width_max:.3f}], "
                f"tolerance={tolerance:.3f}, side_rows=[{self.last_candidate_rows_text}]"
            )

        elif len(side_rows) >= 1:
            # If only the current boundary row is known, extrapolate from it.
            reference_row_v = float(side_rows[0])
            center_v = reference_row_v + sign * (step - 0.5) * self.p.expected_row_width
            error = abs(center_v - expected_offset)

            self.last_reference_row_v = reference_row_v
            self.last_target_row_v = reference_row_v + sign * step * self.p.expected_row_width
            self.last_detected_target_offset = float(center_v)
            self.last_target_offset_error = float(error)

            self.last_target_reason = (
                f"not enough fitted row lines on requested side; "
                f"direction={direction.upper()}, step={step}, "
                f"reference_row={reference_row_v:.3f}, side_rows=[{self.last_candidate_rows_text}]; "
                f"using reference-row extrapolated lane center={center_v:.3f}"
            )

            return MapLane(
                valid=True,
                center_v=float(center_v),
                left_row_v=float(center_v + 0.5 * width),
                right_row_v=float(center_v - 0.5 * width),
                width=float(width),
                confidence=0.55,
                source="extrapolated_from_reference_row",
            )

        else:
            self.last_target_reason = (
                f"no fitted reference row on requested side: direction={direction.upper()}, "
                f"step={step}, rows={len(row_positions)}"
            )

        # Secondary fallback: use a fitted lane only if its relative offset matches the pattern.
        lanes_sorted = sorted(lanes, key=lambda lane: lane.center_v)
        candidates = []
        for lane in lanes_sorted:
            detected_offset = float(lane.center_v)
            error = abs(detected_offset - expected_offset)
            same_side = (sign > 0 and detected_offset > 0.0) or (sign < 0 and detected_offset < 0.0)
            if same_side:
                candidates.append((error, lane))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            best_error, best_lane = candidates[0]
            self.last_detected_target_offset = float(best_lane.center_v)
            self.last_target_offset_error = float(best_error)

            if best_error <= tolerance:
                best_lane.source = "detected_lane_fallback"
                self.last_target_reason += (
                    f"; fallback fitted lane accepted: expected={expected_offset:.3f}, "
                    f"detected={best_lane.center_v:.3f}, error={best_error:.3f}"
                )
                return best_lane

            self.last_target_reason += (
                f"; fallback fitted lane rejected: expected={expected_offset:.3f}, "
                f"detected={best_lane.center_v:.3f}, error={best_error:.3f}, "
                f"tolerance={tolerance:.3f}"
            )

        if step > self.p.map_row_max_extrapolated_lanes:
            self.last_target_reason += (
                f"; no extrapolation because requested step {step} exceeds "
                f"map_row_max_extrapolated_lanes={self.p.map_row_max_extrapolated_lanes}"
            )
            return None

        # Last resort: pure pattern-limited lane center. This never jumps beyond
        # the requested count.
        center_v = expected_offset
        self.last_detected_target_offset = float(center_v)
        self.last_target_offset_error = 0.0
        self.last_target_reason += (
            f"; using pure pattern-limited extrapolated lane center={center_v:.3f} m"
        )

        return MapLane(
            valid=True,
            center_v=float(center_v),
            left_row_v=float(center_v + 0.5 * width),
            right_row_v=float(center_v - 0.5 * width),
            width=float(width),
            confidence=0.40,
            source="extrapolated_pattern_limited",
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

        return LocalPath(odom_points, True, frame_id, "TURN_GEOMETRIC_UTURN")


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

        return LocalPath(map_points, True, self.p.map_frame, "TURN_SLAM_LANE_UTURN")


    def plan_constant_curve_base_link(
        self,
        direction: float,
        speed: float,
        angular_speed: float,
        duration_hint: float = 3.0,
        reason: str = "MANEUVER_CURVE",
    ) -> LocalPath:
        """Generate a short base_link arc used as a commanded forward curve.

        This does not replace row following. It is only used in the headland
        maneuver where a controlled left/right arc is desired.
        """
        direction = 1.0 if direction >= 0.0 else -1.0
        v = clamp(abs(speed), 0.0, self.p.max_linear_speed)
        w = max(abs(angular_speed), 1e-3)
        radius = max(v / w, 0.05)
        total_angle = min(abs(w * max(duration_hint, 0.5)), 1.6)

        points: List[PathPoint] = []
        for phi in np.linspace(0.05, total_angle, 36):
            x = radius * math.sin(float(phi))
            y = direction * radius * (1.0 - math.cos(float(phi)))
            yaw = direction * float(phi)
            points.append(PathPoint(float(x), float(y), float(yaw), float(v)))

        return LocalPath(points, True, "base_link", reason)

    def plan_headland_shift_base_link(
        self,
        speed: Optional[float] = None,
        reason: str = "HEADLAND_SHIFT",
    ) -> LocalPath:
        """Drive forward in the current heading during the headland lateral shift."""
        v = self.p.headland_shift_speed if speed is None else speed
        v = clamp(abs(v), 0.0, self.p.max_linear_speed)
        points: List[PathPoint] = []
        for x in np.linspace(0.20, 1.60, 24):
            points.append(PathPoint(float(x), 0.0, 0.0, float(v)))
        return LocalPath(points, True, "base_link", reason)

    def plan_headland_shift_map_heading(
        self,
        pose: Pose2D,
        desired_yaw: float,
        speed: Optional[float] = None,
        reason: str = "HEADLAND_SHIFT_MAP",
    ) -> LocalPath:
        """Map-frame straight segment used to keep the headland drive parallel to row ends."""
        v = self.p.headland_shift_speed if speed is None else speed
        v = clamp(abs(v), 0.0, self.p.max_linear_speed)
        points: List[PathPoint] = []
        c = math.cos(desired_yaw)
        s = math.sin(desired_yaw)
        for d in np.linspace(0.25, 1.80, 28):
            dist = float(d)
            points.append(
                PathPoint(
                    float(pose.x + c * dist),
                    float(pose.y + s * dist),
                    float(desired_yaw),
                    float(v),
                )
            )
        return LocalPath(points, True, self.p.map_frame, reason)


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

        # Vorgewende-Kurven werden als direktes Geschwindigkeitskommando gefahren,
        # nicht mit Pure Pursuit. Dadurch ergibt sich ein reproduzierbarer Radius:
        # R = v / omega. Mit den Defaultwerten liegt R bei ca. 0.375 m.
        if path.reason in ("EXIT_CURVE", "ENTRY_CURVE"):
            direction = 1.0
            if len(path.points) > 0 and (path.points[-1].yaw < 0.0 or path.points[-1].y < 0.0):
                direction = -1.0

            if path.reason == "EXIT_CURVE":
                v_cmd = clamp(abs(self.p.exit_curve_speed), 0.0, self.p.max_linear_speed)
                w_cmd = direction * min(abs(self.p.exit_curve_angular_speed), self.p.turn_max_angular_speed)
            else:
                v_cmd = clamp(abs(self.p.entry_curve_speed), 0.0, self.p.max_linear_speed)
                w_cmd = direction * min(abs(self.p.entry_curve_angular_speed), self.p.turn_max_angular_speed)

            w_cmd = self.rate_limit(w_cmd, self.last_w_base)
            self.last_w_base = w_cmd
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd
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

        is_turn_path = path.reason.startswith("TURN")

        if is_turn_path:
            target = self.find_lookahead_odom(
                path.points,
                pose,
                lookahead_distance=self.p.turn_lookahead_distance,
            )
        else:
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

        lookahead = self.p.turn_lookahead_distance if is_turn_path else self.p.lookahead_distance
        curvature = 2.0 * math.sin(alpha) / max(lookahead, 1e-3)

        v = clamp(target.v, 0.0, self.p.max_linear_speed)

        abs_curvature = abs(curvature)

        if is_turn_path:
            # Beim Vorgewende-U-Turn muss aktiv eingelenkt werden.
            # Zu grosser Lookahead fuehrt sonst zu fast gerader Fahrt entlang des Reihenendes.
            if abs_curvature > 1.5:
                v *= 0.70
            elif abs_curvature > 0.9:
                v *= 0.85
        else:
            if abs_curvature > 1.5:
                v *= 0.45
            elif abs_curvature > 0.9:
                v *= 0.65

        w = clamp(v * curvature, -self.p.turn_max_angular_speed, self.p.turn_max_angular_speed)

        if is_turn_path and abs(w) > 1e-4:
            w = math.copysign(
                max(abs(w), self.p.turn_min_angular_speed),
                w,
            )
            w = clamp(w, -self.p.turn_max_angular_speed, self.p.turn_max_angular_speed)

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

    def find_lookahead_odom(
        self,
        points: List[PathPoint],
        pose: Pose2D,
        lookahead_distance: Optional[float] = None,
    ) -> Optional[PathPoint]:
        if not points:
            return None

        follow_lookahead = self.p.lookahead_distance if lookahead_distance is None else lookahead_distance

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

            if acc >= follow_lookahead:
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
        self.turn_local_row_stable_frames = 0

        # Vorgewende-Manoever-Zustand.
        self.headland_start_pose_map: Optional[Pose2D] = None
        self.exit_curve_start_yaw: Optional[float] = None
        self.entry_curve_start_yaw: Optional[float] = None
        self.headland_direction: float = 1.0
        self.headland_required_shift: float = 0.0
        self.headland_measured_shift: float = 0.0
        self.headland_exit_forward_distance: float = 0.0
        self.headland_reference_row_yaw_map: Optional[float] = None
        self.entry_row_stable_frames: int = 0
        self.entry_reference_side_ok: bool = False
        self.entry_shift_ok: bool = False

        self.map_lanes: List[MapLane] = []
        self.map_row_bands: List[MapRowBand] = []
        self.target_map_lane: Optional[MapLane] = None
        self.last_map_row_reason: str = ""
        self.last_target_lane_reason: str = "not evaluated"
        self.expected_target_offset: float = 0.0
        self.detected_target_offset: float = 0.0
        self.target_offset_error: float = 0.0
        self.reference_row_v: float = 0.0
        self.target_row_v: float = 0.0
        self.candidate_rows_text: str = ""
        self.pattern_steps: List[PatternStep] = []
        self.pattern_index = 0
        self.pattern_completed = False

        self.end_stable_frames = 0
        self.enter_stable_frames = 0

        self.acquire_start_time = None

        self.started = False

    def start(self) -> None:
        self.pattern_steps = self.load_pattern()
        self.pattern_index = 0
        self.pattern_completed = False
        self.apply_current_pattern_step()
        self.started = True
        self.transition(MissionState.FOLLOW_ROW, "start requested")

    def stop(self) -> None:
        self.started = False
        self.active_turn_path = None
        self.active_turn_uses_map_lane = False
        self.turn_replan_attempts = 0
        self.turn_replan_frame_counter = 0
        self.turn_local_row_stable_frames = 0
        self.headland_start_pose_map = None
        self.exit_curve_start_yaw = None
        self.entry_curve_start_yaw = None
        self.headland_direction = 1.0
        self.headland_required_shift = 0.0
        self.headland_measured_shift = 0.0
        self.headland_exit_forward_distance = 0.0
        self.headland_reference_row_yaw_map = None
        self.entry_row_stable_frames = 0
        self.entry_reference_side_ok = False
        self.entry_shift_ok = False
        self.target_map_lane = None
        self.map_lanes = []
        self.map_row_bands = []
        self.last_target_lane_reason = "not evaluated"
        self.expected_target_offset = 0.0
        self.detected_target_offset = 0.0
        self.target_offset_error = 0.0
        self.reference_row_v = 0.0
        self.target_row_v = 0.0
        self.candidate_rows_text = ""
        self.pattern_completed = False
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
            self.turn_local_row_stable_frames = 0

        if new_state == MissionState.EXIT_CURVE:
            self.active_turn_path = None
            self.headland_start_pose_map = None
            self.active_turn_uses_map_lane = False
            self.turn_replan_attempts = 0
            self.turn_replan_frame_counter = 0
            self.turn_local_row_stable_frames = 0
            self.exit_curve_start_yaw = None
            self.entry_curve_start_yaw = None
            self.entry_row_stable_frames = 0
            self.headland_direction = 1.0 if self.p.row_shift_direction.upper() == "L" else -1.0
            self.headland_required_shift = self.headland_direction * self.p.row_shift_count * self.p.expected_row_width
            self.headland_measured_shift = 0.0
            self.headland_exit_forward_distance = 0.0
            self.headland_reference_row_yaw_map = None

        if new_state == MissionState.HEADLAND_SHIFT:
            self.entry_curve_start_yaw = None
            self.entry_row_stable_frames = 0

        if new_state == MissionState.ENTRY_CURVE:
            self.entry_curve_start_yaw = None
            self.entry_row_stable_frames = 0
            self.entry_reference_side_ok = False
            self.entry_shift_ok = False

        if new_state == MissionState.ACQUIRE_ROW:
            self.acquire_start_time = self.node.get_clock().now()
            self.enter_stable_frames = 0
            self.active_turn_uses_map_lane = False
            self.turn_replan_attempts = 0
            self.turn_replan_frame_counter = 0
            self.turn_local_row_stable_frames = 0

        if new_state == MissionState.FOLLOW_ROW:
            self.end_stable_frames = 0
            self.active_turn_uses_map_lane = False
            self.turn_replan_attempts = 0
            self.turn_replan_frame_counter = 0
            self.turn_local_row_stable_frames = 0

        if new_state == MissionState.ENTER_ROW:
            self.turn_local_row_stable_frames = 0

    def _update_headland_measured_shift(self, pose_map: Optional[Pose2D]) -> None:
        if pose_map is None or self.headland_start_pose_map is None:
            return

        dx = pose_map.x - self.headland_start_pose_map.x
        dy = pose_map.y - self.headland_start_pose_map.y
        # Fuer Vorgewende-Manoever wird der Quer-Versatz in der Map-
        # Reihenbasis gemessen, sobald diese verfuegbar ist. Damit haengt der
        # Versatz nicht mehr davon ab, ob der Roboter beim Reihenende schon
        # leicht schraeg stand.
        yaw0 = self.headland_reference_row_yaw_map
        if yaw0 is None:
            yaw0 = self.headland_start_pose_map.yaw
        self.headland_measured_shift = -math.sin(float(yaw0)) * dx + math.cos(float(yaw0)) * dy

    def _update_headland_exit_forward_distance(self, pose_map: Optional[Pose2D]) -> None:
        if pose_map is None or self.headland_start_pose_map is None:
            return

        dx = pose_map.x - self.headland_start_pose_map.x
        dy = pose_map.y - self.headland_start_pose_map.y
        yaw0 = self.headland_start_pose_map.yaw
        self.headland_exit_forward_distance = math.cos(yaw0) * dx + math.sin(yaw0) * dy

    def _effective_entry_curve_yaw_change(self) -> float:
        if self.p.entry_curve_yaw_change > 0.0:
            return self.p.entry_curve_yaw_change

        return max(0.20, self.p.headland_total_yaw_change - self.p.exit_curve_yaw_change)

    def _predicted_entry_lateral_shift(self) -> float:
        # Der zweite Bogen veraendert den lateralen Versatz weiter.
        # Bisher wurde HEADLAND_SHIFT bis zum gesamten Reihenabstand gefahren
        # und danach kam nochmals seitlicher Versatz durch ENTRY_CURVE dazu.
        # Das erzeugt genau den Fehler: bei 1L landet der Roboter eine oder
        # mehrere Reihen zu weit. Deshalb wird HEADLAND_SHIFT vor dem Einfahrbogen
        # gestoppt. Der fehlende Rest wird vom Einfahrbogen erzeugt.
        direction = 1.0 if self.headland_direction >= 0.0 else -1.0
        entry_yaw = self._effective_entry_curve_yaw_change()
        exit_yaw = max(0.0, self.p.exit_curve_yaw_change)
        radius = max(abs(self.p.entry_curve_speed) / max(abs(self.p.entry_curve_angular_speed), 1e-3), 0.05)

        lateral = radius * (math.cos(exit_yaw) - math.cos(exit_yaw + entry_yaw))
        return direction * lateral

    def _headland_pre_entry_shift_target(self) -> float:
        required = abs(self.headland_required_shift)
        predicted_entry = abs(self._predicted_entry_lateral_shift())

        # Bei 2R/2L und groesseren Spruengen darf der Einfahrbogen nicht schon
        # direkt an der theoretischen Tangente beginnen. Die reale Feldkante ist
        # am Vorgewende oft versetzt; die vordere Begrenzungsreihe kann in die
        # neue Gasse hineinragen. Eine kleine Querreserve verlagert den
        # Bogenbeginn nach aussen und verhindert das Hineinschneiden in diese
        # Reihe. Direkte Nachbargassenwechsel bleiben ohne Zusatzreserve.
        extra_shift = 0.0
        if int(self.p.row_shift_count) >= 2:
            extra_shift = max(0.0, float(self.p.multirow_entry_extra_shift))

        return max(0.0, required - predicted_entry + extra_shift)

    def _headland_shift_reached(self) -> bool:
        required = self._headland_pre_entry_shift_target()
        measured = abs(self.headland_measured_shift)
        if required <= 1e-6:
            return True
        return measured >= required - max(0.0, self.p.headland_shift_tolerance)

    def _entry_shift_window_ok(self) -> bool:
        required = abs(self.headland_required_shift)
        measured = abs(self.headland_measured_shift)
        if required <= 1e-6:
            return True

        tolerance = max(0.0, float(self.p.entry_shift_accept_tolerance))

        # Für direkte Nachbargassenwechsel muss die Querposition eng um das
        # Sollzentrum liegen; sonst wird leicht eine Reihe übersprungen.
        if int(self.p.row_shift_count) <= 1:
            return abs(measured - required) <= tolerance

        # Bei 2R/3R/... darf ein moderater Überschuss nicht dazu führen, dass
        # ACQUIRE_ROW minutenlang geradeaus weiterfährt. Genau das erzeugte im
        # 3R-Log den Drift: required=2.25 m, measured≈3.0 m, lokale Reihe stabil,
        # aber entry_shift_ok blieb false. Mehrreihige Turns akzeptieren deshalb
        # "Sollversatz erreicht" plus begrenzten Überschuss. Die Begrenzung
        # verhindert, dass eine komplett falsche Gasse übernommen wird.
        max_overshoot = max(
            tolerance,
            max(0.0, float(self.p.multirow_entry_shift_overshoot_rows))
            * max(0.05, float(self.p.expected_row_width)),
        )
        return (measured >= required - tolerance) and (measured <= required + max_overshoot)

    def _headland_total_yaw_progress(self, pose_map: Optional[Pose2D]) -> float:
        if pose_map is None or self.exit_curve_start_yaw is None:
            return 0.0

        # Der U-Turn liegt nominal bei pi rad. wrap_to_pi() springt direkt nach
        # Ueberschreiten von pi auf negative Werte. Genau dadurch wurde im Log
        # aus einer bereits ueberdrehten Rechtswende ein Fortschritt von ca.
        # -2.95 rad; _entry_yaw_ok() blieb false und ACQUIRE_ROW blockierte
        # dauerhaft. Der Drehfortschritt wird deshalb entlang der gewuenschten
        # Drehrichtung auf [0, 2*pi) entrollt.
        progress = self.headland_direction * wrap_to_pi(pose_map.yaw - float(self.exit_curve_start_yaw))
        target = max(0.0, float(self.p.headland_total_yaw_change))

        if target > math.pi / 2.0 and progress < -max(0.0, float(self.p.entry_yaw_accept_tolerance)):
            progress += 2.0 * math.pi

        return progress

    def _entry_yaw_ok(self, pose_map: Optional[Pose2D]) -> bool:
        tolerance = max(0.0, float(self.p.entry_yaw_accept_tolerance))
        map_target_yaw = self._target_entry_row_yaw_from_map(pose_map, None)
        if pose_map is not None and map_target_yaw is not None:
            return abs(wrap_to_pi(float(map_target_yaw) - pose_map.yaw)) <= tolerance

        progress = self._headland_total_yaw_progress(pose_map)
        target = max(0.0, float(self.p.headland_total_yaw_change))
        return progress >= target - tolerance

    def _entry_reference_side_detected(self, row: RowModel) -> bool:
        if not self.p.neighbor_reference_turn_enabled:
            return True

        if self.p.row_shift_count != 1:
            return True

        if not self.p.neighbor_reference_requires_same_side_row:
            return True

        det = row.last_detection
        if det is None:
            return False

        direction = self.p.row_shift_direction.upper()
        if direction == "L":
            return bool(det.left_valid)
        if direction == "R":
            return bool(det.right_valid)

        return True

    def _entry_row_geometry_ok(self, row: RowModel) -> bool:
        if not self.p.entry_require_full_lane:
            return True

        if not row.valid:
            return False

        det = row.last_detection
        if det is None:
            return False

        both_rows_visible = bool(det.left_valid and det.right_valid)
        center_ok = abs(row.center_b) <= max(0.02, float(self.p.entry_center_b_tolerance))
        yaw_ok = abs(row.row_yaw_base) <= max(0.02, float(self.p.entry_row_yaw_tolerance))

        width_ok = True
        if det.lane_width > 0.05:
            width_ok = abs(det.lane_width - self.p.expected_row_width) <= max(0.05, float(self.p.entry_lane_width_tolerance))

        return bool(both_rows_visible and center_ok and yaw_ok and width_ok)

    def _entry_row_relaxed_geometry_ok(self, row: RowModel) -> bool:
        if not self.p.entry_relaxed_geometry_after_yaw_shift:
            return False

        if not row.valid:
            return False

        center_ok = abs(row.center_b) <= max(0.02, float(self.p.entry_relaxed_center_b_tolerance))
        yaw_ok = abs(row.row_yaw_base) <= max(0.02, float(self.p.entry_relaxed_row_yaw_tolerance))
        return bool(center_ok and yaw_ok)

    def _entry_local_row_takeover_ok(self, row: RowModel, pose_map: Optional[Pose2D]) -> bool:
        if int(self.p.row_shift_count) < 2:
            return False

        if not row.valid:
            return False

        min_conf = max(
            float(self.p.entry_row_min_confidence),
            float(self.p.multirow_local_takeover_min_confidence),
        )
        if row.confidence < min_conf:
            return False

        if not self._entry_row_relaxed_geometry_ok(row):
            return False

        progress = self._headland_total_yaw_progress(pose_map)
        min_progress = max(0.0, float(self.p.multirow_local_takeover_min_yaw_progress))
        return progress >= min_progress

    def _entry_yaw_or_local_row_ok(self, row: RowModel, pose_map: Optional[Pose2D]) -> bool:
        if self._entry_yaw_ok(pose_map):
            return True

        # Bei 3R war die lokale Reihe bereits mit hoher Konfidenz erfasst, der
        # entrollte Gesamt-Gierfortschritt lag aber noch bei ca. 2.23 rad. Das
        # blockierte ACQUIRE_ROW, obwohl das Weiterfahren ohne Reihenführung den
        # Roboter quer aus der Gasse driften ließ. Eine stabile, mittige und
        # gerade lokale Reihenhypothese darf deshalb den restlichen Yaw-Fehler
        # übernehmen.
        return self._entry_local_row_takeover_ok(row, pose_map)

    def _multirow_safe_acquire_steering_ok(self, row: RowModel, shift_ok: bool) -> bool:
        if int(self.p.row_shift_count) < 2:
            return False
        if not shift_ok:
            return False
        if not row.valid:
            return False
        return row.confidence >= max(
            float(self.p.entry_row_min_confidence),
            float(self.p.multirow_local_takeover_min_confidence),
        )

    def _entry_geometry_accept_ok(self, row: RowModel, shift_ok: bool, yaw_ok: bool) -> bool:
        if self._entry_row_geometry_ok(row):
            return True

        # Fuer 2R/2L und groessere Spruenge verhindert die volle Gassenpruefung
        # das fruehe Einhaken in eine einzelne Reihe. Sobald der pattern-relative
        # Versatz und die U-Turn-Ausrichtung bzw. eine stabile lokale
        # Reihenhypothese erreicht sind, darf die Uebergabe ausloesen. Sonst
        # bleibt ACQUIRE_ROW am Reihenanfang stehen, wenn nur eine Begrenzungsreihe
        # im Sichtfeld liegt.
        if int(self.p.row_shift_count) >= 2 and shift_ok and yaw_ok:
            return self._entry_row_relaxed_geometry_ok(row)

        return False

    def _capture_headland_reference_row_yaw(self, map_detector: Optional[MapRowDetector]) -> None:
        """Latch the map row axis for the whole headland maneuver.

        PCA yaw is axial and can flip by pi between frames. Latching one
        orientation prevents sign changes in the lateral shift calculation and
        keeps the headland path exactly perpendicular to the map rows.
        """
        if not self.p.headland_use_map_row_heading:
            return
        if self.headland_reference_row_yaw_map is not None:
            return
        if map_detector is None or map_detector.last_row_yaw_map is None:
            return
        if map_detector.last_row_yaw_confidence < max(0.0, float(self.p.headland_map_heading_min_confidence)):
            return

        row_yaw = float(map_detector.last_row_yaw_map)
        if self.headland_start_pose_map is not None:
            # PCA has no direction. Use the orientation closest to the row yaw at
            # the moment the robot left the previous row.
            if abs(wrap_to_pi(row_yaw + math.pi - self.headland_start_pose_map.yaw)) < abs(wrap_to_pi(row_yaw - self.headland_start_pose_map.yaw)):
                row_yaw = wrap_to_pi(row_yaw + math.pi)

        self.headland_reference_row_yaw_map = row_yaw

    def _desired_headland_shift_yaw(self, pose_map: Pose2D, map_detector: Optional[MapRowDetector]) -> Optional[float]:
        if not self.p.headland_use_map_row_heading:
            return None

        self._capture_headland_reference_row_yaw(map_detector)
        if self.headland_reference_row_yaw_map is None:
            return None

        # Querfahrt im Vorgewende: exakt senkrecht zur aus der Map bekannten
        # Reihenrichtung. Fuer L ist das die positive v-Achse, fuer R die
        # negative v-Achse. Kein Umschalten auf die jeweils naehere Gegenrichtung,
        # denn das kann bei 3R die physikalisch falsche Seite waehlen.
        row_yaw = float(self.headland_reference_row_yaw_map)
        desired = wrap_to_pi(row_yaw + self.headland_direction * math.pi / 2.0)
        return desired

    def _target_entry_row_yaw_from_map(self, pose_map: Optional[Pose2D], map_detector: Optional[MapRowDetector]) -> Optional[float]:
        if pose_map is None or not self.p.headland_use_map_row_heading:
            return None

        self._capture_headland_reference_row_yaw(map_detector)
        if self.headland_reference_row_yaw_map is None:
            return None

        row_yaw = float(self.headland_reference_row_yaw_map)
        if self.exit_curve_start_yaw is not None:
            nominal = wrap_to_pi(float(self.exit_curve_start_yaw) + self.headland_direction * max(0.0, float(self.p.headland_total_yaw_change)))
        else:
            nominal = wrap_to_pi(pose_map.yaw + self.headland_direction * max(0.0, float(self.p.headland_total_yaw_change)))

        candidates = [wrap_to_pi(row_yaw), wrap_to_pi(row_yaw + math.pi)]
        return min(candidates, key=lambda yaw: abs(wrap_to_pi(yaw - nominal)))

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
                if self.pattern_completed:
                    self.transition(MissionState.FINISHED, "final row end detected by end_probability")
                    return LocalPath([], False, "base_link", "pattern complete at final row end")

                if self.p.headland_maneuver_enabled:
                    self.transition(MissionState.EXIT_CURVE, "row end detected; starting headland maneuver")
                    return planner.plan_headland_shift_base_link(
                        speed=self.p.headland_exit_straight_speed,
                        reason="EXIT_STRAIGHT",
                    )

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
                if self.pattern_completed:
                    self.node.get_logger().warn(
                        "FOLLOW_ROW fallback: final row lost for several frames, "
                        "interpreting as final row end and stopping"
                    )
                    self.transition(MissionState.FINISHED, "final row lost fallback")
                    return LocalPath([], False, "base_link", "pattern complete at final row end")

                self.node.get_logger().warn(
                    "FOLLOW_ROW fallback: row lost for several frames, "
                    "interpreting as row end and switching to EXIT_ROW"
                )
                if self.p.headland_maneuver_enabled:
                    self.transition(MissionState.EXIT_CURVE, "row lost fallback; starting headland maneuver")
                    return planner.plan_headland_shift_base_link(
                        speed=self.p.headland_exit_straight_speed,
                        reason="EXIT_STRAIGHT",
                    )

                self.transition(MissionState.EXIT_ROW, "row lost fallback")
                return planner.plan_exit_row()

            return planner.plan_follow_row(row)


        if self.state == MissionState.EXIT_CURVE:
            if not self.p.use_slam_map or not self.p.require_map_for_turns:
                return LocalPath([], False, self.p.map_frame, "headland maneuver requires map pose")

            if pose_map is None:
                return LocalPath([], False, self.p.map_frame, "EXIT_CURVE blocked: no map pose")

            if self.headland_start_pose_map is None:
                self.headland_start_pose_map = pose_map
                self.exit_curve_start_yaw = None
                self.headland_direction = 1.0 if self.p.row_shift_direction.upper() == "L" else -1.0
                self.headland_required_shift = self.headland_direction * self.p.row_shift_count * self.p.expected_row_width
                self.headland_measured_shift = 0.0
                self.headland_exit_forward_distance = 0.0
                self.node.get_logger().info(
                    f"HEADLAND_EXIT_STRAIGHT start: direction={self.p.row_shift_direction}, "
                    f"count={self.p.row_shift_count}, required_shift={self.headland_required_shift:.3f} m, "
                    f"straight_distance={self.p.headland_exit_straight_distance:.3f} m"
                )

            # Zuerst gerade aus der Reihe herausfahren. Erst danach beginnt der eigentliche
            # Links-/Rechtsbogen. Das schafft am Reihenende Abstand zu den letzten Pflanzen
            # und verhindert besonders beim 2. Manöver, dass der Bogen noch in die Reihe schneidet.
            self._update_headland_exit_forward_distance(pose_map)
            self._update_headland_measured_shift(pose_map)

            if self.exit_curve_start_yaw is None:
                if self.headland_exit_forward_distance < max(0.0, self.p.headland_exit_straight_distance):
                    return planner.plan_headland_shift_base_link(
                        speed=self.p.headland_exit_straight_speed,
                        reason="EXIT_STRAIGHT",
                    )

                self.exit_curve_start_yaw = pose_map.yaw
                self.node.get_logger().info(
                    f"EXIT_CURVE arc start after straight clearance: "
                    f"forward={self.headland_exit_forward_distance:.3f} m"
                )

            yaw_progress = self.headland_direction * wrap_to_pi(pose_map.yaw - float(self.exit_curve_start_yaw))
            if yaw_progress >= self.p.exit_curve_yaw_change:
                self.transition(MissionState.HEADLAND_SHIFT, "exit curve yaw reached")
                return planner.plan_headland_shift_base_link(reason="HEADLAND_SHIFT")

            return planner.plan_constant_curve_base_link(
                self.headland_direction,
                self.p.exit_curve_speed,
                self.p.exit_curve_angular_speed,
                reason="EXIT_CURVE",
            )

        if self.state == MissionState.HEADLAND_SHIFT:
            if pose_map is None:
                return LocalPath([], False, self.p.map_frame, "HEADLAND_SHIFT blocked: no map pose")

            if self.headland_start_pose_map is None:
                self.headland_start_pose_map = pose_map
                self.headland_direction = 1.0 if self.p.row_shift_direction.upper() == "L" else -1.0
                self.headland_required_shift = self.headland_direction * self.p.row_shift_count * self.p.expected_row_width

            self._update_headland_measured_shift(pose_map)

            remaining = self._headland_pre_entry_shift_target() - abs(self.headland_measured_shift)

            # SLAM-Reihenlinien weiterhin auswerten, aber hier nur fuer Diagnose und
            # Plausibilisierung. Das Manoever faehrt nicht beliebig zu einer absoluten
            # Map-Lane, sondern nur den pattern-relativen Sollversatz.
            if self.p.map_row_detection_enabled and latest_map is not None and map_detector is not None:
                self.map_lanes, self.map_row_bands, self.last_map_row_reason = map_detector.detect_lanes(
                    latest_map,
                    pose_map,
                )
                _ = map_detector.select_target_lane(
                    self.map_lanes,
                    self.map_row_bands,
                    self.p.row_shift_direction,
                    self.p.row_shift_count,
                )
                self.last_target_lane_reason = map_detector.last_target_reason
                self.expected_target_offset = map_detector.last_expected_target_offset
                self.detected_target_offset = map_detector.last_detected_target_offset
                self.target_offset_error = map_detector.last_target_offset_error
                self.reference_row_v = map_detector.last_reference_row_v
                self.target_row_v = map_detector.last_target_row_v
                self.candidate_rows_text = map_detector.last_candidate_rows_text
                self._capture_headland_reference_row_yaw(map_detector)
                self._update_headland_measured_shift(pose_map)
                remaining = self._headland_pre_entry_shift_target() - abs(self.headland_measured_shift)

            if remaining <= self.p.headland_shift_tolerance:
                self.transition(MissionState.ENTRY_CURVE, "required headland shift reached")
                return planner.plan_constant_curve_base_link(
                    self.headland_direction,
                    self.p.entry_curve_speed,
                    self.p.entry_curve_angular_speed,
                    reason="ENTRY_CURVE",
                )

            desired_shift_yaw = self._desired_headland_shift_yaw(pose_map, map_detector)
            if desired_shift_yaw is not None:
                return planner.plan_headland_shift_map_heading(
                    pose_map,
                    desired_shift_yaw,
                    reason="HEADLAND_SHIFT_MAP",
                )

            return planner.plan_headland_shift_base_link(reason="HEADLAND_SHIFT")

        if self.state == MissionState.ENTRY_CURVE:
            if pose_map is None:
                return LocalPath([], False, self.p.map_frame, "ENTRY_CURVE blocked: no map pose")

            if self.entry_curve_start_yaw is None:
                self.entry_curve_start_yaw = pose_map.yaw
                self.entry_row_stable_frames = 0

            # Sobald die Zielgasse lokal stabil sichtbar ist, uebernimmt die
            # vorhandene gute Einfahr-/Reihenfuehrung.
            # Fuer direkte Nachbargassenwechsel 1L/1R wird die lokale Reihe aber
            # erst akzeptiert, wenn der geforderte seitliche Versatz erreicht ist
            # und die Maisreihe auf der erwarteten Seite sichtbar ist. Das verhindert,
            # dass eine frueh erkannte falsche Struktur als Zielgasse genommen wird.
            self._update_headland_measured_shift(pose_map)

            # Harte Begrenzung gegen Ueberspringen: Wenn der seitliche
            # Gesamtversatz groesser ist als der Pattern-Sollversatz plus
            # Schutztoleranz, wird nicht weiter in die Kurve hineingefahren.
            # Danach wird langsam lokal gesucht, statt weiter seitlich abzudriften.
            overshoot_tolerance = max(0.0, float(self.p.headland_shift_overshoot_tolerance))
            if int(self.p.row_shift_count) >= 2:
                overshoot_tolerance += max(0.0, float(self.p.multirow_entry_extra_shift))

            if abs(self.headland_measured_shift) > abs(self.headland_required_shift) + overshoot_tolerance:
                self.node.get_logger().warn(
                    f"ENTRY_CURVE overshoot guard: measured_shift={self.headland_measured_shift:.3f}, "
                    f"required_shift={self.headland_required_shift:.3f}, "
                    f"tolerance={overshoot_tolerance:.3f}. Switching to ACQUIRE_ROW."
                )
                self.transition(MissionState.ACQUIRE_ROW, "entry curve overshoot guard")
                return planner.plan_acquire_row(row)

            self.entry_shift_ok = self._entry_shift_window_ok()
            self.entry_reference_side_ok = self._entry_reference_side_detected(row)
            entry_yaw_ok = self._entry_yaw_or_local_row_ok(row, pose_map)
            entry_geometry_ok = self._entry_geometry_accept_ok(row, self.entry_shift_ok, entry_yaw_ok)

            if self.p.neighbor_reference_entry_requires_shift and self.p.row_shift_count == 1:
                entry_shift_condition = self.entry_shift_ok
            else:
                # Auch bei uebersprungenen Gassen darf lokal erst uebergeben werden,
                # wenn der pattern-relative Versatz plausibel ist. Sonst kann eine
                # zufaellige Zwischenstruktur als Zielgasse akzeptiert werden.
                entry_shift_condition = self.entry_shift_ok

            if (
                row.valid
                and row.confidence >= self.p.entry_row_min_confidence
                and entry_shift_condition
                and entry_yaw_ok
                and self.entry_reference_side_ok
                and entry_geometry_ok
            ):
                self.entry_row_stable_frames += 1
            else:
                self.entry_row_stable_frames = 0

            if self.entry_row_stable_frames >= max(1, self.p.entry_row_stable_frames):
                self.transition(MissionState.ENTER_ROW, "target row detected during entry curve with yaw/shift/reference checks")
                return planner.plan_enter_row(row)

            map_target_yaw = self._target_entry_row_yaw_from_map(pose_map, map_detector)
            if map_target_yaw is not None:
                yaw_error_to_map_row = abs(wrap_to_pi(float(map_target_yaw) - pose_map.yaw))
                if yaw_error_to_map_row <= max(0.0, float(self.p.entry_yaw_accept_tolerance)):
                    self.transition(MissionState.ACQUIRE_ROW, "entry curve aligned to map row yaw; target lane not geometrically centered yet")
                    return planner.plan_acquire_row(row)

            yaw_progress = self.headland_direction * wrap_to_pi(pose_map.yaw - float(self.entry_curve_start_yaw))
            target_entry_yaw = self._effective_entry_curve_yaw_change()
            if yaw_progress >= target_entry_yaw:
                self.transition(MissionState.ACQUIRE_ROW, "entry curve completed; target lane not geometrically centered yet")
                return planner.plan_acquire_row(row)

            return planner.plan_constant_curve_base_link(
                self.headland_direction,
                self.p.entry_curve_speed,
                self.p.entry_curve_angular_speed,
                reason="ENTRY_CURVE",
            )

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
                        self.map_row_bands,
                        self.p.row_shift_direction,
                        self.p.row_shift_count,
                    )
                    self.last_target_lane_reason = map_detector.last_target_reason
                    self.expected_target_offset = map_detector.last_expected_target_offset
                    self.detected_target_offset = map_detector.last_detected_target_offset
                    self.target_offset_error = map_detector.last_target_offset_error
                    self.reference_row_v = map_detector.last_reference_row_v
                    self.target_row_v = map_detector.last_target_row_v
                    self.candidate_rows_text = map_detector.last_candidate_rows_text

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
                            # Nur echte, in der SLAM-Map detektierte Zielgassen gelten als final.
                            # Extrapolierte Zielgassen werden gefahren, duerfen aber waehrend
                            # EXECUTE_TURN neu geplant werden, sobald die echte Zielgasse sichtbar wird.
                            self.active_turn_uses_map_lane = self.target_map_lane.source.startswith("detected")
                            self.turn_replan_attempts = 0
                            self.turn_replan_frame_counter = 0
                            self.node.get_logger().info(
                                f"PLAN_TURN ok: target lane v={self.target_map_lane.center_v:.3f} m, "
                                f"width={self.target_map_lane.width:.3f} m, "
                                f"source={self.target_map_lane.source}, "
                                f"final_slam_lane={self.active_turn_uses_map_lane}, "
                                f"path_points={len(self.active_turn_path.points)}"
                            )
                            self.transition(MissionState.EXECUTE_TURN, "turn path planned to target lane")
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
                        self.map_row_bands,
                        self.p.row_shift_direction,
                        self.p.row_shift_count,
                    )
                    self.last_target_lane_reason = map_detector.last_target_reason
                    self.expected_target_offset = map_detector.last_expected_target_offset
                    self.detected_target_offset = map_detector.last_detected_target_offset
                    self.target_offset_error = map_detector.last_target_offset_error
                    self.reference_row_v = map_detector.last_reference_row_v
                    self.target_row_v = map_detector.last_target_row_v
                    self.candidate_rows_text = map_detector.last_candidate_rows_text

                    if (
                        replanning_target_lane is not None
                        and replanning_target_lane.source.startswith("detected")
                    ):
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

            # Frueher Ausstieg aus dem globalen Turn:
            # Wenn die Zielgasse lokal mit LiDAR stabil erkannt wird, wird der globale
            # Turn-Pfad verlassen. Sonst kann der Roboter trotz richtiger lokaler
            # Erkennung an der Zielgasse vorbei bis zu einer spaeteren Reihe fahren.
            if self.p.turn_exit_on_local_row:
                if row.valid and row.confidence >= self.p.turn_exit_min_confidence:
                    self.turn_local_row_stable_frames += 1
                else:
                    self.turn_local_row_stable_frames = 0

                if self.turn_local_row_stable_frames >= max(1, self.p.turn_exit_stable_frames):
                    self.node.get_logger().warn(
                        "EXECUTE_TURN early exit: target row detected locally. "
                        f"row_conf={row.confidence:.3f}, stable_frames={self.turn_local_row_stable_frames}. "
                        "Switching to ENTER_ROW before global turn path goal."
                    )
                    self.active_turn_path = None
                    self.active_turn_uses_map_lane = False
                    self.transition(MissionState.ENTER_ROW, "target row detected during turn")
                    return planner.plan_enter_row(row)

            if active_pose is not None and controller.path_goal_reached(self.active_turn_path, active_pose):
                self.active_turn_path = None
                self.active_turn_uses_map_lane = False
                self.transition(MissionState.ACQUIRE_ROW, "turn path reached")
                return planner.plan_acquire_row(row)

            return self.active_turn_path

        if self.state == MissionState.ACQUIRE_ROW:
            acquire_shift_ok = True
            acquire_yaw_ok = True
            acquire_reference_ok = True
            acquire_geometry_ok = True

            if self.p.headland_maneuver_enabled and self.headland_required_shift != 0.0:
                if pose_map is not None and self.headland_start_pose_map is not None:
                    self._update_headland_measured_shift(pose_map)
                acquire_shift_ok = self._entry_shift_window_ok()
                acquire_yaw_ok = self._entry_yaw_or_local_row_ok(row, pose_map)
                acquire_reference_ok = self._entry_reference_side_detected(row)
                acquire_geometry_ok = self._entry_geometry_accept_ok(row, acquire_shift_ok, acquire_yaw_ok)

            acquire_row_ok = (
                row.valid
                and row.confidence >= self.p.min_enter_confidence
                and acquire_shift_ok
                and acquire_yaw_ok
                and acquire_reference_ok
                and acquire_geometry_ok
            )

            if acquire_row_ok:
                self.enter_stable_frames += 1
            else:
                self.enter_stable_frames = 0

            if self.enter_stable_frames >= self.p.enter_stable_frames_required:
                self.transition(MissionState.ENTER_ROW, "target row acquired with shift/yaw/reference/geometry guards")
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

            # Critical safety guard: do not let ACQUIRE_ROW follow a single or
            # geometrically invalid row too early. Bei 3R war der Sollversatz
            # jedoch bereits erreicht/überschritten und die lokale Reihe stabil;
            # weiteres Geradeausfahren ohne Reihenregelung ließ den Roboter aus
            # der Gasse driften. Für Mehrreihensprünge wird dann langsam mit der
            # lokalen Reihenhypothese nachgeführt, statt stur geradeaus weiter zu
            # fahren.
            if self.p.headland_maneuver_enabled and self.headland_required_shift != 0.0 and not acquire_row_ok:
                if self._multirow_safe_acquire_steering_ok(row, acquire_shift_ok):
                    return planner.plan_acquire_row(row)
                return planner.plan_headland_shift_base_link(
                    speed=min(self.p.enter_speed, self.p.slow_speed),
                    reason="ACQUIRE_ROW_WAIT_FULL_LANE",
                )

            return planner.plan_acquire_row(row)

        if self.state == MissionState.ENTER_ROW:
            if pose_map is not None and self.headland_start_pose_map is not None:
                self._update_headland_measured_shift(pose_map)

            shift_ok_for_follow = True
            yaw_ok_for_follow = True
            reference_ok_for_follow = True

            if self.p.headland_maneuver_enabled and self.headland_required_shift != 0.0:
                shift_ok_for_follow = self._entry_shift_window_ok()
                yaw_ok_for_follow = self._entry_yaw_or_local_row_ok(row, pose_map)
                reference_ok_for_follow = self._entry_reference_side_detected(row)

            geometry_ok_for_follow = (
                self._entry_geometry_accept_ok(row, shift_ok_for_follow, yaw_ok_for_follow)
                if self.p.headland_maneuver_enabled
                else True
            )

            enter_guards_ok = (
                row.valid
                and row.confidence >= self.p.min_follow_confidence
                and shift_ok_for_follow
                and yaw_ok_for_follow
                and reference_ok_for_follow
                and geometry_ok_for_follow
            )

            if enter_guards_ok:
                self.enter_stable_frames += 1
            else:
                self.enter_stable_frames = 0

            if self.enter_stable_frames >= self.p.enter_stable_frames_required:
                if self.advance_pattern():
                    self.transition(MissionState.FOLLOW_ROW, "stable row following")
                    return planner.plan_follow_row(row)

                # Alle Pattern-Wechsel wurden ausgefuehrt.
                # Wichtig: Die zuletzt erreichte Gasse wird jetzt noch komplett durchfahren.
                # Gestoppt wird erst am Ende dieser finalen Gasse.
                self.pattern_completed = True
                self.node.get_logger().info(
                    "All pattern transitions executed. Driving final row; will stop at next row end."
                )
                self.transition(MissionState.FOLLOW_ROW, "stable final row following")
                return planner.plan_follow_row(row)

            # Same guard as ACQUIRE_ROW. Bei Mehrreihensprüngen darf eine stabile
            # lokale Reihe nach erreichtem Sollversatz weiter als Lenkreferenz
            # dienen, auch wenn die vollständige Gassengeometrie noch nicht
            # bestätigt ist.
            if self.p.headland_maneuver_enabled and self.headland_required_shift != 0.0 and not enter_guards_ok:
                if self._multirow_safe_acquire_steering_ok(row, shift_ok_for_follow):
                    return planner.plan_enter_row(row)
                return planner.plan_headland_shift_base_link(
                    speed=min(self.p.enter_speed, self.p.slow_speed),
                    reason="ENTER_ROW_WAIT_FULL_LANE",
                )

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
        p.turn_lookahead_distance = float(declare("turn_lookahead_distance", p.turn_lookahead_distance))
        p.turn_min_angular_speed = float(declare("turn_min_angular_speed", p.turn_min_angular_speed))
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

        p.headland_maneuver_enabled = bool(declare("headland_maneuver_enabled", p.headland_maneuver_enabled))
        p.headland_exit_straight_distance = float(declare("headland_exit_straight_distance", p.headland_exit_straight_distance))
        p.headland_exit_straight_speed = float(declare("headland_exit_straight_speed", p.headland_exit_straight_speed))
        p.exit_curve_speed = float(declare("exit_curve_speed", p.exit_curve_speed))
        p.exit_curve_angular_speed = float(declare("exit_curve_angular_speed", p.exit_curve_angular_speed))
        p.exit_curve_yaw_change = float(declare("exit_curve_yaw_change", p.exit_curve_yaw_change))
        p.headland_shift_speed = float(declare("headland_shift_speed", p.headland_shift_speed))
        p.headland_shift_tolerance = float(declare("headland_shift_tolerance", p.headland_shift_tolerance))
        p.headland_shift_overshoot_tolerance = float(declare("headland_shift_overshoot_tolerance", p.headland_shift_overshoot_tolerance))
        p.headland_yaw_tolerance = float(declare("headland_yaw_tolerance", p.headland_yaw_tolerance))
        p.headland_use_map_row_heading = bool(declare("headland_use_map_row_heading", p.headland_use_map_row_heading))
        p.headland_heading_kp = float(declare("headland_heading_kp", p.headland_heading_kp))
        p.headland_heading_max_yaw_error = float(declare("headland_heading_max_yaw_error", p.headland_heading_max_yaw_error))
        p.headland_map_heading_min_confidence = float(declare("headland_map_heading_min_confidence", p.headland_map_heading_min_confidence))
        p.entry_curve_speed = float(declare("entry_curve_speed", p.entry_curve_speed))
        p.entry_curve_angular_speed = float(declare("entry_curve_angular_speed", p.entry_curve_angular_speed))
        p.headland_total_yaw_change = float(declare("headland_total_yaw_change", p.headland_total_yaw_change))
        p.entry_curve_yaw_change = float(declare("entry_curve_yaw_change", p.entry_curve_yaw_change))
        p.entry_yaw_accept_tolerance = float(declare("entry_yaw_accept_tolerance", p.entry_yaw_accept_tolerance))
        p.entry_shift_accept_tolerance = float(declare("entry_shift_accept_tolerance", p.entry_shift_accept_tolerance))
        p.entry_row_min_confidence = float(declare("entry_row_min_confidence", p.entry_row_min_confidence))
        p.entry_row_stable_frames = int(declare("entry_row_stable_frames", p.entry_row_stable_frames))
        p.entry_require_full_lane = bool(declare("entry_require_full_lane", p.entry_require_full_lane))
        p.entry_center_b_tolerance = float(declare("entry_center_b_tolerance", p.entry_center_b_tolerance))
        p.entry_lane_width_tolerance = float(declare("entry_lane_width_tolerance", p.entry_lane_width_tolerance))
        p.entry_row_yaw_tolerance = float(declare("entry_row_yaw_tolerance", p.entry_row_yaw_tolerance))
        p.entry_relaxed_geometry_after_yaw_shift = bool(declare("entry_relaxed_geometry_after_yaw_shift", p.entry_relaxed_geometry_after_yaw_shift))
        p.entry_relaxed_center_b_tolerance = float(declare("entry_relaxed_center_b_tolerance", p.entry_relaxed_center_b_tolerance))
        p.entry_relaxed_row_yaw_tolerance = float(declare("entry_relaxed_row_yaw_tolerance", p.entry_relaxed_row_yaw_tolerance))
        p.multirow_entry_extra_shift = float(declare("multirow_entry_extra_shift", p.multirow_entry_extra_shift))
        p.multirow_entry_shift_overshoot_rows = float(declare("multirow_entry_shift_overshoot_rows", p.multirow_entry_shift_overshoot_rows))
        p.multirow_local_takeover_min_yaw_progress = float(declare("multirow_local_takeover_min_yaw_progress", p.multirow_local_takeover_min_yaw_progress))
        p.multirow_local_takeover_min_confidence = float(declare("multirow_local_takeover_min_confidence", p.multirow_local_takeover_min_confidence))
        p.map_counted_lane_inward_bias_tolerance = float(declare("map_counted_lane_inward_bias_tolerance", p.map_counted_lane_inward_bias_tolerance))
        p.neighbor_reference_turn_enabled = bool(declare("neighbor_reference_turn_enabled", p.neighbor_reference_turn_enabled))
        p.neighbor_reference_entry_requires_shift = bool(declare("neighbor_reference_entry_requires_shift", p.neighbor_reference_entry_requires_shift))
        p.neighbor_reference_requires_same_side_row = bool(declare("neighbor_reference_requires_same_side_row", p.neighbor_reference_requires_same_side_row))

        p.map_row_detection_enabled = bool(declare("map_row_detection_enabled", p.map_row_detection_enabled))
        p.map_row_occupancy_threshold = int(declare("map_row_occupancy_threshold", p.map_row_occupancy_threshold))
        p.map_row_search_x_forward = float(declare("map_row_search_x_forward", p.map_row_search_x_forward))
        p.map_row_search_x_backward = float(declare("map_row_search_x_backward", p.map_row_search_x_backward))
        p.map_row_search_y_side = float(declare("map_row_search_y_side", p.map_row_search_y_side))
        p.map_row_use_pca_orientation = bool(declare("map_row_use_pca_orientation", p.map_row_use_pca_orientation))
        p.map_row_pca_radius = float(declare("map_row_pca_radius", p.map_row_pca_radius))
        p.map_row_pca_min_points = int(declare("map_row_pca_min_points", p.map_row_pca_min_points))
        p.map_row_lateral_bin = float(declare("map_row_lateral_bin", p.map_row_lateral_bin))
        p.map_row_min_band_points = int(declare("map_row_min_band_points", p.map_row_min_band_points))
        p.map_row_min_band_length = float(declare("map_row_min_band_length", p.map_row_min_band_length))
        p.map_row_max_extrapolated_lanes = int(declare("map_row_max_extrapolated_lanes", p.map_row_max_extrapolated_lanes))
        p.map_row_line_ransac_iterations = int(declare("map_row_line_ransac_iterations", p.map_row_line_ransac_iterations))
        p.map_row_line_distance = float(declare("map_row_line_distance", p.map_row_line_distance))
        p.map_row_min_line_inliers = int(declare("map_row_min_line_inliers", p.map_row_min_line_inliers))
        p.map_row_min_line_length = float(declare("map_row_min_line_length", p.map_row_min_line_length))
        p.map_row_max_abs_line_slope = float(declare("map_row_max_abs_line_slope", p.map_row_max_abs_line_slope))
        p.map_row_max_lines = int(declare("map_row_max_lines", p.map_row_max_lines))
        p.map_row_line_merge_distance = float(declare("map_row_line_merge_distance", p.map_row_line_merge_distance))
        p.map_lane_accept_tolerance = float(declare("map_lane_accept_tolerance", p.map_lane_accept_tolerance))

        p.turn_replan_enabled = bool(declare("turn_replan_enabled", p.turn_replan_enabled))
        p.turn_replan_period_frames = int(declare("turn_replan_period_frames", p.turn_replan_period_frames))
        p.turn_replan_max_attempts = int(declare("turn_replan_max_attempts", p.turn_replan_max_attempts))
        p.turn_exit_on_local_row = bool(declare("turn_exit_on_local_row", p.turn_exit_on_local_row))
        p.turn_exit_min_confidence = float(declare("turn_exit_min_confidence", p.turn_exit_min_confidence))
        p.turn_exit_stable_frames = int(declare("turn_exit_stable_frames", p.turn_exit_stable_frames))

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

        if state in (MissionState.PLAN_TURN, MissionState.EXECUTE_TURN):
            reasons.append(
                f"target_offset_expected={self.mission.expected_target_offset:.3f}, "
                f"detected={self.mission.detected_target_offset:.3f}, "
                f"error={self.mission.target_offset_error:.3f}, "
                f"reason={self.mission.last_target_lane_reason}"
            )

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
            f" | pattern_index={self.mission.pattern_index}"
            f" | pattern_completed={self.mission.pattern_completed}"
            f" | headland_enabled={self.p.headland_maneuver_enabled}"
            f" | headland_required_shift={self.mission.headland_required_shift:.3f}"
            f" | headland_measured_shift={self.mission.headland_measured_shift:.3f}"
            f" | headland_exit_forward={self.mission.headland_exit_forward_distance:.3f}"
            f" | headland_pre_entry_target={self.mission._headland_pre_entry_shift_target():.3f}"
            f" | headland_total_yaw_progress={self.mission._headland_total_yaw_progress(pose_map):.3f}"
            f" | map_row_yaw={self.map_row_detector.last_row_yaw_map if self.map_row_detector.last_row_yaw_map is not None else 'None'}"
            f" | map_row_yaw_conf={self.map_row_detector.last_row_yaw_confidence:.3f}"
            f" | entry_row_stable_frames={self.mission.entry_row_stable_frames}"
            f" | expected_target_offset={self.mission.expected_target_offset:.3f}"
            f" | detected_target_offset={self.mission.detected_target_offset:.3f}"
            f" | target_offset_error={self.mission.target_offset_error:.3f}"
            f" | target_lane_reason='{self.mission.last_target_lane_reason}'"
            f" | reference_row_v={self.mission.reference_row_v:.3f}"
            f" | target_row_v={self.mission.target_row_v:.3f}"
            f" | candidate_rows_side='{self.mission.candidate_rows_text}'"
            f" | turn_local_row_stable_frames={self.mission.turn_local_row_stable_frames}"
            f" | turn_replan_attempts={self.mission.turn_replan_attempts}"
            f" | turn_replan_enabled={self.p.turn_replan_enabled}"
            f" | turn_exit_on_local_row={self.p.turn_exit_on_local_row}"
            f" | entry_shift_ok={self.mission.entry_shift_ok}"
            f" | entry_reference_side_ok={self.mission.entry_reference_side_ok}"
            f" | entry_geometry_ok={self.mission._entry_row_geometry_ok(row)}"
            f" | neighbor_reference_enabled={self.p.neighbor_reference_turn_enabled}"
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
