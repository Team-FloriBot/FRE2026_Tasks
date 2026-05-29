#!/usr/bin/env python3

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf_transformations import euler_from_quaternion


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_vec(yaw: float) -> Tuple[float, float]:
    return math.cos(yaw), math.sin(yaw)


def local_to_world(x: float, y: float, pose: "Pose2D", ref_yaw: float) -> Tuple[float, float]:
    c = math.cos(ref_yaw)
    s = math.sin(ref_yaw)
    return pose.x + c * x - s * y, pose.y + s * x + c * y


def world_to_local(x: float, y: float, pose: "Pose2D", ref_yaw: float) -> Tuple[float, float]:
    dx = x - pose.x
    dy = y - pose.y
    c = math.cos(ref_yaw)
    s = math.sin(ref_yaw)
    return c * dx + s * dy, -s * dx + c * dy


def quaternion_yaw(q) -> float:
    return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class PatternStep:
    count: int
    side: str


@dataclass
class RowCandidate:
    peak_v: float
    line_a: float
    line_b: float
    line_yaw: float
    inliers: int
    length: float
    confidence: float
    points_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    points_world: List[Point] = field(default_factory=list)


@dataclass
class RowTrack:
    row_id: int
    line_yaw: float
    line_point: Point
    confidence: float = 0.0
    local_offset: float = 0.0
    points: List[Point] = field(default_factory=list)


@dataclass
class CorridorModel:
    left_row_id: int
    right_row_id: int
    center_yaw: float
    center_point: Point
    center_points: List[Point] = field(default_factory=list)
    length: float = 0.0
    confidence: float = 0.0


class MissionState(Enum):
    IDLE = 0
    INITIALIZING = 1
    FOLLOW_ROW = 2
    EXIT_ROW = 3
    TURN_OUT = 4
    SHIFT = 5
    TURN_IN = 6
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
    row_search_max_dist: float = 2.0
    row_search_width: float = 0.50
    row_exclusion_distance: float = 0.30
    row_extension_max_spline_distance: float = 0.50
    row_window_length: float = 2.0
    row_window_step: float = 0.50
    row_window_beam_width: int = 6
    row_window_candidate_limit: int = 6
    row_window_support_weight: float = 3.2
    row_window_prediction_weight: float = 1.6
    row_window_smoothness_weight: float = 1.8
    row_window_gap_penalty: float = 1.4
    row_cluster_radius: float = 0.18
    row_end_front_point_ratio: float = 0.08
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
    turn_speed: float = 0.24

    max_linear_speed: float = 0.45
    max_angular_speed: float = 1.40
    follow_max_angular_speed: float = 0.90
    turn_max_angular_speed: float = 1.40
    angular_rate_limit: float = 2.8

    lookahead_distance: float = 0.75
    turn_lookahead_distance: float = 0.32
    turn_min_angular_speed: float = 0.30

    path_goal_xy_tolerance: float = 0.20
    path_goal_yaw_tolerance: float = 0.40

    exit_distance: float = 0.70
    turn_forward_distance: float = 2.20
    min_turn_radius: float = 0.38
    enter_distance: float = 0.90

    pattern: str = "1L 2R 1L 3R"

    row_shift_count: int = 1
    row_shift_direction: str = "L"
    turn_180: bool = True

    headland_maneuver_enabled: bool = True
    headland_exit_straight_distance: float = 0.45
    headland_exit_straight_speed: float = 0.18
    exit_curve_speed: float = 0.18
    exit_curve_angular_speed: float = 0.48
    exit_curve_yaw_change: float = 1.35

    headland_shift_speed: float = 0.22
    headland_shift_tolerance: float = 0.04
    headland_shift_overshoot_tolerance: float = 0.08
    headland_yaw_tolerance: float = 0.25
    headland_use_map_row_heading: bool = True
    headland_heading_kp: float = 1.4
    headland_heading_max_yaw_error: float = 0.75

    entry_curve_speed: float = 0.16
    entry_curve_angular_speed: float = 0.427
    headland_total_yaw_change: float = math.pi
    entry_curve_yaw_change: float = -1.0
    entry_yaw_accept_tolerance: float = 0.35
    entry_shift_accept_tolerance: float = 0.10
    entry_row_min_confidence: float = 0.32
    entry_row_stable_frames: int = 10
    entry_require_full_lane: bool = True
    entry_center_b_tolerance: float = 0.14
    entry_lane_width_tolerance: float = 0.22
    entry_row_yaw_tolerance: float = 0.35

    neighbor_reference_turn_enabled: bool = True
    neighbor_reference_entry_requires_shift: bool = True
    neighbor_reference_requires_same_side_row: bool = True

    map_row_detection_enabled: bool = True
    map_row_occupancy_threshold: int = 50
    map_row_search_x_forward: float = 5.0
    map_row_search_x_backward: float = 4.0
    map_row_search_y_side: float = 5.0
    map_row_use_pca_orientation: bool = True
    map_row_pca_radius: float = 5.0
    map_row_pca_min_points: int = 80
    map_row_lateral_bin: float = 0.10
    map_row_min_band_points: int = 12
    map_row_min_band_length: float = 1.2
    map_row_max_extrapolated_lanes: int = 3

    map_row_line_ransac_iterations: int = 180
    map_row_line_distance: float = 0.12
    map_row_min_line_inliers: int = 18
    map_row_min_line_length: float = 1.20
    map_row_max_abs_line_slope: float = 0.70
    map_row_max_lines: int = 12
    map_row_line_merge_distance: float = 0.22

    map_lane_accept_tolerance: float = 0.45

    turn_replan_enabled: bool = True
    turn_replan_period_frames: int = 5
    turn_replan_max_attempts: int = 60

    turn_exit_on_local_row: bool = True
    turn_exit_min_confidence: float = 0.32
    turn_exit_stable_frames: int = 10

    enable_safety: bool = False
    obstacle_stop_distance: float = 0.25
    obstacle_slow_distance: float = 0.45

    publish_debug: bool = True


class MaizeNavigator(Node):
    def __init__(self):
        super().__init__("maize_navigator")
        self.p = self.load_params()

        self.state = MissionState.IDLE
        self.pattern_steps = self.parse_pattern(self.p.pattern)
        self.current_pattern_idx = 0

        self.latest_map: Optional[OccupancyGrid] = None
        self.robot_pose: Optional[Pose2D] = None

        self.frame_counter = 0
        self.end_confirm_frames = 0
        self.turn_confirm_frames = 0
        self.exit_start_pose: Optional[Pose2D] = None
        self.turn_start_pose: Optional[Pose2D] = None
        self.row_axis_yaw: Optional[float] = None
        self.reference_row_yaw: Optional[float] = None
        self.current_corridor: Optional[CorridorModel] = None
        self.target_corridor_ids: Optional[Tuple[int, int]] = None

        self.known_rows: Dict[int, RowTrack] = {}
        self.row_ids_in_order: List[int] = []
        self.next_row_id = 1
        self.min_known_row_id = 1
        self.max_known_row_id = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(OccupancyGrid, self.p.map_topic, self.map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "navigation_markers", 10)

        self.start_srv = self.create_service(Trigger, "start_navigation", self.start_cb)
        self.stop_srv = self.create_service(Trigger, "stop_navigation", self.stop_cb)

        self.timer = self.create_timer(1.0 / self.p.control_frequency, self.control_loop)
        self.get_logger().info("Maize Navigator ready")

    def load_params(self) -> NavigatorParams:
        p = NavigatorParams()

        def get_param(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        for field_name in p.__dataclass_fields__:
            setattr(p, field_name, get_param(field_name, getattr(p, field_name)))

        return p

    def parse_pattern(self, pattern_str: str) -> List[PatternStep]:
        normalized = pattern_str.replace("-", " ").replace(",", " ").upper()
        tokens = [token for token in normalized.split() if token]
        steps: List[PatternStep] = []
        pending_count: Optional[int] = None

        for token in tokens:
            if token.isdigit():
                pending_count = int(token)
                continue

            if len(token) >= 2 and token[:-1].isdigit() and token[-1] in ("L", "R"):
                steps.append(PatternStep(max(1, int(token[:-1])), token[-1]))
                pending_count = None
                continue

            if token in ("L", "R") and pending_count is not None:
                steps.append(PatternStep(max(1, pending_count), token))
                pending_count = None
                continue

            self.get_logger().error(f"Invalid pattern token: {token}")
            return []

        if pending_count is not None:
            self.get_logger().error(f"Invalid pattern: dangling count in {pattern_str}")
            return []

        return steps

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def start_cb(self, req, res):
        self.state = MissionState.INITIALIZING
        self.current_pattern_idx = 0
        self.end_confirm_frames = 0
        self.turn_confirm_frames = 0
        self.exit_start_pose = None
        self.turn_start_pose = None
        self.current_corridor = None
        self.target_corridor_ids = None
        res.success = True
        res.message = "Navigation started"
        return res

    def stop_cb(self, req, res):
        self.state = MissionState.IDLE
        self.cmd_pub.publish(Twist())
        res.success = True
        res.message = "Navigation stopped"
        return res

    def get_robot_pose(self) -> Optional[Pose2D]:
        try:
            transform = self.tf_buffer.lookup_transform(self.p.map_frame, self.p.base_frame, rclpy.time.Time())
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return Pose2D(
                transform.transform.translation.x,
                transform.transform.translation.y,
                yaw,
            )
        except Exception:
            return None

    def control_loop(self):
        self.frame_counter += 1
        self.robot_pose = self.get_robot_pose()
        if self.robot_pose is None or self.latest_map is None:
            return

        if self.state == MissionState.IDLE:
            self.cmd_pub.publish(Twist())
            self.publish_visuals()
            return

        if self.state == MissionState.INITIALIZING:
            self.handle_initializing()
        elif self.state == MissionState.FOLLOW_ROW:
            self.handle_follow_row()
        elif self.state == MissionState.EXIT_ROW:
            self.handle_exit_row()
        elif self.state == MissionState.TURN_OUT:
            self.handle_turn_out()
        elif self.state == MissionState.SHIFT:
            self.handle_shift()
        elif self.state == MissionState.TURN_IN:
            self.handle_turn_in()

        self.publish_visuals()

    def _map_points_local(self, ref_yaw: float, *, use_acquire_roi: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid = self.latest_map
        pose = self.robot_pose
        if grid is None or pose is None:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0, 2), dtype=float)

        width = int(grid.info.width)
        height = int(grid.info.height)
        resolution = float(grid.info.resolution)
        if width <= 0 or height <= 0 or resolution <= 0.0:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0, 2), dtype=float)

        data = np.asarray(grid.data, dtype=np.int16).reshape((height, width))
        occ_r, occ_c = np.where(data >= int(self.p.map_row_occupancy_threshold))
        if len(occ_r) == 0:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), np.zeros((0, 2), dtype=float)

        if len(occ_r) > 35000:
            idx = np.linspace(0, len(occ_r) - 1, 35000).astype(int)
            occ_r = occ_r[idx]
            occ_c = occ_c[idx]

        origin = grid.info.origin
        origin_yaw = quaternion_yaw(origin.orientation)
        co = math.cos(origin_yaw)
        so = math.sin(origin_yaw)

        gx = (occ_c.astype(float) + 0.5) * resolution
        gy = (occ_r.astype(float) + 0.5) * resolution
        mx = origin.position.x + co * gx - so * gy
        my = origin.position.y + so * gx + co * gy

        u = np.zeros_like(mx)
        v = np.zeros_like(my)
        for i in range(len(mx)):
            u[i], v[i] = world_to_local(float(mx[i]), float(my[i]), pose, ref_yaw)

        if use_acquire_roi:
            mask = (
                (u >= self.p.acquire_roi_x_min)
                & (u <= self.p.acquire_roi_x_max)
                & (np.abs(v) >= self.p.acquire_roi_y_abs_min)
                & (np.abs(v) <= self.p.acquire_roi_y_abs_max)
            )
        else:
            mask = (
                (u >= self.p.roi_x_min)
                & (u <= self.p.roi_x_max)
                & (np.abs(v) >= self.p.roi_y_abs_min)
                & (np.abs(v) <= self.p.roi_y_abs_max)
            )

        local = np.stack([u[mask], v[mask]], axis=1)
        return u[mask], v[mask], local

    def _build_histogram(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        half_width = max(self.p.map_row_search_y_side, 0.5)
        bin_size = max(self.p.map_row_lateral_bin, 0.05)
        bins = np.arange(-half_width, half_width + bin_size, bin_size)
        hist, edges = np.histogram(v, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        if len(hist) >= 5:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
            kernel /= kernel.sum()
            hist = np.convolve(hist.astype(float), kernel, mode="same")
        return hist, centers

    def _find_hist_peaks(self, hist: np.ndarray, centers: np.ndarray) -> List[Tuple[float, float]]:
        if len(hist) < 3:
            return []

        peaks: List[Tuple[float, float]] = []
        for idx in range(1, len(hist) - 1):
            if hist[idx] >= hist[idx - 1] and hist[idx] >= hist[idx + 1] and hist[idx] >= self.p.map_row_min_band_points:
                peaks.append((float(centers[idx]), float(hist[idx])))

        peaks.sort(key=lambda item: item[1], reverse=True)
        return peaks[: max(1, int(self.p.row_window_candidate_limit))]

    def _fit_local_line(self, u: np.ndarray, v: np.ndarray) -> Tuple[bool, float, float, np.ndarray]:
        n = len(u)
        if n < max(2, int(self.p.map_row_min_line_inliers)):
            return False, 0.0, 0.0, np.zeros((0,), dtype=bool)

        rng = random.Random(42 + self.frame_counter)
        iterations = max(1, int(self.p.map_row_line_ransac_iterations))
        dist_th = max(0.02, float(self.p.map_row_line_distance))
        max_slope = max(0.05, float(self.p.map_row_max_abs_line_slope))
        indices = list(range(n))

        best_mask: Optional[np.ndarray] = None
        best_count = 0
        best_a = 0.0
        best_b = 0.0

        for _ in range(iterations):
            i1, i2 = rng.sample(indices, 2)
            du = float(u[i2] - u[i1])
            if abs(du) < 1e-4:
                continue
            a = float((v[i2] - v[i1]) / du)
            if abs(a) > max_slope:
                continue
            b = float(v[i1] - a * u[i1])
            denom = math.sqrt(a * a + 1.0)
            dist = np.abs(a * u - v + b) / denom
            mask = dist <= dist_th
            count = int(np.count_nonzero(mask))
            if count > best_count:
                best_count = count
                best_mask = mask
                best_a = a
                best_b = b

        if best_mask is None or best_count < int(self.p.map_row_min_line_inliers):
            return False, 0.0, 0.0, np.zeros((0,), dtype=bool)

        inlier_u = u[best_mask]
        inlier_v = v[best_mask]
        if len(inlier_u) < int(self.p.map_row_min_line_inliers):
            return False, 0.0, 0.0, np.zeros((0,), dtype=bool)

        try:
            a, b = np.polyfit(inlier_u, inlier_v, 1)
            a = float(a)
            b = float(b)
        except Exception:
            a = best_a
            b = best_b

        if abs(a) > max_slope:
            return False, 0.0, 0.0, np.zeros((0,), dtype=bool)

        length = float(np.max(inlier_u) - np.min(inlier_u)) if len(inlier_u) else 0.0
        if length < float(self.p.map_row_min_line_length):
            return False, 0.0, 0.0, np.zeros((0,), dtype=bool)

        return True, a, b, best_mask

    def _sample_line_world(
        self,
        pose: Pose2D,
        ref_yaw: float,
        a: float,
        b: float,
        start_u: float = 0.0,
        max_u: Optional[float] = None,
    ) -> Tuple[Point, float, List[Point], float]:
        line_yaw = wrap_to_pi(ref_yaw + math.atan(a))
        if max_u is None:
            max_u = max(self.p.row_window_length, self.p.row_search_max_dist)

        step = max(0.05, float(self.p.row_window_step))
        points: List[Point] = []
        s = start_u
        while s <= max_u + 1e-6:
            u = s
            v = a * u + b
            wx, wy = local_to_world(u, v, pose, ref_yaw)
            points.append(Point(x=float(wx), y=float(wy), z=0.0))
            s += step

        anchor_x, anchor_y = local_to_world(0.0, b, pose, ref_yaw)
        return Point(x=float(anchor_x), y=float(anchor_y), z=0.0), line_yaw, points, float(max_u)

    def _build_candidates(self, ref_yaw: float, use_acquire_roi: bool = False) -> List[RowCandidate]:
        if self.robot_pose is None:
            return []

        u, v, _local = self._map_points_local(ref_yaw, use_acquire_roi=use_acquire_roi)
        if len(u) == 0:
            return []

        hist, centers = self._build_histogram(v)
        peaks = self._find_hist_peaks(hist, centers)
        if not peaks:
            return []

        band_width = max(self.p.row_search_width, self.p.row_cluster_radius * 2.0)
        candidates: List[RowCandidate] = []
        used = np.zeros(len(u), dtype=bool)

        for peak_v, _strength in peaks:
            if len(candidates) >= int(self.p.row_window_candidate_limit):
                break

            band_mask = np.abs(v - peak_v) <= band_width
            band_mask &= ~used
            if np.count_nonzero(band_mask) < max(6, int(self.p.map_row_min_line_inliers)):
                continue

            band_u = u[band_mask]
            band_v = v[band_mask]
            valid, a, b, inlier_mask = self._fit_local_line(band_u, band_v)
            if not valid or len(inlier_mask) == 0:
                continue

            inlier_u = band_u[inlier_mask]
            inlier_v = band_v[inlier_mask]
            used_indices = np.where(band_mask)[0][inlier_mask]
            used[used_indices] = True

            length = float(np.max(inlier_u) - np.min(inlier_u)) if len(inlier_u) else 0.0
            confidence = clamp(
                0.5 * (len(inlier_u) / max(float(self.p.map_row_min_line_inliers), 1.0))
                + 0.5 * (length / max(float(self.p.map_row_min_line_length), 1e-3)),
                0.0,
                1.0,
            )

            line_point, line_yaw, sampled_points, _ = self._sample_line_world(self.robot_pose, ref_yaw, a, b)
            candidates.append(
                RowCandidate(
                    peak_v=float(peak_v),
                    line_a=float(a),
                    line_b=float(b),
                    line_yaw=float(line_yaw),
                    inliers=int(len(inlier_u)),
                    length=float(length),
                    confidence=float(confidence),
                    points_local=np.stack([inlier_u, inlier_v], axis=1),
                    points_world=sampled_points if sampled_points else [line_point],
                )
            )

        candidates.sort(key=lambda cand: cand.peak_v)
        return candidates

    def _track_local_offset(self, row: RowTrack, ref_yaw: float) -> float:
        pose = self.robot_pose
        if pose is None:
            return row.local_offset

        source_points = row.points if row.points else [row.line_point]
        offsets = []
        for pt in source_points[: min(5, len(source_points))]:
            _, v = world_to_local(pt.x, pt.y, pose, ref_yaw)
            offsets.append(v)
        if not offsets:
            return row.local_offset
        return float(np.median(offsets))

    def _allocate_row_id_above(self) -> int:
        row_id = self.max_known_row_id + 1 if self.known_rows else 1
        while row_id in self.known_rows:
            row_id += 1
        self.max_known_row_id = max(self.max_known_row_id, row_id)
        self.next_row_id = max(self.next_row_id, row_id + 1)
        return row_id

    def _allocate_row_id_below(self) -> int:
        row_id = self.min_known_row_id - 1 if self.known_rows else 1
        while row_id in self.known_rows:
            row_id -= 1
        self.min_known_row_id = min(self.min_known_row_id, row_id)
        return row_id

    def _store_row(self, row_id: int, cand: RowCandidate):
        if not cand.points_world:
            return

        anchor = cand.points_world[0]
        if row_id in self.known_rows:
            existing = self.known_rows[row_id]
            alpha = clamp(float(self.p.tracker_alpha), 0.01, 1.0)
            existing.line_yaw = wrap_to_pi((1.0 - alpha) * existing.line_yaw + alpha * cand.line_yaw)
            existing.line_point = Point(
                x=float((1.0 - alpha) * existing.line_point.x + alpha * anchor.x),
                y=float((1.0 - alpha) * existing.line_point.y + alpha * anchor.y),
                z=0.0,
            )
            existing.confidence = clamp((1.0 - float(self.p.confidence_decay)) * existing.confidence + float(self.p.confidence_decay) * cand.confidence, 0.0, 1.0)
            existing.local_offset = cand.peak_v
            existing.points = list(cand.points_world)
        else:
            self.known_rows[row_id] = RowTrack(
                row_id=row_id,
                line_yaw=cand.line_yaw,
                line_point=Point(x=float(anchor.x), y=float(anchor.y), z=0.0),
                confidence=float(cand.confidence),
                local_offset=float(cand.peak_v),
                points=list(cand.points_world),
            )

        self.min_known_row_id = min(self.min_known_row_id, row_id)
        self.max_known_row_id = max(self.max_known_row_id, row_id)

    def _rebuild_row_order(self):
        self.row_ids_in_order = sorted(self.known_rows.keys(), key=lambda rid: self.known_rows[rid].local_offset)

    def _match_candidates_to_rows(self, candidates: List[RowCandidate], ref_yaw: float) -> List[int]:
        assigned_ids: List[int] = []
        if not candidates:
            return assigned_ids

        if not self.known_rows:
            for cand in candidates:
                row_id = self._allocate_row_id_above()
                self._store_row(row_id, cand)
                assigned_ids.append(row_id)
            self._rebuild_row_order()
            return assigned_ids

        matched: Dict[int, int] = {}
        for cand_idx, cand in enumerate(candidates):
            best_id = None
            best_error = float("inf")
            for row_id, row in self.known_rows.items():
                error = abs(cand.peak_v - self._track_local_offset(row, ref_yaw))
                if error < best_error:
                    best_error = error
                    best_id = row_id
            if best_id is not None and best_error <= float(self.p.map_lane_accept_tolerance):
                matched[cand_idx] = best_id

        if not matched:
            for cand in candidates:
                row_id = self._allocate_row_id_above()
                self._store_row(row_id, cand)
                assigned_ids.append(row_id)
            self._rebuild_row_order()
            return assigned_ids

        matched_indices = sorted(matched.keys())
        left_anchor = matched_indices[0]
        right_anchor = matched_indices[-1]

        for idx, cand in enumerate(candidates):
            if idx in matched:
                row_id = matched[idx]
                self._store_row(row_id, cand)
                assigned_ids.append(row_id)
                continue

            if idx < left_anchor:
                row_id = self._allocate_row_id_below()
            elif idx > right_anchor:
                row_id = self._allocate_row_id_above()
            else:
                lower = [matched[i] for i in matched_indices if i < idx]
                upper = [matched[i] for i in matched_indices if i > idx]
                if lower and upper:
                    row_id = lower[-1] + 1
                    if row_id in self.known_rows:
                        row_id = upper[0] - 1
                elif lower:
                    row_id = lower[-1] + 1
                elif upper:
                    row_id = upper[0] - 1
                else:
                    row_id = self._allocate_row_id_above()

            self._store_row(row_id, cand)
            assigned_ids.append(row_id)

        self._rebuild_row_order()
        return assigned_ids

    def _choose_start_pair(self, candidates: List[RowCandidate]) -> Optional[Tuple[int, int]]:
        if len(candidates) < 2:
            return None

        left_idx = None
        right_idx = None
        for idx, cand in enumerate(candidates):
            if cand.peak_v < 0.0:
                left_idx = idx
            elif cand.peak_v > 0.0 and right_idx is None:
                right_idx = idx

        if left_idx is not None and right_idx is not None:
            return left_idx, right_idx

        ordered = sorted(range(len(candidates)), key=lambda idx: abs(candidates[idx].peak_v))
        return ordered[0], ordered[1]

    def _build_corridor(self, left_id: int, right_id: int) -> Optional[CorridorModel]:
        left = self.known_rows.get(left_id)
        right = self.known_rows.get(right_id)
        if left is None or right is None:
            return None

        center_yaw = wrap_to_pi(0.5 * (left.line_yaw + right.line_yaw))
        center_point = Point(
            x=float(0.5 * (left.line_point.x + right.line_point.x)),
            y=float(0.5 * (left.line_point.y + right.line_point.y)),
            z=0.0,
        )

        center_points: List[Point] = []
        sample_count = min(len(left.points) if left.points else 1, len(right.points) if right.points else 1)
        sample_count = max(1, sample_count)
        for i in range(sample_count):
            lp = left.points[min(i, len(left.points) - 1)] if left.points else left.line_point
            rp = right.points[min(i, len(right.points) - 1)] if right.points else right.line_point
            center_points.append(Point(x=float(0.5 * (lp.x + rp.x)), y=float(0.5 * (lp.y + rp.y)), z=0.0))

        return CorridorModel(
            left_row_id=left_id,
            right_row_id=right_id,
            center_yaw=center_yaw,
            center_point=center_point,
            center_points=center_points,
            length=float(len(center_points)) * float(self.p.row_window_step),
            confidence=float(min(left.confidence, right.confidence)),
        )

    def _pattern_shift(self, count: int, side: str) -> int:
        if self.current_corridor is None or self.robot_pose is None:
            return count if side == "L" else -count

        delta = wrap_to_pi(self.robot_pose.yaw - self.current_corridor.center_yaw)
        orientation_sign = 1 if math.cos(delta) >= 0.0 else -1
        side_sign = 1 if side == "L" else -1
        return orientation_sign * side_sign * max(1, int(count))

    def _select_target_pair(self) -> Optional[Tuple[int, int]]:
        if self.current_corridor is None or not self.pattern_steps:
            return None

        step = self.pattern_steps[min(self.current_pattern_idx, len(self.pattern_steps) - 1)]
        shift = self._pattern_shift(step.count, step.side)
        return self.current_corridor.left_row_id + shift, self.current_corridor.right_row_id + shift

    def _front_density(self, corridor: CorridorModel) -> int:
        pose = self.robot_pose
        if pose is None:
            return 0

        count = 0
        for row_id in (corridor.left_row_id, corridor.right_row_id):
            row = self.known_rows.get(row_id)
            if row is None:
                continue
            points = row.points if row.points else [row.line_point]
            for pt in points:
                u, v = world_to_local(pt.x, pt.y, pose, corridor.center_yaw)
                if self.p.front_density_x_min <= u <= self.p.front_density_x_max and abs(v) <= self.p.front_density_y_abs:
                    count += 1
        return count

    def _follow_twist_from_corridor(self, corridor: CorridorModel) -> Twist:
        pose = self.robot_pose
        if pose is None:
            return Twist()

        cx, cy = corridor.center_point.x, corridor.center_point.y
        lookahead = max(float(self.p.lookahead_distance), 0.3)
        cux, cuy = yaw_to_vec(corridor.center_yaw)
        proj_u = (pose.x - cx) * cux + (pose.y - cy) * cuy
        target_u = proj_u + lookahead
        tx = cx + target_u * cux
        ty = cy + target_u * cuy

        heading_error = wrap_to_pi(math.atan2(ty - pose.y, tx - pose.x) - pose.yaw)
        linear = clamp(self.p.follow_speed * max(0.35, 1.0 - abs(heading_error) / 1.2), 0.0, self.p.max_linear_speed)
        angular = clamp(2.0 * linear * math.sin(heading_error) / max(lookahead, 1e-3), -self.p.follow_max_angular_speed, self.p.follow_max_angular_speed)

        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        return cmd

    def _turn_twist_to_yaw(self, target_yaw: float, gain: float) -> Tuple[Twist, bool]:
        pose = self.robot_pose
        if pose is None:
            return Twist(), False

        error = wrap_to_pi(target_yaw - pose.yaw)
        if abs(error) <= float(self.p.path_goal_yaw_tolerance):
            return Twist(), True

        cmd = Twist()
        angular = self.p.turn_min_angular_speed + abs(error) * gain
        cmd.angular.z = float(clamp(angular, -self.p.turn_max_angular_speed, self.p.turn_max_angular_speed))
        if error < 0.0:
            cmd.angular.z *= -1.0
        return cmd, False

    def _corridor_is_ending(self, corridor: CorridorModel) -> bool:
        density = self._front_density(corridor)
        if density < int(self.p.front_density_threshold):
            self.end_confirm_frames += 1
        else:
            self.end_confirm_frames = 0
        return self.end_confirm_frames >= int(self.p.end_stable_frames_required)

    def handle_initializing(self):
        pose = self.robot_pose
        if pose is None:
            return

        ref_yaw = self.row_axis_yaw if self.row_axis_yaw is not None else pose.yaw
        candidates = self._build_candidates(ref_yaw, use_acquire_roi=True)
        if len(candidates) < 2:
            self.cmd_pub.publish(Twist())
            return

        if self.reference_row_yaw is None:
            self.reference_row_yaw = ref_yaw
        self.row_axis_yaw = self.reference_row_yaw

        assigned_ids = self._match_candidates_to_rows(candidates, self.reference_row_yaw)
        self._rebuild_row_order()

        pair = self._choose_start_pair(candidates)
        if pair is None:
            self.cmd_pub.publish(Twist())
            return

        left_idx, right_idx = pair
        ordered_ids = assigned_ids if assigned_ids else sorted(self.known_rows.keys(), key=lambda rid: self.known_rows[rid].local_offset)
        if left_idx < len(ordered_ids) and right_idx < len(ordered_ids):
            left_id = ordered_ids[left_idx]
            right_id = ordered_ids[right_idx]
        else:
            left_id = ordered_ids[0]
            right_id = ordered_ids[1]

        self.current_corridor = self._build_corridor(left_id, right_id)
        if self.current_corridor is None:
            self.cmd_pub.publish(Twist())
            return

        self.state = MissionState.FOLLOW_ROW
        self.get_logger().info(f"Start corridor: {left_id} / {right_id}")

    def handle_follow_row(self):
        if self.current_corridor is None:
            self.state = MissionState.INITIALIZING
            return

        ref_yaw = self.current_corridor.center_yaw
        candidates = self._build_candidates(ref_yaw, use_acquire_roi=False)
        if candidates:
            self._match_candidates_to_rows(candidates, ref_yaw)
            self._rebuild_row_order()
            updated = self._build_corridor(self.current_corridor.left_row_id, self.current_corridor.right_row_id)
            if updated is not None:
                self.current_corridor = updated

        if self._corridor_is_ending(self.current_corridor):
            self.exit_start_pose = self.robot_pose
            self.state = MissionState.EXIT_ROW
            self.end_confirm_frames = 0
            return

        self.cmd_pub.publish(self._follow_twist_from_corridor(self.current_corridor))

    def handle_exit_row(self):
        if self.current_corridor is None or self.robot_pose is None:
            self.cmd_pub.publish(Twist())
            self.state = MissionState.INITIALIZING
            return

        if self.exit_start_pose is None:
            self.exit_start_pose = self.robot_pose

        dx = self.robot_pose.x - self.exit_start_pose.x
        dy = self.robot_pose.y - self.exit_start_pose.y
        travel = dx * math.cos(self.current_corridor.center_yaw) + dy * math.sin(self.current_corridor.center_yaw)

        if travel >= float(self.p.headland_exit_straight_distance):
            self.state = MissionState.TURN_OUT
            self.turn_start_pose = self.robot_pose
            self.cmd_pub.publish(Twist())
            return

        cmd = Twist()
        cmd.linear.x = float(self.p.headland_exit_straight_speed)
        self.cmd_pub.publish(cmd)

    def handle_turn_out(self):
        self.cmd_pub.publish(Twist())
        if self.target_corridor_ids is None:
            self.target_corridor_ids = self._select_target_pair()
        self.state = MissionState.SHIFT

    def handle_shift(self):
        if self.robot_pose is None or self.current_corridor is None:
            self.cmd_pub.publish(Twist())
            return

        ref_yaw = self.current_corridor.center_yaw
        candidates = self._build_candidates(ref_yaw, use_acquire_roi=True)
        if candidates:
            self._match_candidates_to_rows(candidates, ref_yaw)
            self._rebuild_row_order()

        target = self.target_corridor_ids or self._select_target_pair()
        if target is None:
            self.cmd_pub.publish(Twist())
            return

        left_id, right_id = target
        if left_id in self.known_rows and right_id in self.known_rows:
            corridor = self._build_corridor(left_id, right_id)
            if corridor is not None:
                self.current_corridor = corridor
                self.state = MissionState.TURN_IN
                self.turn_start_pose = self.robot_pose
                self.cmd_pub.publish(Twist())
                return

        self.cmd_pub.publish(Twist())

    def handle_turn_in(self):
        if self.current_corridor is None:
            self.state = MissionState.INITIALIZING
            return

        target_yaw = wrap_to_pi(self.current_corridor.center_yaw + math.pi)
        cmd, done = self._turn_twist_to_yaw(target_yaw, float(self.p.entry_curve_angular_speed))
        if done:
            if self.current_pattern_idx < max(0, len(self.pattern_steps) - 1):
                self.current_pattern_idx += 1
                self.state = MissionState.FOLLOW_ROW
            else:
                self.state = MissionState.FINISHED
            self.cmd_pub.publish(Twist())
            return

        if self.p.turn_180:
            self.cmd_pub.publish(cmd)
        else:
            self.cmd_pub.publish(Twist())
            self.state = MissionState.FOLLOW_ROW

    def publish_visuals(self):
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()

        marker_id = 0
        for row_id in sorted(self.known_rows.keys()):
            row = self.known_rows[row_id]
            if not row.points:
                continue

            line = Marker()
            line.header = Header(frame_id=self.p.map_frame, stamp=now)
            line.ns = "rows"
            line.id = marker_id
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.06
            line.color.a = 1.0
            hue = (abs(row_id) * 0.17) % 1.0
            line.color.r = float(0.4 + 0.6 * math.sin(6.28318 * hue) ** 2)
            line.color.g = float(0.4 + 0.6 * math.sin(6.28318 * (hue + 0.33)) ** 2)
            line.color.b = float(0.4 + 0.6 * math.sin(6.28318 * (hue + 0.66)) ** 2)
            line.points = list(row.points)
            msg.markers.append(line)

            text = Marker()
            text.header = Header(frame_id=self.p.map_frame, stamp=now)
            text.ns = "row_ids"
            text.id = 1000 + marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = row.points[0]
            text.pose.position.z = 0.25
            text.scale.z = 0.25
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.text = str(row_id)
            msg.markers.append(text)

            marker_id += 1

        if self.current_corridor is not None:
            center = Marker()
            center.header = Header(frame_id=self.p.map_frame, stamp=now)
            center.ns = "centerline"
            center.id = 2000
            center.type = Marker.LINE_STRIP
            center.action = Marker.ADD
            center.scale.x = 0.08
            center.color.a = 1.0
            center.color.r = 0.1
            center.color.g = 1.0
            center.color.b = 0.2
            center.points = list(self.current_corridor.center_points)
            msg.markers.append(center)

        self.marker_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MaizeNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()