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

from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler


# --- Utils ---

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


# --- Data Structures for Sliding Window Paths ---

class PlantSpline:
    def __init__(
        self,
        points: np.ndarray,
        heading_yaw: float = 0.0,
        window_length: float = 0.65,
        window_step: float = 0.28,
        beam_width: int = 4,
        candidate_limit: int = 5,
        support_radius: float = 0.16,
        cluster_radius: float = 0.18,
        support_weight: float = 2.8,
        prediction_weight: float = 2.2,
        smoothness_weight: float = 1.1,
        gap_penalty: float = 0.9,
        logger: Optional[Node] = None,
    ):
        self.valid = False
        self.t_min = 0.0
        self.t_max = 0.0
        self.linear_mode = False
        self.points = np.asarray(points, dtype=float)
        self.main_dir = None
        self.perp_dir = None
        self.mean = None
        self.window_length = float(window_length)
        self.window_step = float(window_step)
        self.beam_width = int(beam_width)
        self.candidate_limit = int(candidate_limit)
        self.support_radius = float(support_radius)
        self.cluster_radius = float(cluster_radius)
        self.support_weight = float(support_weight)
        self.prediction_weight = float(prediction_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.gap_penalty = float(gap_penalty)
        self.anchor_t = np.array([], dtype=float)
        self.anchor_lateral = np.array([], dtype=float)
        self.anchor_world = np.empty((0, 2), dtype=float)
        self.segment_bounds: List[Tuple[float, float]] = []
        self.segment_splines: List[object] = []
        self.lateral_spline = None
        # Debug: store last window candidates and rejected hypotheses for visualization
        self._last_rejected: List[dict] = []
        self._last_accepted: List[dict] = []

        if len(self.points) < 4:
            return

        self.mean = np.mean(self.points, axis=0)
        centered = self.points - self.mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        self.main_dir = eigenvectors[:, np.argmax(eigenvalues)]
        self.perp_dir = np.array([-self.main_dir[1], self.main_dir[0]])

        heading_vec = np.array([math.cos(heading_yaw), math.sin(heading_yaw)])
        if np.dot(self.main_dir, heading_vec) < 0:
            self.main_dir = -self.main_dir
            self.perp_dir = -self.perp_dir

        projections = centered @ self.main_dir
        lateral = centered @ self.perp_dir
        sort_idx = np.argsort(projections)
        self.points = self.points[sort_idx]
        self.t = projections[sort_idx]
        self.lateral = lateral[sort_idx]

        unique_t, unique_idx = np.unique(self.t, return_index=True)
        if len(unique_t) < 4:
            return

        self.t_min = float(unique_t[0])
        self.t_max = float(unique_t[-1])

        if logger:
            logger.info(f"Building sliding-window row model with {len(unique_t)} unique points, t range: [{self.t_min:.2f}, {self.t_max:.2f}]")

        unique_lateral = self.lateral[unique_idx]
        if len(unique_t) < 6:
            self._build_linear_fallback(logger)
            return

        if not self._build_window_path(unique_t, unique_lateral, heading_yaw=heading_yaw, logger=logger):
            self._build_linear_fallback(logger)

    def _spawn_extension(self) -> "PlantSpline":
        clone = object.__new__(PlantSpline)
        clone.valid = self.valid
        clone.t_min = self.t_min
        clone.t_max = self.t_max
        clone.linear_mode = self.linear_mode
        clone.points = np.array(self.points, copy=True)
        clone.main_dir = np.array(self.main_dir, copy=True) if self.main_dir is not None else None
        clone.perp_dir = np.array(self.perp_dir, copy=True) if self.perp_dir is not None else None
        clone.mean = np.array(self.mean, copy=True) if self.mean is not None else None
        clone.segment_splines = list(self.segment_splines)
        clone.segment_bounds = list(self.segment_bounds)
        clone.lateral_spline = self.lateral_spline
        if self.linear_mode:
            clone.line_start = np.array(self.line_start, copy=True)
            clone.line_end = np.array(self.line_end, copy=True)
        if hasattr(self, "t"):
            clone.t = np.array(self.t, copy=True)
        if hasattr(self, "lateral"):
            clone.lateral = np.array(self.lateral, copy=True)
        return clone

    def extend_with_points(
        self,
        new_points: np.ndarray,
        heading_yaw: float = 0.0,
        window_length: float = 0.65,
        window_step: float = 0.28,
        beam_width: int = 4,
        candidate_limit: int = 5,
        support_radius: float = 0.16,
        cluster_radius: float = 0.18,
        support_weight: float = 2.8,
        prediction_weight: float = 2.2,
        smoothness_weight: float = 1.1,
        gap_penalty: float = 0.9,
        logger: Optional[Node] = None,
    ) -> "PlantSpline":
        if new_points is None or len(new_points) == 0:
            return self
        if self.points is None or len(self.points) == 0:
            combined = np.asarray(new_points, dtype=float)
        else:
            combined = np.vstack((self.points, np.asarray(new_points, dtype=float)))
        return PlantSpline(
            combined,
            heading_yaw=heading_yaw,
            window_length=window_length,
            window_step=window_step,
            beam_width=beam_width,
            candidate_limit=candidate_limit,
            support_radius=support_radius,
            cluster_radius=cluster_radius,
            support_weight=support_weight,
            prediction_weight=prediction_weight,
            smoothness_weight=smoothness_weight,
            gap_penalty=gap_penalty,
            logger=logger,
        )

    def _build_linear_fallback(self, logger: Optional[Node] = None) -> None:
        self.linear_mode = True
        self.valid = True
        self.line_start = self.mean + self.t_min * self.main_dir
        self.line_end = self.mean + self.t_max * self.main_dir
        self.anchor_t = np.array([self.t_min, self.t_max], dtype=float)
        self.anchor_lateral = np.array([
            float((self.line_start - self.mean) @ self.perp_dir),
            float((self.line_end - self.mean) @ self.perp_dir),
        ], dtype=float)
        self.anchor_world = np.array([self.line_start, self.line_end], dtype=float)
        self.segment_bounds = [(float(self.t_min), float(self.t_max))]
        self.segment_splines = []
        self.lateral_spline = None
        if logger:
            logger.info(f"Using linear fallback (n={len(self.points)})")

    def _build_window_path(self, fit_t: np.ndarray, fit_lateral: np.ndarray, heading_yaw: float = 0.0, logger: Optional[Node] = None) -> bool:
        forward_max_dist = max(0.5, float(self.window_length))
        step_distance = max(0.25, float(self.window_step))
        orthogonal_max_dist = max(0.20, float(self.support_radius))
        cluster_radius = max(0.10, float(self.cluster_radius))

        if len(self.points) < 2:
            return False

        self._last_rejected = []
        self._last_accepted = []

        def normalize(vec: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                return np.array([1.0, 0.0], dtype=float)
            return vec / norm

        def cluster_points(candidate_points: np.ndarray) -> List[np.ndarray]:
            if len(candidate_points) == 0:
                return []
            visited = np.zeros(len(candidate_points), dtype=bool)
            clusters: List[np.ndarray] = []
            for seed_idx in range(len(candidate_points)):
                if visited[seed_idx]:
                    continue
                stack = [seed_idx]
                visited[seed_idx] = True
                member_indices = []
                while stack:
                    idx = stack.pop()
                    member_indices.append(idx)
                    deltas = candidate_points - candidate_points[idx]
                    neighbor_mask = np.linalg.norm(deltas, axis=1) <= cluster_radius
                    neighbors = np.where(neighbor_mask & (~visited))[0]
                    for neighbor_idx in neighbors:
                        visited[neighbor_idx] = True
                        stack.append(int(neighbor_idx))
                cluster = candidate_points[np.array(member_indices, dtype=int)]
                if len(cluster) >= 2:
                    clusters.append(cluster)
            return clusters

        def fit_cluster_direction(cluster: np.ndarray) -> np.ndarray:
            if len(cluster) < 2:
                return self.main_dir
            centered = cluster - np.mean(cluster, axis=0)
            cov_local = np.cov(centered.T)
            eigenvalues_local, eigenvectors_local = np.linalg.eig(cov_local)
            return normalize(eigenvectors_local[:, int(np.argmax(eigenvalues_local))])

        start_anchor = np.mean(self.points, axis=0)
        current_anchor = np.array(start_anchor, copy=True)
        current_dir = normalize(np.array(self.main_dir, copy=True))
        if np.dot(current_dir, np.array([math.cos(heading_yaw), math.sin(heading_yaw)])) < 0:
            current_dir = -current_dir

        anchors_world = [current_anchor]
        anchors_t = [0.0]
        anchors_l = [float((current_anchor - self.mean) @ self.perp_dir)]

        remaining_points = np.array(self.points, copy=True)
        max_steps = max(1, int(math.ceil(forward_max_dist / step_distance)) + 2)

        for _step_idx in range(max_steps):
            rel = remaining_points - current_anchor
            along = rel @ current_dir
            perp = rel @ np.array([-current_dir[1], current_dir[0]])
            candidate_mask = (along >= 0.0) & (along <= forward_max_dist) & (np.abs(perp) <= orthogonal_max_dist)
            candidate_points = remaining_points[candidate_mask]
            if len(candidate_points) == 0:
                break

            clusters = cluster_points(candidate_points)
            if not clusters:
                break

            best_cluster = None
            best_score = -float("inf")
            for cluster in clusters:
                centroid = np.mean(cluster, axis=0)
                rel_centroid = centroid - current_anchor
                along_c = float(rel_centroid @ current_dir)
                perp_c = float(rel_centroid @ np.array([-current_dir[1], current_dir[0]]))
                score = float(len(cluster))
                score -= 1.2 * abs(along_c - step_distance)
                score -= 0.6 * abs(perp_c)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if best_cluster is None or best_score < 1.0:
                break

            centroid = np.mean(best_cluster, axis=0)
            step_vec = centroid - current_anchor
            step_len = float(np.linalg.norm(step_vec))
            if step_len < 0.12:
                break

            current_dir = normalize(0.6 * current_dir + 0.4 * (step_vec / step_len))
            current_anchor = centroid
            anchors_world.append(np.array(current_anchor, copy=True))
            anchors_t.append(anchors_t[-1] + step_len)
            anchors_l.append(float((current_anchor - self.mean) @ self.perp_dir))

            keep_mask = np.ones(len(remaining_points), dtype=bool)
            for cluster_point in best_cluster:
                keep_mask &= np.linalg.norm(remaining_points - cluster_point, axis=1) > cluster_radius
            remaining_points = remaining_points[keep_mask]
            if len(remaining_points) == 0:
                break

        if len(anchors_world) < 2:
            return False

        self.anchor_world = np.asarray(anchors_world, dtype=float)
        self.anchor_t = np.asarray(anchors_t, dtype=float)
        self.anchor_lateral = np.asarray(anchors_l, dtype=float)

        self.segment_bounds = [(float(self.anchor_t[i]), float(self.anchor_t[i + 1])) for i in range(len(self.anchor_t) - 1)]
        self.segment_splines = []
        self.lateral_spline = None
        self.valid = True
        if logger:
            logger.info(f"Built clustered row with {len(self.anchor_t)} anchors and {len(self.segment_bounds)} segments")
        return True

    def _interpolate_lateral(self, t: float) -> float:
        if len(self.anchor_t) == 0:
            return 0.0
        if len(self.anchor_t) == 1:
            return float(self.anchor_lateral[0])
        t_clamped = float(np.clip(t, self.anchor_t[0], self.anchor_t[-1]))
        return float(np.interp(t_clamped, self.anchor_t, self.anchor_lateral))

    def _project_to_polyline(self, point: np.ndarray) -> float:
        if len(self.anchor_world) < 2:
            centered = point - self.mean
            return float(centered @ self.main_dir)

        best_t = float(self.anchor_t[0])
        best_dist2 = float("inf")
        for i in range(len(self.anchor_world) - 1):
            a = self.anchor_world[i]
            b = self.anchor_world[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                continue
            u = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
            proj = a + u * ab
            dist2 = float(np.sum((point - proj) ** 2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_t = float(self.anchor_t[i] + u * (self.anchor_t[i + 1] - self.anchor_t[i]))
        return best_t

    def _local_direction(self, t: float) -> np.ndarray:
        if len(self.anchor_t) < 2:
            return self.main_dir
        idx = int(np.searchsorted(self.anchor_t, t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.anchor_t) - 2))
        dt = float(self.anchor_t[idx + 1] - self.anchor_t[idx])
        slope = 0.0 if abs(dt) < 1e-6 else float((self.anchor_lateral[idx + 1] - self.anchor_lateral[idx]) / dt)
        return self.main_dir + slope * self.perp_dir

    def _segment_candidates(self, t: float) -> List[int]:
        if len(self.anchor_t) < 2:
            return []
        candidates = [i for i, (t_start, t_end) in enumerate(self.segment_bounds) if t_start <= t <= t_end]
        if candidates:
            return candidates
        centers = [0.5 * (t_start + t_end) for t_start, t_end in self.segment_bounds]
        return [int(np.argmin([abs(t - c) for c in centers]))]

    def _evaluate_lateral(self, t: float) -> float:
        return self._interpolate_lateral(t)

    def _evaluate_lateral_derivative(self, t: float) -> float:
        if len(self.anchor_t) < 2:
            return 0.0
        direction = self._local_direction(t)
        denom = float(np.dot(self.perp_dir, self.perp_dir))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(direction - self.main_dir, self.perp_dir) / denom)

    def evaluate(self, t: float) -> Tuple[float, float]:
        if self.linear_mode and hasattr(self, "line_start") and hasattr(self, "line_end"):
            if self.t_max <= self.t_min:
                p = self.line_start
            else:
                alpha = (t - self.t_min) / (self.t_max - self.t_min)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                p = self.line_start + alpha * (self.line_end - self.line_start)
            return float(p[0]), float(p[1])
        lateral = self._interpolate_lateral(t)
        p = self.mean + t * self.main_dir + lateral * self.perp_dir
        return float(p[0]), float(p[1])

    def get_direction(self, t: float) -> float:
        if self.linear_mode and hasattr(self, "line_start") and hasattr(self, "line_end"):
            direction = self.line_end - self.line_start
            return math.atan2(direction[1], direction[0])
        direction = self._local_direction(t)
        dx, dy = float(direction[0]), float(direction[1])
        return math.atan2(dy, dx)

    def get_global_direction(self) -> float:
        if len(self.anchor_world) >= 2:
            direction = self.anchor_world[-1] - self.anchor_world[0]
            return float(math.atan2(direction[1], direction[0]))
        return float(math.atan2(self.main_dir[1], self.main_dir[0]))

    def get_points(self, num: int = 50) -> np.ndarray:
        if len(self.anchor_t) < 2:
            if self.linear_mode and hasattr(self, "line_start") and hasattr(self, "line_end"):
                t_vals = np.linspace(self.t_min, self.t_max, num)
                return np.array([self.evaluate(t) for t in t_vals])
            return np.empty((0, 2), dtype=float)
        t_vals = np.linspace(float(self.anchor_t[0]), float(self.anchor_t[-1]), num)
        if self.linear_mode and hasattr(self, "line_start") and hasattr(self, "line_end"):
            pts = [self.evaluate(t) for t in t_vals]
            return np.array(pts)
        return np.array([self.evaluate(t) for t in t_vals])
    
    def project(self, point: np.ndarray) -> float:
        return self._project_to_polyline(point)


# --- Mission States ---

class MissionState(Enum):
    IDLE = 0
    INITIALIZING = 1      # Create histogram, find start rows
    FOLLOW_ROW = 2       # Drive on mid-spline
    EXIT_ROW = 3         # Drive past row end
    TURN_OUT = 4         # 90 deg curve away from row
    SHIFT = 5            # Lateral shift along headland
    TURN_IN = 6          # 90 deg curve into new row
    FINISHED = 7


@dataclass
class NavigatorParams:
    scan_topic: str = "/sensors/merged_scan"
    cmd_vel_topic: str = "/cmd_vel"
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    map_topic: str = "/map"
    map_frame: str = "map"
    
    control_frequency: float = 30.0
    expected_row_width: float = 0.75
    
    # Histogram params
    hist_roi_size: float = 5.0
    hist_bin_size: float = 0.05
    occ_threshold: int = 50
    
    # Driving params
    follow_speed: float = 0.1
    turn_speed: float = 0.1
    exit_distance: float = 0.5
    min_turn_radius: float = 0.5
    lookahead_dist: float = 0.6
    row_search_max_dist: float = 2.0
    row_search_width: float = 0.50
    row_exclusion_distance: float = 0.30
    row_extension_max_spline_distance: float = 0.50
    row_window_length: float = 0.65
    row_window_step: float = 0.28
    row_window_beam_width: int = 4
    row_window_candidate_limit: int = 5
    row_window_support_weight: float = 2.8
    row_window_prediction_weight: float = 2.2
    row_window_smoothness_weight: float = 1.1
    row_window_gap_penalty: float = 0.9
    row_cluster_radius: float = 0.18
    row_end_front_point_ratio: float = 0.08
    min_lane_width: float = 0.5
    max_lane_width: float = 1.5
    
    # Control
    pos_kp: float = 2.0
    yaw_kp: float = 1.2
    
    # Pattern
    pattern: str = "1L 2R"
    publish_debug: bool = True


class MaizeNavigator(Node):
    def __init__(self):
        super().__init__("maize_navigator")
        self.p = self.load_params()
        
        self.state = MissionState.IDLE
        self.current_pattern_idx = 0
        self.pattern_steps = self.parse_pattern(self.p.pattern)
        
        self.latest_map: Optional[OccupancyGrid] = None
        self.robot_pose: Optional[Pose2D] = None
        
        # We store and dynamically extend the compiled historical points of each active row
        self.left_row_points: Optional[np.ndarray] = None
        self.right_row_points: Optional[np.ndarray] = None
        
        self.left_row_spline: Optional[PlantSpline] = None
        self.right_row_spline: Optional[PlantSpline] = None
        self.current_left_row_id: Optional[int] = None
        self.current_right_row_id: Optional[int] = None
        self.next_row_id: int = 1
        self.known_rows: dict[int, PlantSpline] = {}
        self.row_ids_in_order: List[int] = []
        self.row_exclusion_distance: float = self.p.row_exclusion_distance
        
        # State specific variables
        self.exit_start_pose: Optional[Pose2D] = None
        self.turn_start_pose: Optional[Pose2D] = None
        self.target_row_y_offset: float = 0.0
        self.row_entry_pose: Optional[Pose2D] = None
        self.row_end_confirm_frames: int = 0
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers/Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, self.p.map_topic, self.map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "navigation_markers", 10)
        
        self.start_srv = self.create_service(Trigger, "start_navigation", self.start_cb)
        self.stop_srv = self.create_service(Trigger, "stop_navigation", self.stop_cb)
        
        self.timer = self.create_timer(1.0 / self.p.control_frequency, self.control_loop)
        self.get_logger().info("Maize Navigator Initialized with Sliding Windows")

    def load_params(self) -> NavigatorParams:
        p = NavigatorParams()
        def get_param(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value
            
        p.pattern = get_param("pattern", p.pattern)
        p.expected_row_width = get_param("expected_row_width", p.expected_row_width)
        p.follow_speed = get_param("follow_speed", p.follow_speed)
        p.min_turn_radius = get_param("min_turn_radius", p.min_turn_radius)
        p.exit_distance = get_param("exit_distance", p.exit_distance)
        p.row_search_max_dist = get_param("row_search_max_dist", p.row_search_max_dist)
        p.row_search_width = get_param("row_search_width", p.row_search_width)
        p.row_exclusion_distance = get_param("row_exclusion_distance", p.row_exclusion_distance)
        p.row_extension_max_spline_distance = get_param("row_extension_max_spline_distance", p.row_extension_max_spline_distance)
        p.row_window_length = get_param("row_window_length", p.row_window_length)
        p.row_window_step = get_param("row_window_step", p.row_window_step)
        p.row_window_beam_width = get_param("row_window_beam_width", p.row_window_beam_width)
        p.row_window_candidate_limit = get_param("row_window_candidate_limit", p.row_window_candidate_limit)
        p.row_window_support_weight = get_param("row_window_support_weight", p.row_window_support_weight)
        p.row_window_prediction_weight = get_param("row_window_prediction_weight", p.row_window_prediction_weight)
        p.row_window_smoothness_weight = get_param("row_window_smoothness_weight", p.row_window_smoothness_weight)
        p.row_window_gap_penalty = get_param("row_window_gap_penalty", p.row_window_gap_penalty)
        p.row_cluster_radius = get_param("row_cluster_radius", p.row_cluster_radius)
        p.row_end_front_point_ratio = get_param("row_end_front_point_ratio", p.row_end_front_point_ratio)
        p.publish_debug = get_param("publish_debug", p.publish_debug)
        p.lookahead_dist = get_param("lookahead_dist", p.lookahead_dist)
        p.min_lane_width = get_param("min_lane_width", p.min_lane_width)
        p.max_lane_width = get_param("max_lane_width", p.max_lane_width)
        return p

    def parse_pattern(self, pattern_str: str) -> List[Tuple[int, str]]:
        steps = []
        try:
            for part in pattern_str.split():
                num = int(part[:-1])
                side = part[-1].upper()
                steps.append((num, side))
        except:
            self.get_logger().error(f"Invalid pattern: {pattern_str}")
        return steps

    def remember_row(self, row_id: int, spline: PlantSpline) -> None:
        self.known_rows[row_id] = spline
        if row_id not in self.row_ids_in_order:
            self.row_ids_in_order.append(row_id)

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def start_cb(self, req, res):
        self.state = MissionState.INITIALIZING
        res.success = True
        res.message = "Navigation started"
        return res

    def stop_cb(self, req, res):
        self.state = MissionState.IDLE
        self.cmd_pub.publish(Twist())
        res.success = True
        return res

    def get_robot_pose(self) -> Optional[Pose2D]:
        try:
            t = self.tf_buffer.lookup_transform(self.p.map_frame, self.p.base_frame, rclpy.time.Time())
            q = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return Pose2D(t.transform.translation.x, t.transform.translation.y, yaw)
        except Exception as e:
            return None

    def control_loop(self):
        self.robot_pose = self.get_robot_pose()
        if self.robot_pose is None or self.latest_map is None:
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

    def handle_initializing(self):
        self.get_logger().info("Detecting rows...", throttle_duration_sec=2.0)
        points = self.get_map_points_in_roi(self.robot_pose, self.p.hist_roi_size)
        if len(points) < 5: 
            self.get_logger().info(f"Not enough points in ROI: {len(points)}", throttle_duration_sec=2.0)
            return

        points = self.filter_points_near_known_rows(points)
        if len(points) < 5:
            self.get_logger().info("All ROI points belong to already known rows.", throttle_duration_sec=2.0)
            return

        # Histogram logic
        c, s = math.cos(-self.robot_pose.yaw), math.sin(-self.robot_pose.yaw)
        dx, dy = points[:, 0] - self.robot_pose.x, points[:, 1] - self.robot_pose.y
        local_y = s * dx + c * dy
        
        bins = np.arange(-self.p.hist_roi_size/2, self.p.hist_roi_size/2, self.p.hist_bin_size)
        hist, bin_edges = np.histogram(local_y, bins=bins)
        
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > 2 and hist[i] >= hist[i-1] and hist[i] >= hist[i+1]:
                peaks.append((bin_edges[i] + bin_edges[i+1])/2)
        
        self.get_logger().info(f"Found {len(points)} points, peaks at: {peaks}", throttle_duration_sec=2.0)
        
        if len(peaks) < 2:
            self.get_logger().info(f"Not enough peaks for row pair selection: {peaks}", throttle_duration_sec=2.0)
            return

        best_left = None
        best_right = None
        best_score = float('inf')

        # Use adjacent peaks so peak 1 maps to spline 1, peak 2 to spline 2, etc.
        # This keeps the row order stable even if all peaks are on the same side.
        sorted_peaks = sorted(peaks)
        for i in range(len(sorted_peaks) - 1):
            p1 = sorted_peaks[i]
            p2 = sorted_peaks[i + 1]
            sep = abs(p2 - p1)
            if sep < self.p.min_lane_width or sep > self.p.max_lane_width:
                continue
            score = abs(sep - self.p.expected_row_width)
            if score < best_score:
                best_score = score
                best_left = p2
                best_right = p1

        if best_left is None or best_right is None:
            self.get_logger().info(f"Could not select a row pair from peaks: {peaks}", throttle_duration_sec=2.0)
            return

        self.get_logger().info(
            f"Selected peak pair: upper={best_left:.3f}, lower={best_right:.3f}, sep={abs(best_left - best_right):.3f}, expected={self.p.expected_row_width:.3f}",
            throttle_duration_sec=2.0,
        )

        if abs(best_left - best_right) < self.p.min_lane_width or abs(best_left - best_right) > self.p.max_lane_width:
            self.get_logger().info(
                f"Rejecting row pair by width: sep={abs(best_left - best_right):.2f}, expected={self.p.expected_row_width:.2f}",
                throttle_duration_sec=2.0,
            )
            return
        
        self.left_row_points = points[np.abs(local_y - best_left) < 0.25]
        self.right_row_points = points[np.abs(local_y - best_right) < 0.25]
        
        self.left_row_spline = PlantSpline(
            self.left_row_points,
            heading_yaw=self.robot_pose.yaw,
            window_length=self.p.row_window_length,
            window_step=self.p.row_window_step,
            beam_width=self.p.row_window_beam_width,
            candidate_limit=self.p.row_window_candidate_limit,
            support_radius=self.p.row_extension_max_spline_distance,
            cluster_radius=self.p.row_cluster_radius,
            support_weight=self.p.row_window_support_weight,
            prediction_weight=self.p.row_window_prediction_weight,
            smoothness_weight=self.p.row_window_smoothness_weight,
            gap_penalty=self.p.row_window_gap_penalty,
            logger=self.get_logger(),
        )
        self.right_row_spline = PlantSpline(
            self.right_row_points,
            heading_yaw=self.robot_pose.yaw,
            window_length=self.p.row_window_length,
            window_step=self.p.row_window_step,
            beam_width=self.p.row_window_beam_width,
            candidate_limit=self.p.row_window_candidate_limit,
            support_radius=self.p.row_extension_max_spline_distance,
            cluster_radius=self.p.row_cluster_radius,
            support_weight=self.p.row_window_support_weight,
            prediction_weight=self.p.row_window_prediction_weight,
            smoothness_weight=self.p.row_window_smoothness_weight,
            gap_penalty=self.p.row_window_gap_penalty,
            logger=self.get_logger(),
        )
        
        if self.left_row_spline.valid and self.right_row_spline.valid:
            self.current_left_row_id = self.next_row_id
            self.next_row_id += 1
            self.current_right_row_id = self.next_row_id
            self.next_row_id += 1
            self.remember_row(self.current_left_row_id, self.left_row_spline)
            self.remember_row(self.current_right_row_id, self.right_row_spline)
            self.row_entry_pose = self.robot_pose
            self.row_end_confirm_frames = 0
            self.state = MissionState.FOLLOW_ROW
            self.get_logger().info(
                f"Rows found. IDs: L={self.current_left_row_id}, R={self.current_right_row_id}. Assigned to peak order {self.row_ids_in_order}. Width: {best_left - best_right:.2f}m. Following..."
            )

    def handle_follow_row(self):
        # 1. Project robot onto current splines
        p_robot = np.array([self.robot_pose.x, self.robot_pose.y])
        t_l = self.left_row_spline.project(p_robot)
        t_r = self.right_row_spline.project(p_robot)

        # 2. Paired cluster march: a cluster may only belong to one row, and both rows advance together.
        new_pts_l, new_pts_r, shared_heading_yaw = self._march_row_pair_clusters(self.left_row_spline, self.right_row_spline)

        if len(new_pts_l) > 0:
            self.left_row_points = self.accumulate_unique_points(
                self.left_row_points, new_pts_l, exclude_row_id=self.current_left_row_id
            )
            temp_spline = self.left_row_spline.extend_with_points(
                new_pts_l,
                heading_yaw=shared_heading_yaw,
                window_length=self.p.row_window_length,
                window_step=self.p.row_window_step,
                beam_width=self.p.row_window_beam_width,
                candidate_limit=self.p.row_window_candidate_limit,
                support_radius=self.p.row_extension_max_spline_distance,
                cluster_radius=self.p.row_cluster_radius,
                support_weight=self.p.row_window_support_weight,
                prediction_weight=self.p.row_window_prediction_weight,
                smoothness_weight=self.p.row_window_smoothness_weight,
                gap_penalty=self.p.row_window_gap_penalty,
                logger=self.get_logger(),
            )
            if temp_spline.valid:
                ok = True
                if self.right_row_spline and self.right_row_spline.valid:
                    dists = self.spline_point_distances(temp_spline.get_points(num=60), self.right_row_spline)
                    if len(dists) > 0 and np.min(dists) < max(self.row_exclusion_distance, self.p.expected_row_width * 0.45):
                        ok = False
                        self.get_logger().info(f"Rejecting left temp_spline: too close to right row (min_dist={np.min(dists):.3f})")
                if ok and self.right_row_spline and self.right_row_spline.valid:
                    if not self.validate_row_pair_geometry(temp_spline, self.right_row_spline):
                        ok = False
                        self.get_logger().info("Rejecting left temp_spline: invalid left/right row geometry")
                if ok:
                    self.left_row_spline = temp_spline
                    if self.current_left_row_id is not None:
                        self.remember_row(self.current_left_row_id, self.left_row_spline)

        if len(new_pts_r) > 0:
            self.right_row_points = self.accumulate_unique_points(
                self.right_row_points, new_pts_r, exclude_row_id=self.current_right_row_id
            )
            temp_spline = self.right_row_spline.extend_with_points(
                new_pts_r,
                heading_yaw=shared_heading_yaw,
                window_length=self.p.row_window_length,
                window_step=self.p.row_window_step,
                beam_width=self.p.row_window_beam_width,
                candidate_limit=self.p.row_window_candidate_limit,
                support_radius=self.p.row_extension_max_spline_distance,
                cluster_radius=self.p.row_cluster_radius,
                support_weight=self.p.row_window_support_weight,
                prediction_weight=self.p.row_window_prediction_weight,
                smoothness_weight=self.p.row_window_smoothness_weight,
                gap_penalty=self.p.row_window_gap_penalty,
                logger=self.get_logger(),
            )
            if temp_spline.valid:
                ok = True
                if self.left_row_spline and self.left_row_spline.valid:
                    dists = self.spline_point_distances(temp_spline.get_points(num=60), self.left_row_spline)
                    if len(dists) > 0 and np.min(dists) < max(self.row_exclusion_distance, self.p.expected_row_width * 0.45):
                        ok = False
                        self.get_logger().info(f"Rejecting right temp_spline: too close to left row (min_dist={np.min(dists):.3f})")
                if ok and self.left_row_spline and self.left_row_spline.valid:
                    if not self.validate_row_pair_geometry(self.left_row_spline, temp_spline):
                        ok = False
                        self.get_logger().info("Rejecting right temp_spline: invalid left/right row geometry")
                if ok:
                    self.right_row_spline = temp_spline
                    if self.current_right_row_id is not None:
                        self.remember_row(self.current_right_row_id, self.right_row_spline)

        # 3. Project robot on updated, highly robust splines
        t_l = self.left_row_spline.project(p_robot)
        t_r = self.right_row_spline.project(p_robot)

        # 4. Robust End of row detection using the forward lookahead window
        points_ahead = self.get_map_points_in_front(self.robot_pose, distance=2.5, width=1.0)
        dist_in_row = math.hypot(self.robot_pose.x - self.row_entry_pose.x, self.robot_pose.y - self.row_entry_pose.y)
        min_front_points = self.estimate_front_point_threshold(self.robot_pose)
        
        near_spline_end = (t_l > self.left_row_spline.t_max - 0.2) and (t_r > self.right_row_spline.t_max - 0.2)
        no_points_ahead = len(points_ahead) < min_front_points

        end_candidate = dist_in_row > 2.0 and near_spline_end and no_points_ahead
        if end_candidate:
            self.row_end_confirm_frames += 1
        else:
            self.row_end_confirm_frames = 0

        if self.row_end_confirm_frames >= 4:
            self.get_logger().info(
                f"Row end confirmed after debounce. Dist: {dist_in_row:.2f}m, Points ahead: {len(points_ahead)}/{min_front_points}, frames: {self.row_end_confirm_frames}"
            )
            self.exit_start_pose = self.robot_pose
            self.row_end_confirm_frames = 0
            self.state = MissionState.EXIT_ROW
            return

        # 5. Pure Pursuit on mid-spline with local cross-track correction
        t_lookahead = (t_l + t_r) / 2.0 + self.p.lookahead_dist
        t_l_eval = max(self.left_row_spline.t_min, min(t_lookahead, self.left_row_spline.t_max))
        t_r_eval = max(self.right_row_spline.t_min, min(t_lookahead, self.right_row_spline.t_max))
        
        p_l = np.array(self.left_row_spline.evaluate(t_l_eval))
        p_r = np.array(self.right_row_spline.evaluate(t_r_eval))
        p_target = (p_l + p_r) / 2.0
        
        # Local lateral error correction
        curr_p_l = np.array(self.left_row_spline.evaluate(t_l))
        curr_p_r = np.array(self.right_row_spline.evaluate(t_r))
        curr_mid = (curr_p_l + curr_p_r) / 2.0
        
        c, s = math.cos(-self.robot_pose.yaw), math.sin(-self.robot_pose.yaw)
        local_dy = s * (curr_mid[0] - self.robot_pose.x) + c * (curr_mid[1] - self.robot_pose.y)
        
        if abs(local_dy) > 0.05:
            target_yaw = self.robot_pose.yaw + math.pi / 2.0
            p_target[0] += math.cos(target_yaw) * local_dy * 0.5
            p_target[1] += math.sin(target_yaw) * local_dy * 0.5
        
        self.drive_to_point(p_target)

    def find_points_along_tangent(
        self,
        start_pos: np.ndarray,
        yaw: float,
        max_dist: float,
        width: float,
        exclude_row_ids: Optional[set[int]] = None,
        current_spline=None,
        min_t: Optional[float] = None,
    ) -> np.ndarray:
        """Looks for plant points ahead of the spline's end point in the direction of its tangent."""
        # Query map cells in a generous bounding area
        points = self.get_map_points_in_roi(Pose2D(start_pos[0], start_pos[1], yaw), max_dist * 2)
        if len(points) == 0:
            return points
            
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx = points[:, 0] - start_pos[0]
        dy = points[:, 1] - start_pos[1]
        
        # Transform points to lookahead window local frame (x_local: along tangent, y_local: lateral deviation)
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        
        # Filter points inside lookahead search window
        mask = (lx > 0.05) & (lx < max_dist) & (np.abs(ly) < width / 2.0)
        cand = points[mask]

        # If current_spline provided, require points to project beyond current t_max (avoid backward points).
        # Allow a wider overlap so slowly updated maps can still extend the row.
        if current_spline is not None and min_t is not None and len(cand) > 0:
            proj_t = np.array([current_spline.project(p) for p in cand])
            eps = 0.35
            cand = cand[proj_t > (min_t - eps)]

        # Keep tail candidates within a narrow corridor around the current spline.
        # This prevents appending points from neighboring rows.
        if current_spline is not None and len(cand) > 0:
            d_to_spline = self.spline_point_distances(cand, current_spline)
            if len(d_to_spline) > 0:
                max_dist = max(0.10, float(self.p.row_extension_max_spline_distance))
                cand = cand[d_to_spline <= max_dist]

        return self.filter_points_near_known_rows(cand, exclude_row_ids=exclude_row_ids)

    def accumulate_unique_points(self, current_pts: np.ndarray, new_pts: np.ndarray, exclude_row_id: Optional[int] = None) -> np.ndarray:
        """Filters new points to only append those that aren't already represented in current_pts.

        Also rejects points that are closer to any known row spline (excluding exclude_row_id) than
        `self.row_exclusion_distance` to avoid stealing points from other rows.
        """
        kept = []

        # Estimate along-axis of the current points to allow accepting points
        # that are further along the row even if they are spatially close
        if current_pts is None or len(current_pts) == 0:
            along_dir = None
            max_proj_cur = -float('inf')
            cur_origin = None
        else:
            cur_origin = current_pts[0]
            vec = current_pts[-1] - current_pts[0]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                along_dir = None
                max_proj_cur = -float('inf')
            else:
                along_dir = vec / norm
                proj_vals = np.dot(current_pts - cur_origin, along_dir)
                max_proj_cur = float(np.max(proj_vals))

        for pt in new_pts:
            # distance to existing current points
            if current_pts is None or len(current_pts) == 0:
                min_cur = float('inf')
            else:
                dists = np.linalg.norm(current_pts - pt, axis=1)
                min_cur = float(np.min(dists))

            # distance to known rows (exclude the current row id)
            min_known = float('inf')
            for row_id, spline in self.known_rows.items():
                if exclude_row_id is not None and row_id == exclude_row_id:
                    continue
                d = self.spline_point_distances(np.array([pt]), spline)
                if len(d) > 0:
                    min_known = min(min_known, float(d[0]))

            accept = False
            # standard acceptance when point is sufficiently far from existing points
            if min_cur > 0.02 and min_known > self.row_exclusion_distance:
                accept = True
            else:
                # If point projects further along the current point chain, accept
                if along_dir is not None and cur_origin is not None:
                    proj_pt = float(np.dot(pt - cur_origin, along_dir))
                    if proj_pt > max_proj_cur and min_known > self.row_exclusion_distance:
                        accept = True

            if accept:
                kept.append(pt)

        if len(kept) > 0:
            if current_pts is None or len(current_pts) == 0:
                return np.array(kept)
            return np.vstack((current_pts, np.array(kept)))
        return current_pts

    def is_candidate_continuous(self, current_spline: PlantSpline, candidate_spline: PlantSpline, max_delta: float = 0.7) -> bool:
        if current_spline is None or candidate_spline is None:
            return False
        if not current_spline.valid or not candidate_spline.valid:
            return False

        t_ref = min(current_spline.t_max, candidate_spline.t_max)
        t_ref = max(candidate_spline.t_min, min(t_ref, candidate_spline.t_max))
        t_cur = max(current_spline.t_min, min(t_ref, current_spline.t_max))

        yaw_old = current_spline.get_direction(t_cur)
        yaw_new = candidate_spline.get_direction(t_ref)
        delta = abs(wrap_to_pi(yaw_new - yaw_old))
        return delta <= max_delta

    def validate_row_pair_geometry(self, left_spline: PlantSpline, right_spline: PlantSpline) -> bool:
        if self.robot_pose is None:
            return True
        if left_spline is None or right_spline is None:
            return False
        if not left_spline.valid or not right_spline.valid:
            return False

        p_robot = np.array([self.robot_pose.x, self.robot_pose.y])
        t_l = left_spline.project(p_robot)
        t_r = right_spline.project(p_robot)
        t_l = max(left_spline.t_min, min(t_l, left_spline.t_max))
        t_r = max(right_spline.t_min, min(t_r, right_spline.t_max))

        p_l = np.array(left_spline.evaluate(t_l))
        p_r = np.array(right_spline.evaluate(t_r))

        c, s = math.cos(-self.robot_pose.yaw), math.sin(-self.robot_pose.yaw)
        dy_l = s * (p_l[0] - self.robot_pose.x) + c * (p_l[1] - self.robot_pose.y)
        dy_r = s * (p_r[0] - self.robot_pose.x) + c * (p_r[1] - self.robot_pose.y)
        sep = dy_l - dy_r

        min_sep = max(0.25, 0.45 * self.p.expected_row_width)
        max_sep = 2.2 * self.p.expected_row_width
        return sep > min_sep and sep < max_sep

    def spline_point_distances(self, points: np.ndarray, spline: PlantSpline) -> np.ndarray:
        if len(points) == 0 or spline is None or not spline.valid:
            return np.array([])
        spline_points = spline.get_points(num=60)
        deltas = points[:, None, :] - spline_points[None, :, :]
        return np.min(np.linalg.norm(deltas, axis=2), axis=1)

    def filter_points_near_known_rows(self, points: np.ndarray, exclude_row_ids: Optional[set[int]] = None, min_dist: Optional[float] = None) -> np.ndarray:
        if len(points) == 0 or not self.known_rows:
            return points
        threshold = self.row_exclusion_distance if min_dist is None else min_dist
        mask = np.ones(len(points), dtype=bool)
        for row_id, spline in self.known_rows.items():
            if exclude_row_ids and row_id in exclude_row_ids:
                continue
            distances = self.spline_point_distances(points, spline)
            if len(distances) > 0:
                mask &= distances > threshold
        return points[mask]

    def get_map_points_in_front(self, pose: Pose2D, distance: float, width: float) -> np.ndarray:
        points = self.get_map_points_in_roi(pose, distance * 2)
        if len(points) == 0: return points
        c, s = math.cos(-pose.yaw), math.sin(-pose.yaw)
        dx, dy = points[:, 0] - pose.x, points[:, 1] - pose.y
        local_x, local_y = c * dx - s * dy, s * dx + c * dy
        mask = (local_x > 0.3) & (local_x < distance) & (np.abs(local_y) < width/2.0)
        return points[mask]

    def estimate_front_point_threshold(self, pose: Pose2D) -> int:
        """Estimate the minimum number of points expected ahead of the robot.

        Dense local crops should demand more missing points before a row end is plausible.
        Sparse crops keep a small floor so the robot can still finish rows in weak maps.
        """
        local_points = self.get_map_points_in_roi(pose, self.p.hist_roi_size)
        if len(local_points) == 0:
            return 3

        dynamic_threshold = int(round(len(local_points) * float(self.p.row_end_front_point_ratio)))
        return int(np.clip(dynamic_threshold, 4, 12))

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return vec / norm

    def _cluster_nearby_points(self, points: np.ndarray, cluster_radius: float) -> List[np.ndarray]:
        if len(points) == 0:
            return []
        visited = np.zeros(len(points), dtype=bool)
        clusters: List[np.ndarray] = []
        for seed_idx in range(len(points)):
            if visited[seed_idx]:
                continue
            stack = [seed_idx]
            visited[seed_idx] = True
            member_indices: List[int] = []
            while stack:
                idx = stack.pop()
                member_indices.append(idx)
                deltas = points - points[idx]
                neighbors = np.where((np.linalg.norm(deltas, axis=1) <= cluster_radius) & (~visited))[0]
                for neighbor_idx in neighbors:
                    visited[neighbor_idx] = True
                    stack.append(int(neighbor_idx))
            cluster = points[np.array(member_indices, dtype=int)]
            if len(cluster) >= 2:
                clusters.append(cluster)
        return clusters

    def _direction_from_points(self, points: np.ndarray, fallback_dir: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return self._normalize_vector(np.asarray(fallback_dir, dtype=float))
        centered = points - np.mean(points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        direction = self._normalize_vector(direction)
        fallback_dir = self._normalize_vector(np.asarray(fallback_dir, dtype=float))
        if np.dot(direction, fallback_dir) < 0:
            direction = -direction
        return direction

    def _shared_pair_heading(self, left_spline: PlantSpline, right_spline: PlantSpline, fallback_yaw: float) -> float:
        vectors = []
        if left_spline is not None and left_spline.valid:
            vectors.append(np.array([math.cos(left_spline.get_direction(left_spline.t_max)), math.sin(left_spline.get_direction(left_spline.t_max))]))
        if right_spline is not None and right_spline.valid:
            vectors.append(np.array([math.cos(right_spline.get_direction(right_spline.t_max)), math.sin(right_spline.get_direction(right_spline.t_max))]))
        if not vectors:
            return float(fallback_yaw)
        shared_vec = self._normalize_vector(np.sum(vectors, axis=0))
        return float(math.atan2(shared_vec[1], shared_vec[0]))

    def _march_row_pair_clusters(self, left_spline: PlantSpline, right_spline: PlantSpline) -> Tuple[np.ndarray, np.ndarray, float]:
        if left_spline is None or right_spline is None or not left_spline.valid or not right_spline.valid:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), float(self.robot_pose.yaw if self.robot_pose else 0.0)

        step_distance = 0.5
        forward_max_dist = 2.0
        orthogonal_max_dist = 0.5
        cluster_radius = max(0.12, float(self.p.row_cluster_radius))

        left_end = np.array(left_spline.evaluate(left_spline.t_max), dtype=float)
        right_end = np.array(right_spline.evaluate(right_spline.t_max), dtype=float)
        pair_center = 0.5 * (left_end + right_end)
        shared_heading_yaw = self._shared_pair_heading(left_spline, right_spline, self.robot_pose.yaw if self.robot_pose else 0.0)
        shared_dir = self._normalize_vector(np.array([math.cos(shared_heading_yaw), math.sin(shared_heading_yaw)], dtype=float))

        collected_left: List[np.ndarray] = []
        collected_right: List[np.ndarray] = []
        remaining_points = self.get_map_points_in_roi(Pose2D(pair_center[0], pair_center[1], shared_heading_yaw), 4.5)
        if len(remaining_points) == 0:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), shared_heading_yaw

        left_points_ref = np.array(self.left_row_points, copy=True) if self.left_row_points is not None else np.empty((0, 2), dtype=float)
        right_points_ref = np.array(self.right_row_points, copy=True) if self.right_row_points is not None else np.empty((0, 2), dtype=float)

        for _ in range(12):
            pair_perp = np.array([-shared_dir[1], shared_dir[0]], dtype=float)
            rel = remaining_points - pair_center
            along = rel @ shared_dir
            orth = rel @ pair_perp
            candidate_mask = (along >= 0.0) & (along <= forward_max_dist) & (np.abs(orth) <= orthogonal_max_dist)
            candidate_points = remaining_points[candidate_mask]
            if len(candidate_points) == 0:
                break

            clusters = self._cluster_nearby_points(candidate_points, cluster_radius)
            if not clusters:
                break

            left_ref = float((left_end - pair_center) @ pair_perp)
            right_ref = float((right_end - pair_center) @ pair_perp)
            step_left_clusters: List[np.ndarray] = []
            step_right_clusters: List[np.ndarray] = []

            for cluster in clusters:
                centroid = np.mean(cluster, axis=0)
                centroid_rel = centroid - pair_center
                centroid_along = float(centroid_rel @ shared_dir)
                centroid_orth = float(centroid_rel @ pair_perp)
                if centroid_along < 0.0 or centroid_along > forward_max_dist or abs(centroid_orth) > orthogonal_max_dist:
                    continue
                d_left = abs(centroid_orth - left_ref)
                d_right = abs(centroid_orth - right_ref)
                if d_left <= d_right:
                    step_left_clusters.append(cluster)
                else:
                    step_right_clusters.append(cluster)

            if not step_left_clusters and not step_right_clusters:
                break

            if step_left_clusters:
                left_step_points = np.vstack(step_left_clusters)
                collected_left.append(left_step_points)
            else:
                left_step_points = np.empty((0, 2), dtype=float)
            if step_right_clusters:
                right_step_points = np.vstack(step_right_clusters)
                collected_right.append(right_step_points)
            else:
                right_step_points = np.empty((0, 2), dtype=float)

            left_dir = self._direction_from_points(left_step_points, shared_dir) if len(left_step_points) > 0 else shared_dir
            right_dir = self._direction_from_points(right_step_points, shared_dir) if len(right_step_points) > 0 else shared_dir
            shared_dir = self._normalize_vector(left_dir + right_dir)
            shared_heading_yaw = float(math.atan2(shared_dir[1], shared_dir[0]))

            remove_points = np.vstack([pts for pts in (left_step_points, right_step_points) if len(pts) > 0])
            if len(remove_points) > 0:
                keep_mask = np.ones(len(remaining_points), dtype=bool)
                for ref_point in remove_points:
                    keep_mask &= np.linalg.norm(remaining_points - ref_point, axis=1) > cluster_radius
                remaining_points = remaining_points[keep_mask]
            pair_center = pair_center + shared_dir * step_distance
            left_end = left_end + shared_dir * step_distance
            right_end = right_end + shared_dir * step_distance
            if len(remaining_points) == 0:
                break

        left_out = np.vstack(collected_left) if collected_left else np.empty((0, 2), dtype=float)
        right_out = np.vstack(collected_right) if collected_right else np.empty((0, 2), dtype=float)
        return left_out, right_out, shared_heading_yaw

    def handle_exit_row(self):
        dist = math.hypot(self.robot_pose.x - self.exit_start_pose.x, self.robot_pose.y - self.exit_start_pose.y)
        if dist >= self.p.exit_distance:
            self.get_logger().info("Exit complete. Turning...")
            self.turn_start_pose = self.robot_pose
            self.state = MissionState.TURN_OUT
            return
        cmd = Twist()
        cmd.linear.x = self.p.follow_speed
        self.cmd_pub.publish(cmd)

    def handle_turn_out(self):
        num, side = self.pattern_steps[self.current_pattern_idx]
        dir = 1.0 if side == "L" else -1.0
        yaw_diff = abs(wrap_to_pi(self.robot_pose.yaw - self.turn_start_pose.yaw))
        if yaw_diff >= math.pi / 2.0 - 0.1:
            self.get_logger().info("Turn out complete. Shifting...")
            self.target_row_y_offset = num * self.p.expected_row_width
            self.turn_start_pose = self.robot_pose
            self.state = MissionState.SHIFT
            return
        cmd = Twist()
        cmd.linear.x = self.p.turn_speed
        cmd.angular.z = dir * (self.p.turn_speed / self.p.min_turn_radius)
        self.cmd_pub.publish(cmd)

    def handle_shift(self):
        dist = math.hypot(self.robot_pose.x - self.turn_start_pose.x, self.robot_pose.y - self.turn_start_pose.y)
        if dist >= abs(self.target_row_y_offset) - 2 * self.p.min_turn_radius:
            self.get_logger().info("Shift complete. Turning in...")
            self.turn_start_pose = self.robot_pose
            self.state = MissionState.TURN_IN
            return
        cmd = Twist()
        cmd.linear.x = self.p.follow_speed
        self.cmd_pub.publish(cmd)

    def handle_turn_in(self):
        num, side = self.pattern_steps[self.current_pattern_idx]
        dir = 1.0 if side == "L" else -1.0
        yaw_diff = abs(wrap_to_pi(self.robot_pose.yaw - self.turn_start_pose.yaw))
        if yaw_diff >= math.pi / 2.0 - 0.1:
            self.get_logger().info("Entered new row. Re-initializing...")
            self.current_pattern_idx = (self.current_pattern_idx + 1) % len(self.pattern_steps)
            self.row_entry_pose = self.robot_pose
            self.state = MissionState.INITIALIZING
            return
        cmd = Twist()
        cmd.linear.x = self.p.turn_speed
        cmd.angular.z = dir * (self.p.turn_speed / self.p.min_turn_radius)
        self.cmd_pub.publish(cmd)

    def drive_to_point(self, target: np.ndarray):
        dx, dy = target[0] - self.robot_pose.x, target[1] - self.robot_pose.y
        angle_to_target = math.atan2(dy, dx)
        yaw_err = wrap_to_pi(angle_to_target - self.robot_pose.yaw)
        yaw_err = np.clip(yaw_err, -0.4, 0.4)
        cmd = Twist()
        speed_factor = np.clip(1.0 - abs(yaw_err)/0.6, 0.3, 1.0)
        cmd.linear.x = self.p.follow_speed * speed_factor
        cmd.angular.z = self.p.yaw_kp * yaw_err
        
        self.get_logger().info(
            f"Fahrbefehl: v={cmd.linear.x:.2f}, w={cmd.angular.z:.2f} (yaw_err={yaw_err:.2f}, dist={math.hypot(dx, dy):.2f})",
            throttle_duration_sec=0.5
        )
        self.cmd_pub.publish(cmd)

    def get_map_points_in_roi(self, pose: Pose2D, size: float) -> np.ndarray:
        if self.latest_map is None: return np.array([])
        info = self.latest_map.info
        grid = np.array(self.latest_map.data).reshape((info.height, info.width))
        ix = int((pose.x - info.origin.position.x) / info.resolution)
        iy = int((pose.y - info.origin.position.y) / info.resolution)
        r = int(size / (2 * info.resolution))
        y_slice = slice(max(0, iy-r), min(info.height, iy+r))
        x_slice = slice(max(0, ix-r), min(info.width, ix+r))
        roi = grid[y_slice, x_slice]
        occ_y, occ_x = np.where(roi > self.p.occ_threshold)
        world_x = (occ_x + x_slice.start) * info.resolution + info.origin.position.x
        world_y = (occ_y + y_slice.start) * info.resolution + info.origin.position.y
        return np.column_stack((world_x, world_y))

    def publish_visuals(self):
        markers = MarkerArray()
        ordered_ids = self.row_ids_in_order if self.row_ids_in_order else sorted(self.known_rows.keys())
        for row_id in ordered_ids:
            spline = self.known_rows[row_id]
            if spline is None or not spline.valid:
                continue
            is_current = row_id in {self.current_left_row_id, self.current_right_row_id}
            color = [0.0, 1.0, 0.0] if is_current else [0.65, 0.65, 0.65]
            alpha = 1.0 if is_current else 0.55
            markers.markers.append(self.create_spline_marker(spline, row_id, color, alpha=alpha))
            if spline.segment_bounds and len(spline.segment_bounds) > 1:
                markers.markers.append(self.create_segment_boundary_marker(spline, row_id, is_current=is_current))
            # publish debug markers for rejected/considered hypotheses
            if self.p.publish_debug and hasattr(spline, "_last_rejected") and len(spline._last_rejected) > 0:
                for idx, beam in enumerate(spline._last_rejected[:12]):
                    m = Marker(header=Header(frame_id=self.p.map_frame, stamp=self.get_clock().now().to_msg()))
                    m.ns, m.id, m.type, m.action = "spline_rejected", 20000 + row_id * 100 + idx, Marker.LINE_STRIP, Marker.ADD
                    m.scale.x = 0.03
                    # more transparent for lower-scoring beams
                    alpha = 0.18 + 0.55 * float(np.clip((beam.get("score", 0.0) + 5.0) / 10.0, 0.0, 1.0))
                    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.12, 0.12, alpha
                    for t_val, l_val in zip(beam.get("t", []), beam.get("l", [])):
                        x = spline.mean[0] + float(t_val) * spline.main_dir[0] + float(l_val) * spline.perp_dir[0]
                        y = spline.mean[1] + float(t_val) * spline.main_dir[1] + float(l_val) * spline.perp_dir[1]
                        m.points.append(Point(x=float(x), y=float(y), z=0.0))
                    markers.markers.append(m)
            if self.p.publish_debug and hasattr(spline, "_last_accepted") and len(spline._last_accepted) > 0:
                for idx, beam in enumerate(spline._last_accepted[:3]):
                    m2 = Marker(header=Header(frame_id=self.p.map_frame, stamp=self.get_clock().now().to_msg()))
                    m2.ns, m2.id, m2.type, m2.action = "spline_accepted", 30000 + row_id * 100 + idx, Marker.LINE_STRIP, Marker.ADD
                    m2.scale.x = 0.05
                    m2.color.r, m2.color.g, m2.color.b, m2.color.a = 0.15, 1.0, 0.15, 0.95
                    for t_val, l_val in zip(beam.get("t", []), beam.get("l", [])):
                        x = spline.mean[0] + float(t_val) * spline.main_dir[0] + float(l_val) * spline.perp_dir[0]
                        y = spline.mean[1] + float(t_val) * spline.main_dir[1] + float(l_val) * spline.perp_dir[1]
                        m2.points.append(Point(x=float(x), y=float(y), z=0.0))
                    markers.markers.append(m2)
        self.marker_pub.publish(markers)

    def create_spline_marker(self, spline: PlantSpline, id: int, color: list, alpha: float = 1.0) -> Marker:
        m = Marker(header=Header(frame_id=self.p.map_frame, stamp=self.get_clock().now().to_msg()))
        m.ns, m.id, m.type, m.action = "splines", id, Marker.LINE_STRIP, Marker.ADD
        m.scale.x = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], alpha
        for p in spline.get_points():
            m.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
        return m

    def create_segment_boundary_marker(self, spline: PlantSpline, id: int, is_current: bool = False) -> Marker:
        m = Marker(header=Header(frame_id=self.p.map_frame, stamp=self.get_clock().now().to_msg()))
        m.ns, m.id, m.type, m.action = "row_segments", 10000 + id, Marker.SPHERE_LIST, Marker.ADD
        m.scale.x = 0.12 if is_current else 0.08
        m.scale.y = 0.12 if is_current else 0.08
        m.scale.z = 0.12 if is_current else 0.08
        if is_current:
            m.color.r = 1.0
            m.color.g = 0.9
            m.color.b = 0.05
            m.color.a = 0.98
        else:
            m.color.r = 1.0
            m.color.g = 0.35
            m.color.b = 0.05
            m.color.a = 0.72

        for boundary_start, boundary_end in spline.segment_bounds[1:]:
            t_boundary = 0.5 * (boundary_start + boundary_end)
            p = spline.evaluate(t_boundary)
            m.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
        return m

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
