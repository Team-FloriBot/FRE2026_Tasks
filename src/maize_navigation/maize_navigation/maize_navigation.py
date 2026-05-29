#!/usr/bin/env python3

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

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


# --- Data Structures for Splines ---

class PlantSpline:
    def __init__(self, points: np.ndarray, heading_yaw: float = 0.0, s: float = 60.0, max_k: int = 3, logger: Optional[Node] = None):
        """
        points: (N, 2) array of (x, y) coordinates in map frame
        heading_yaw: Approximate current heading yaw of the robot to align spline direction
        s: Balanced smoothing factor to allow tracking real curves while filtering high-frequency noise
        logger: Optional logger for debugging messages
        """
        self.valid = False
        self.t_min = 0.0
        self.t_max = 0.0
        self.linear_mode = False
        self.points = points
        self.main_dir = None
        self.perp_dir = None
        self.mean = None
        self.segment_splines: List[UnivariateSpline] = []
        self.segment_bounds: List[Tuple[float, float]] = []
        self.lateral_spline = None
        
        if len(points) < 4:
            return
            
        # PCA to find main direction
        self.mean = np.mean(points, axis=0)
        centered = points - self.mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        self.main_dir = eigenvectors[:, np.argmax(eigenvalues)]
        self.perp_dir = np.array([-self.main_dir[1], self.main_dir[0]])
        
        # Align spline direction with the approximate robot heading
        heading_vec = np.array([math.cos(heading_yaw), math.sin(heading_yaw)])
        if np.dot(self.main_dir, heading_vec) < 0:
            self.main_dir = -self.main_dir
            
        # Project points onto main direction to sort them
        projections = centered @ self.main_dir
        lateral = centered @ self.perp_dir
        sort_idx = np.argsort(projections)
        self.points = points[sort_idx]
        
        self.t = projections[sort_idx]
        self.lateral = lateral[sort_idx]
        # Remove duplicate t values for spline fitting
        unique_t, unique_idx = np.unique(self.t, return_index=True)
        if len(unique_t) < 4:
            return 

        self.t_min = unique_t[0]
        self.t_max = unique_t[-1]

        if logger:
            logger.info(f"Fitting spline with {len(unique_t)} unique points, t range: [{self.t_min:.2f}, {self.t_max:.2f}]")

        # Zähle räumliche Cluster: wenn Punkte in Gruppen angehäuft sind (z.B. 2-3 Anhäufungen),
        # ist die Reihe wahrscheinlich gerade. Viele verstreute Cluster deuten auf Krümmung hin.
        sorted_pts = self.points[unique_idx]
        num_clusters = 1
        cluster_gap_threshold = 0.60  # Meter: ab dieser Lücke ein neuer Cluster
        
        for i in range(len(sorted_pts) - 1):
            dist = np.linalg.norm(sorted_pts[i+1] - sorted_pts[i])
            if dist > cluster_gap_threshold:
                num_clusters += 1
        
        # Bei wirklich wenigen Punkten ist ein stabiler Linear-Fallback besser als ein instabiler Spline.
        # Die Cluster-Heuristik bleibt nur als Hinweis im Log erhalten, sie erzwingt aber keinen Linear-Fallback mehr.
        if len(unique_t) < 6:
            self.linear_mode = True
            self.line_start = self.mean + self.t_min * self.main_dir
            self.line_end = self.mean + self.t_max * self.main_dir
            self.valid = True
            if logger:
                logger.info(f"Using linear mode (n={len(unique_t)}, clusters={num_clusters})")
            return

        
        fit_t = unique_t
        fit_lateral = self.lateral[unique_idx]

        # Build robust local control points so individual displaced plants do not drag the row centerline away.
        # Median values per sliding window keep small bends while suppressing outliers.
        if len(fit_t) > 8:
            window_size = max(5, min(12, len(fit_t) // 4))
            step_size = max(2, window_size // 2)
            control_t = []
            control_lateral = []
            for start in range(0, len(fit_t), step_size):
                end = min(len(fit_t), start + window_size)
                if end - start < 4:
                    continue
                control_t.append(float(np.median(fit_t[start:end])))
                control_lateral.append(float(np.median(fit_lateral[start:end])))
            if len(control_t) >= 4:
                fit_t = np.asarray(control_t, dtype=float)
                fit_lateral = np.asarray(control_lateral, dtype=float)

        # Piecewise 1D-Fit: the row is represented by overlapping local splines in the same PCA frame.
        # This keeps small bends and outliers local instead of pulling the complete row shape around.
        try:
            self.segment_splines = []
            self.segment_bounds = []

            if len(fit_t) < 6:
                self.linear_mode = True
                self.line_start = self.mean + self.t_min * self.main_dir
                self.line_end = self.mean + self.t_max * self.main_dir
                self.valid = True
                if logger:
                    logger.info(f"Using linear mode (n={len(fit_t)}, clusters={num_clusters})")
                return

            segment_size = max(6, min(12, len(fit_t) // 5 if len(fit_t) >= 15 else len(fit_t)))
            overlap = max(3, int(round(segment_size * 0.6)))
            start_idx = 0

            while start_idx < len(fit_t) - 3:
                end_idx = min(len(fit_t), start_idx + segment_size)
                seg_t = fit_t[start_idx:end_idx]
                seg_lateral = fit_lateral[start_idx:end_idx]

                if len(seg_t) >= 4:
                    requested_k = int(max_k)
                    line_coeff = np.polyfit(seg_t, seg_lateral, 1)
                    line_pred = np.polyval(line_coeff, seg_t)
                    line_res = seg_lateral - line_pred
                    line_rms = float(np.sqrt(np.mean(line_res ** 2)))
                    line_max = float(np.max(np.abs(line_res)))
                    near_linear = line_rms < 0.03 and line_max < 0.08

                    k = 1 if near_linear else min(3, max(1, requested_k), len(seg_t) - 1)
                    lateral_spread = float(np.std(seg_lateral))
                    s_used = 0.0 if near_linear else max(0.25, float(s) * 0.02 * max(0.75, 1.0 + 0.15 * lateral_spread))
                    segment = UnivariateSpline(seg_t, seg_lateral, k=k, s=s_used)
                    self.segment_splines.append(segment)
                    self.segment_bounds.append((float(seg_t[0]), float(seg_t[-1])))

                if end_idx >= len(fit_t):
                    break
                start_idx = max(0, end_idx - overlap)

            if not self.segment_splines:
                requested_k = int(max_k)
                k = min(max(1, requested_k), 5, len(fit_t) - 1)
                lateral_spread = float(np.std(fit_lateral))
                s_used = max(0.5, float(s) * 0.05 * max(0.75, 1.0 + 0.15 * lateral_spread))
                self.lateral_spline = UnivariateSpline(fit_t, fit_lateral, k=k, s=s_used)
                self.segment_splines = [self.lateral_spline]
                self.segment_bounds = [(float(fit_t[0]), float(fit_t[-1]))]

            self.valid = True
            if logger:
                logger.info(f"Built {len(self.segment_splines)} spline segment(s)")
        except Exception as exc:
            if logger:
                logger.info(f"Spline fit failed: {exc}")
            self.valid = False
            return

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
        s: float = 60.0,
        max_k: int = 3,
        logger: Optional[Node] = None,
        overlap_ratio: float = 0.55,
    ) -> "PlantSpline":
        if new_points is None or len(new_points) == 0:
            return self
        if self.linear_mode:
            combined = np.vstack((self.points, new_points))
            return PlantSpline(combined, heading_yaw=heading_yaw, s=s, max_k=max_k, logger=logger)

        extended = self._spawn_extension()

        new_points = np.asarray(new_points, dtype=float)
        new_centered = new_points - self.mean
        new_t = new_centered @ self.main_dir
        new_lateral = new_centered @ self.perp_dir

        combined_points = np.vstack((self.points, new_points))
        combined_centered = combined_points - self.mean
        combined_t = combined_centered @ self.main_dir
        combined_lateral = combined_centered @ self.perp_dir
        sort_idx = np.argsort(combined_t)
        combined_t = combined_t[sort_idx]
        combined_lateral = combined_lateral[sort_idx]
        combined_points = combined_points[sort_idx]

        if len(combined_t) < 6:
            return self

        # Only fit the tail plus a generous overlap so the earlier part stays untouched.
        row_span = max(0.0, float(self.t_max - self.t_min))
        overlap_t = max(0.45, row_span * overlap_ratio)
        tail_start = float(self.t_max - overlap_t)
        tail_mask = combined_t >= tail_start

        tail_t = combined_t[tail_mask]
        tail_lateral = combined_lateral[tail_mask]
        tail_points = combined_points[tail_mask]

        if len(tail_t) < 6:
            tail_t = combined_t
            tail_lateral = combined_lateral
            tail_points = combined_points

        if len(tail_t) > 8:
            window_size = max(5, min(10, len(tail_t) // 3))
            step_size = max(2, window_size // 2)
            control_t = []
            control_lateral = []
            for start in range(0, len(tail_t), step_size):
                end = min(len(tail_t), start + window_size)
                if end - start < 4:
                    continue
                control_t.append(float(np.median(tail_t[start:end])))
                control_lateral.append(float(np.median(tail_lateral[start:end])))
            if len(control_t) >= 4:
                tail_t = np.asarray(control_t, dtype=float)
                tail_lateral = np.asarray(control_lateral, dtype=float)

        try:
            requested_k = int(max_k)
            line_coeff = np.polyfit(tail_t, tail_lateral, 1)
            line_pred = np.polyval(line_coeff, tail_t)
            line_res = tail_lateral - line_pred
            line_rms = float(np.sqrt(np.mean(line_res ** 2)))
            line_max = float(np.max(np.abs(line_res)))
            near_linear = line_rms < 0.03 and line_max < 0.08

            k = 1 if near_linear else min(3, max(1, requested_k), len(tail_t) - 1)
            lateral_spread = float(np.std(tail_lateral))
            s_used = 0.0 if near_linear else max(0.2, float(s) * 0.015 * max(0.75, 1.0 + 0.10 * lateral_spread))
            segment = UnivariateSpline(tail_t, tail_lateral, k=k, s=s_used)

            extended.segment_splines = list(self.segment_splines) + [segment]
            extended.segment_bounds = list(self.segment_bounds) + [(float(tail_t[0]), float(tail_t[-1]))]
            extended.points = combined_points
            extended.t = combined_t
            extended.lateral = combined_lateral
            extended.t_min = float(combined_t[0])
            extended.t_max = float(combined_t[-1])
            extended.valid = True
            if logger:
                logger.info(f"Extended spline tail with {len(tail_t)} control points; segments={len(extended.segment_splines)}")
            return extended
        except Exception as exc:
            if logger:
                logger.info(f"Spline tail extension failed: {exc}")
            return self

    def _segment_candidates(self, t: float) -> List[int]:
        if not self.segment_bounds:
            return []
        candidates = [i for i, (t_start, t_end) in enumerate(self.segment_bounds) if t_start <= t <= t_end]
        if candidates:
            return candidates
        centers = [0.5 * (t_start + t_end) for t_start, t_end in self.segment_bounds]
        return [int(np.argmin([abs(t - c) for c in centers]))]

    def _evaluate_lateral(self, t: float) -> float:
        if not self.segment_splines:
            return float(self.lateral_spline(t))
        candidates = self._segment_candidates(t)
        if len(candidates) == 1:
            return float(self.segment_splines[candidates[0]](t))

        scored = sorted(candidates, key=lambda idx: abs(t - 0.5 * (self.segment_bounds[idx][0] + self.segment_bounds[idx][1])))
        first_idx = scored[0]
        second_idx = scored[1]
        first_center = 0.5 * (self.segment_bounds[first_idx][0] + self.segment_bounds[first_idx][1])
        second_center = 0.5 * (self.segment_bounds[second_idx][0] + self.segment_bounds[second_idx][1])
        first_weight = 1.0 / (abs(t - first_center) + 1e-3)
        second_weight = 1.0 / (abs(t - second_center) + 1e-3)
        return float(
            (first_weight * self.segment_splines[first_idx](t) + second_weight * self.segment_splines[second_idx](t))
            / (first_weight + second_weight)
        )

    def _evaluate_lateral_derivative(self, t: float) -> float:
        if not self.segment_splines:
            return float(self.lateral_spline.derivative()(t))
        candidates = self._segment_candidates(t)
        if len(candidates) == 1:
            return float(self.segment_splines[candidates[0]].derivative()(t))

        scored = sorted(candidates, key=lambda idx: abs(t - 0.5 * (self.segment_bounds[idx][0] + self.segment_bounds[idx][1])))
        first_idx = scored[0]
        second_idx = scored[1]
        first_center = 0.5 * (self.segment_bounds[first_idx][0] + self.segment_bounds[first_idx][1])
        second_center = 0.5 * (self.segment_bounds[second_idx][0] + self.segment_bounds[second_idx][1])
        first_weight = 1.0 / (abs(t - first_center) + 1e-3)
        second_weight = 1.0 / (abs(t - second_center) + 1e-3)
        return float(
            (first_weight * self.segment_splines[first_idx].derivative()(t) + second_weight * self.segment_splines[second_idx].derivative()(t))
            / (first_weight + second_weight)
        )

    def evaluate(self, t: float) -> Tuple[float, float]:
        if self.linear_mode:
            if self.t_max <= self.t_min:
                p = self.line_start
            else:
                alpha = (t - self.t_min) / (self.t_max - self.t_min)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                p = self.line_start + alpha * (self.line_end - self.line_start)
            return float(p[0]), float(p[1])
        lateral = self._evaluate_lateral(t)
        p = self.mean + t * self.main_dir + lateral * self.perp_dir
        return float(p[0]), float(p[1])

    def get_direction(self, t: float) -> float:
        if self.linear_mode:
            direction = self.line_end - self.line_start
            return math.atan2(direction[1], direction[0])
        lateral_d = self._evaluate_lateral_derivative(t)
        direction = self.main_dir + lateral_d * self.perp_dir
        dx, dy = float(direction[0]), float(direction[1])
        return math.atan2(dy, dx)

    def get_global_direction(self) -> float:
        """Global row trend from PCA main_dir"""
        return float(math.atan2(self.main_dir[1], self.main_dir[0]))

    def get_points(self, num: int = 50) -> np.ndarray:
        t_vals = np.linspace(self.t_min, self.t_max, num)
        if self.linear_mode:
            pts = [self.evaluate(t) for t in t_vals]
            return np.array(pts)
        lateral_vals = np.array([self._evaluate_lateral(t) for t in t_vals])
        return np.column_stack((self.mean[0] + t_vals * self.main_dir[0] + lateral_vals * self.perp_dir[0],
                                self.mean[1] + t_vals * self.main_dir[1] + lateral_vals * self.perp_dir[1]))
    
    def project(self, point: np.ndarray) -> float:
        """Project world point onto spline parameter t"""
        centered = point - self.mean
        return float(centered @ self.main_dir)


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
    row_search_max_dist: float = 2.2
    row_search_width: float = 0.50
    row_exclusion_distance: float = 0.30
    row_extension_max_spline_distance: float = 0.22
    spline_s: float = 30.0
    max_spline_k: int = 5
    min_lane_width: float = 0.5
    max_lane_width: float = 1.5
    
    # Control
    pos_kp: float = 2.0
    yaw_kp: float = 1.2
    
    # Pattern
    pattern: str = "1L 2R"


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
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers/Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, self.p.map_topic, self.map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "navigation_markers", 10)
        
        self.start_srv = self.create_service(Trigger, "start_navigation", self.start_cb)
        self.stop_srv = self.create_service(Trigger, "stop_navigation", self.stop_cb)
        
        self.timer = self.create_timer(1.0 / self.p.control_frequency, self.control_loop)
        self.get_logger().info("Maize Navigator Initialized with Splines")

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
        p.spline_s = get_param("spline_s", p.spline_s)
        p.max_spline_k = get_param("max_spline_k", p.max_spline_k)
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
        
        self.left_row_spline = PlantSpline(self.left_row_points, heading_yaw=self.robot_pose.yaw, s=self.p.spline_s, max_k=self.p.max_spline_k, logger=self.get_logger())
        self.right_row_spline = PlantSpline(self.right_row_points, heading_yaw=self.robot_pose.yaw, s=self.p.spline_s, max_k=self.p.max_spline_k, logger=self.get_logger())
        
        if self.left_row_spline.valid and self.right_row_spline.valid:
            self.current_left_row_id = self.next_row_id
            self.next_row_id += 1
            self.current_right_row_id = self.next_row_id
            self.next_row_id += 1
            self.remember_row(self.current_left_row_id, self.left_row_spline)
            self.remember_row(self.current_right_row_id, self.right_row_spline)
            self.row_entry_pose = self.robot_pose
            self.state = MissionState.FOLLOW_ROW
            self.get_logger().info(
                f"Rows found. IDs: L={self.current_left_row_id}, R={self.current_right_row_id}. Assigned to peak order {self.row_ids_in_order}. Width: {best_left - best_right:.2f}m. Following..."
            )

    def handle_follow_row(self):
        # 1. Project robot onto current splines
        p_robot = np.array([self.robot_pose.x, self.robot_pose.y])
        t_l = self.left_row_spline.project(p_robot)
        t_r = self.right_row_spline.project(p_robot)

        # 2. Piecewise Chaining logic: Segment growing to prevent global fit deformation!
        # We enforce C1-continuity at the chaining boundary (position and tangent direction).
        # We only fit new cubic splines locally in the latest lookahead segment (max 1.5m blocks).
        
        # For Left Spline:
        end_val_l = np.array(self.left_row_spline.evaluate(self.left_row_spline.t_max))
        local_tangent_l = self.left_row_spline.get_direction(self.left_row_spline.t_max)
        global_trend_l = self.left_row_spline.get_global_direction()
        weighted_yaw_l = wrap_to_pi(0.4 * local_tangent_l + 0.6 * global_trend_l)
        
        new_pts_l = self.find_points_along_tangent(
            end_val_l,
            weighted_yaw_l,
            max_dist=self.p.row_search_max_dist,
            width=self.p.row_search_width,
            exclude_row_ids={self.current_right_row_id} if self.current_right_row_id is not None else None,
            current_spline=self.left_row_spline,
            min_t=self.left_row_spline.t_max if self.left_row_spline is not None else None,
        )
        if len(new_pts_l) > 0:
            self.left_row_points = self.accumulate_unique_points(
                self.left_row_points, new_pts_l, exclude_row_id=self.current_left_row_id
            )
            temp_spline = self.left_row_spline.extend_with_points(
                new_pts_l,
                heading_yaw=self.robot_pose.yaw,
                s=self.p.spline_s,
                max_k=self.p.max_spline_k,
                logger=self.get_logger(),
            )
            if temp_spline.valid:
                # validation: continuity + avoid drifting into opposite row
                ok = True
                if self.left_row_spline and self.left_row_spline.valid:
                    if not self.is_candidate_continuous(self.left_row_spline, temp_spline):
                        ok = False
                        self.get_logger().info("Rejecting left temp_spline: continuity check failed")
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
            else:
                self.get_logger().info("Rejecting left temp_spline: spline fit invalid")

        # For Right Spline:
        end_val_r = np.array(self.right_row_spline.evaluate(self.right_row_spline.t_max))
        local_tangent_r = self.right_row_spline.get_direction(self.right_row_spline.t_max)
        global_trend_r = self.right_row_spline.get_global_direction()
        weighted_yaw_r = wrap_to_pi(0.4 * local_tangent_r + 0.6 * global_trend_r)
        
        new_pts_r = self.find_points_along_tangent(
            end_val_r,
            weighted_yaw_r,
            max_dist=self.p.row_search_max_dist,
            width=self.p.row_search_width,
            exclude_row_ids={self.current_left_row_id} if self.current_left_row_id is not None else None,
            current_spline=self.right_row_spline,
            min_t=self.right_row_spline.t_max if self.right_row_spline is not None else None,
        )
        if len(new_pts_r) > 0:
            self.right_row_points = self.accumulate_unique_points(
                self.right_row_points, new_pts_r, exclude_row_id=self.current_right_row_id
            )
            temp_spline = self.right_row_spline.extend_with_points(
                new_pts_r,
                heading_yaw=self.robot_pose.yaw,
                s=self.p.spline_s,
                max_k=self.p.max_spline_k,
                logger=self.get_logger(),
            )
            if temp_spline.valid:
                ok = True
                if self.right_row_spline and self.right_row_spline.valid:
                    if not self.is_candidate_continuous(self.right_row_spline, temp_spline):
                        ok = False
                        self.get_logger().info("Rejecting right temp_spline: continuity check failed")
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
            else:
                self.get_logger().info("Rejecting right temp_spline: spline fit invalid")

        # 3. Project robot on updated, highly robust splines
        t_l = self.left_row_spline.project(p_robot)
        t_r = self.right_row_spline.project(p_robot)

        # 4. Robust End of row detection using the forward lookahead window
        points_ahead = self.get_map_points_in_front(self.robot_pose, distance=2.5, width=1.0)
        dist_in_row = math.hypot(self.robot_pose.x - self.row_entry_pose.x, self.robot_pose.y - self.row_entry_pose.y)
        
        near_spline_end = (t_l > self.left_row_spline.t_max - 0.3) or (t_r > self.right_row_spline.t_max - 0.3)
        no_points_ahead = len(points_ahead) < 3
        
        if dist_in_row > 2.0 and near_spline_end and no_points_ahead:
            self.get_logger().info(f"Row end confirmed. Dist: {dist_in_row:.2f}m, Points ahead: {len(points_ahead)}")
            self.exit_start_pose = self.robot_pose
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
        m.ns, m.id, m.type, m.action = "spline_segments", 10000 + id, Marker.SPHERE_LIST, Marker.ADD
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
