#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

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


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class SegmentDebug:
    line_points: np.ndarray
    direction: np.ndarray
    big_rect_center: np.ndarray
    big_rect_length: float
    big_rect_width: float
    point3_rect_center: np.ndarray
    point3_rect_length: float
    point3_rect_width: float
    support_count: int
    end_field_count: int


@dataclass
class RowMarchResult:
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    debug_segments: List[SegmentDebug] = field(default_factory=list)
    current_line_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    ended: bool = False
    end_point: Optional[np.ndarray] = None


@dataclass
class RowMarchModel:
    side: str
    initial_point3: np.ndarray
    initial_direction: np.ndarray
    result: RowMarchResult = field(default_factory=RowMarchResult)


class MissionState(Enum):
    IDLE = 0
    INITIALIZING = 1
    FOLLOW_ROW = 2
    FINISHED = 3


@dataclass
class NavigatorParams:
    cmd_vel_topic: str = "/cmd_vel"
    base_frame: str = "base_link"
    map_frame: str = "map"
    map_topic: str = "/map"

    control_frequency: float = 30.0
    expected_row_width: float = 0.75
    min_lane_width: float = 0.55
    max_lane_width: float = 1.20

    hist_roi_size: float = 5.0
    hist_bin_size: float = 0.05
    hist_peak_min_points: int = 3
    occ_threshold: int = 50

    row_segment_point_count: int = 5
    row_segment_point_spacing: float = 0.3
    row_rectangle_width: float = 0.5
    row_point_window_length: float = 0.3
    row_end_min_points_fields_3_to_5: int = 2
    row_max_march_steps: int = 120
    row_min_fit_points: int = 3

    follow_speed: float = 0.20
    lookahead_distance: float = 1.10
    yaw_kp: float = 0.6
    max_angular_speed: float = 0.40
    min_follow_turn_radius: float = 1.0
    angular_rate_limit: float = 1.2
    target_filter_alpha: float = 0.35
    path_goal_xy_tolerance: float = 0.20

    pattern: str = "1L 2R"
    publish_debug: bool = True


class MaizeNavigator(Node):
    def __init__(self) -> None:
        super().__init__("maize_navigator")
        self.p = self.load_params()
        self.validate_params()

        self.state = MissionState.IDLE
        self.latest_map: Optional[OccupancyGrid] = None
        self.robot_pose: Optional[Pose2D] = None

        self.left_row: Optional[RowMarchModel] = None
        self.right_row: Optional[RowMarchModel] = None
        self.midline: np.ndarray = np.empty((0, 2), dtype=float)
        self.row_end_point: Optional[np.ndarray] = None
        self.last_cmd_angular_z: float = 0.0
        self.last_target_point: Optional[np.ndarray] = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(OccupancyGrid, self.p.map_topic, self.map_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "navigation_markers", 10)

        self.start_srv = self.create_service(Trigger, "start_navigation", self.start_cb)
        self.stop_srv = self.create_service(Trigger, "stop_navigation", self.stop_cb)
        self.timer = self.create_timer(1.0 / self.p.control_frequency, self.control_loop)

        self.get_logger().info("Maize Navigator initialized with rectangle marching row detection")

    def load_params(self) -> NavigatorParams:
        p = NavigatorParams()

        def get_param(name: str, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value

        p.cmd_vel_topic = str(get_param("cmd_vel_topic", p.cmd_vel_topic))
        p.base_frame = str(get_param("base_frame", p.base_frame))
        p.map_frame = str(get_param("map_frame", p.map_frame))
        p.map_topic = str(get_param("map_topic", p.map_topic))
        p.control_frequency = float(get_param("control_frequency", p.control_frequency))

        p.expected_row_width = float(get_param("expected_row_width", p.expected_row_width))
        p.min_lane_width = float(get_param("min_lane_width", p.min_lane_width))
        p.max_lane_width = float(get_param("max_lane_width", p.max_lane_width))

        p.hist_roi_size = float(get_param("hist_roi_size", p.hist_roi_size))
        p.hist_bin_size = float(get_param("hist_bin_size", p.hist_bin_size))
        p.hist_peak_min_points = int(get_param("hist_peak_min_points", p.hist_peak_min_points))
        p.occ_threshold = int(get_param("map_row_occupancy_threshold", p.occ_threshold))

        p.row_segment_point_count = int(get_param("row_segment_point_count", p.row_segment_point_count))
        p.row_segment_point_spacing = float(get_param("row_segment_point_spacing", p.row_segment_point_spacing))
        p.row_rectangle_width = float(get_param("row_rectangle_width", p.row_rectangle_width))
        p.row_point_window_length = float(get_param("row_point_window_length", p.row_point_window_length))
        p.row_end_min_points_fields_3_to_5 = int(
            get_param("row_end_min_points_fields_3_to_5", p.row_end_min_points_fields_3_to_5)
        )
        p.row_max_march_steps = int(get_param("row_max_march_steps", p.row_max_march_steps))
        p.row_min_fit_points = int(get_param("row_min_fit_points", p.row_min_fit_points))

        p.follow_speed = float(get_param("follow_speed", p.follow_speed))
        p.lookahead_distance = float(get_param("lookahead_distance", p.lookahead_distance))
        p.yaw_kp = float(get_param("yaw_kp", p.yaw_kp))
        p.max_angular_speed = float(get_param("follow_max_angular_speed", p.max_angular_speed))
        p.min_follow_turn_radius = float(get_param("min_follow_turn_radius", p.min_follow_turn_radius))
        p.angular_rate_limit = float(get_param("angular_rate_limit", p.angular_rate_limit))
        p.target_filter_alpha = float(get_param("target_filter_alpha", p.target_filter_alpha))
        p.path_goal_xy_tolerance = float(get_param("path_goal_xy_tolerance", p.path_goal_xy_tolerance))

        p.pattern = str(get_param("pattern", p.pattern))
        p.publish_debug = bool(get_param("publish_debug", p.publish_debug))
        return p

    def validate_params(self) -> None:
        if self.p.row_segment_point_count < 5:
            self.get_logger().warn("row_segment_point_count must be at least 5; using 5")
            self.p.row_segment_point_count = 5
        if self.p.row_segment_point_count % 2 == 0:
            self.get_logger().warn("row_segment_point_count must be odd; adding one")
            self.p.row_segment_point_count += 1
        self.p.row_segment_point_spacing = max(0.05, self.p.row_segment_point_spacing)
        self.p.row_rectangle_width = max(0.05, self.p.row_rectangle_width)
        self.p.row_point_window_length = max(0.05, self.p.row_point_window_length)
        self.p.row_max_march_steps = max(1, self.p.row_max_march_steps)
        self.p.min_follow_turn_radius = max(0.1, self.p.min_follow_turn_radius)
        self.p.angular_rate_limit = max(0.01, self.p.angular_rate_limit)
        self.p.target_filter_alpha = float(np.clip(self.p.target_filter_alpha, 0.01, 1.0))

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def start_cb(self, req, res):
        self.left_row = None
        self.right_row = None
        self.midline = np.empty((0, 2), dtype=float)
        self.row_end_point = None
        self.reset_controller_state()
        self.state = MissionState.INITIALIZING
        res.success = True
        res.message = "Navigation started"
        return res

    def stop_cb(self, req, res):
        self.state = MissionState.IDLE
        self.reset_controller_state()
        self.cmd_pub.publish(Twist())
        res.success = True
        res.message = "Navigation stopped"
        return res

    def reset_controller_state(self) -> None:
        self.last_cmd_angular_z = 0.0
        self.last_target_point = None

    def get_robot_pose(self) -> Optional[Pose2D]:
        try:
            t = self.tf_buffer.lookup_transform(self.p.map_frame, self.p.base_frame, rclpy.time.Time())
            q = t.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return Pose2D(float(t.transform.translation.x), float(t.transform.translation.y), float(yaw))
        except Exception:
            return None

    def control_loop(self) -> None:
        self.robot_pose = self.get_robot_pose()
        if self.robot_pose is None or self.latest_map is None:
            return

        if self.state == MissionState.INITIALIZING:
            self.handle_initializing()
        elif self.state == MissionState.FOLLOW_ROW:
            self.handle_follow_row()
        elif self.state == MissionState.FINISHED:
            self.cmd_pub.publish(Twist())

        self.publish_visuals()

    def handle_initializing(self) -> None:
        points = self.get_map_points_in_roi(self.robot_pose, self.p.hist_roi_size)
        if len(points) < 5:
            self.get_logger().info(f"Not enough occupied map points for histogram: {len(points)}", throttle_duration_sec=2.0)
            return

        peak_pair = self.find_start_peak_pair(points, self.robot_pose)
        if peak_pair is None:
            self.get_logger().info("No valid left/right histogram peak pair found", throttle_duration_sec=2.0)
            return

        left_peak, right_peak = peak_pair
        forward = self.yaw_to_vector(self.robot_pose.yaw)

        left_point3 = self.start_point3_from_peak(points, self.robot_pose, left_peak)
        right_point3 = self.start_point3_from_peak(points, self.robot_pose, right_peak)
        self.left_row = RowMarchModel("left", left_point3, forward)
        self.right_row = RowMarchModel("right", right_point3, forward)

        self.recompute_rows()
        if len(self.midline) < 2:
            self.get_logger().info("Initial peaks found, but rectangle marching produced no usable midline", throttle_duration_sec=2.0)
            return

        self.state = MissionState.FOLLOW_ROW
        self.get_logger().info(
            f"Initial rows locked once from histogram peaks: left={left_peak:.3f}, right={right_peak:.3f}, width={left_peak - right_peak:.3f}"
        )

    def handle_follow_row(self) -> None:
        self.recompute_rows()
        if len(self.midline) < 2:
            self.get_logger().warn("No usable midline from rectangle marching; stopping")
            self.cmd_pub.publish(Twist())
            return

        robot_xy = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
        if self.row_end_point is not None:
            dist_to_end = float(np.linalg.norm(robot_xy - self.row_end_point))
            if dist_to_end <= self.p.path_goal_xy_tolerance:
                self.get_logger().info("First row end reached. Stopping; row switching is disabled for this version.")
                self.reset_controller_state()
                self.cmd_pub.publish(Twist())
                self.state = MissionState.FINISHED
                return

        closest_idx = int(np.argmin(np.linalg.norm(self.midline - robot_xy, axis=1)))
        target = self.point_at_polyline_distance(self.midline, closest_idx, self.p.lookahead_distance)
        if target is None:
            target = self.midline[-1]
        self.drive_to_point(np.asarray(target, dtype=float))

    def find_start_peak_pair(self, points: np.ndarray, pose: Pose2D) -> Optional[Tuple[float, float]]:
        forward = self.yaw_to_vector(pose.yaw)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        robot_xy = np.array([pose.x, pose.y], dtype=float)
        rel = points - robot_xy
        local_y = rel @ lateral

        half_roi = 0.5 * self.p.hist_roi_size
        bins = np.arange(-half_roi, half_roi + self.p.hist_bin_size, self.p.hist_bin_size)
        if len(bins) < 3:
            return None
        hist, bin_edges = np.histogram(local_y, bins=bins)

        peaks: List[Tuple[float, int]] = []
        for idx in range(1, len(hist) - 1):
            if hist[idx] >= self.p.hist_peak_min_points and hist[idx] >= hist[idx - 1] and hist[idx] >= hist[idx + 1]:
                center = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])
                peaks.append((float(center), int(hist[idx])))

        if len(peaks) < 2:
            self.get_logger().info(f"Histogram peaks: {[round(p[0], 3) for p in peaks]}", throttle_duration_sec=2.0)
            return None

        best_pair = None
        best_score = float("inf")
        sorted_peaks = sorted(peaks, key=lambda item: item[0])
        for left_idx in range(len(sorted_peaks) - 1):
            right_peak = sorted_peaks[left_idx][0]
            left_peak = sorted_peaks[left_idx + 1][0]
            sep = left_peak - right_peak
            if sep < self.p.min_lane_width or sep > self.p.max_lane_width:
                continue
            count_bonus = 0.01 * (sorted_peaks[left_idx][1] + sorted_peaks[left_idx + 1][1])
            score = abs(sep - self.p.expected_row_width) - count_bonus
            if score < best_score:
                best_score = score
                best_pair = (left_peak, right_peak)

        self.get_logger().info(
            f"Histogram peaks: {[round(p[0], 3) for p in sorted_peaks]}, selected={best_pair}",
            throttle_duration_sec=2.0,
        )
        return best_pair

    def start_point3_from_peak(self, points: np.ndarray, pose: Pose2D, peak_y: float) -> np.ndarray:
        forward = self.yaw_to_vector(pose.yaw)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        robot_xy = np.array([pose.x, pose.y], dtype=float)
        fallback = robot_xy + peak_y * lateral

        if len(points) == 0:
            return fallback

        rel = points - robot_xy
        local_x = rel @ forward
        local_y = rel @ lateral

        peak_half_width = max(self.p.hist_bin_size, 0.5 * self.p.row_rectangle_width)
        band_mask = (local_x >= 0.0) & (np.abs(local_y - peak_y) <= peak_half_width)
        band_points = points[band_mask]
        band_x = local_x[band_mask]

        if len(band_points) == 0:
            self.get_logger().info(
                f"No occupied start points ahead for peak {peak_y:.3f}; using robot-height fallback",
                throttle_duration_sec=2.0,
            )
            return fallback

        first_x = float(np.min(band_x))
        start_window = max(self.p.row_point_window_length, self.p.hist_bin_size)
        start_mask = band_x <= first_x + start_window
        start_points = band_points[start_mask]
        if len(start_points) == 0:
            return fallback

        point3 = np.mean(start_points, axis=0)
        self.get_logger().info(
            f"Initial point 3 for peak {peak_y:.3f}: first_x={first_x:.2f}, n={len(start_points)}",
            throttle_duration_sec=2.0,
        )
        return np.asarray(point3, dtype=float)

    def recompute_rows(self) -> None:
        if self.left_row is None or self.right_row is None:
            return
        map_points = self.get_all_map_points()
        if len(map_points) == 0:
            self.midline = np.empty((0, 2), dtype=float)
            self.row_end_point = None
            return

        self.left_row.result = self.march_row(self.left_row, map_points)
        self.right_row.result = self.march_row(self.right_row, map_points)
        self.midline = self.build_midline(self.left_row.result.points, self.right_row.result.points)

        if self.left_row.result.ended and self.right_row.result.ended:
            left_end = self.left_row.result.end_point
            right_end = self.right_row.result.end_point
            if left_end is not None and right_end is not None:
                self.row_end_point = 0.5 * (left_end + right_end)
            elif len(self.midline) > 0:
                self.row_end_point = self.midline[-1]
        else:
            self.row_end_point = None

    def march_row(self, model: RowMarchModel, map_points: np.ndarray) -> RowMarchResult:
        result = RowMarchResult()
        direction = self.normalize(model.initial_direction)
        line_points = self.build_initial_line_from_point3(model.initial_point3, direction)
        exact_points: List[np.ndarray] = []
        previous_valid_point: Optional[np.ndarray] = None

        for _ in range(self.p.row_max_march_steps):
            search_point3 = line_points[2]
            big_rect_length = self.big_rectangle_length()
            fit_points = self.points_in_oriented_rectangle(
                map_points,
                search_point3,
                direction,
                big_rect_length,
                self.p.row_rectangle_width,
            )

            corrected_direction = direction
            corrected_point3 = np.array(search_point3, copy=True)
            if len(fit_points) >= self.p.row_min_fit_points:
                fit_origin, fitted_direction = self.fit_line(fit_points, direction)
                corrected_direction = fitted_direction
                corrected_point3 = self.project_point_to_line(search_point3, fit_origin, fitted_direction)

            corrected_line_points = self.build_initial_line_from_point3(corrected_point3, corrected_direction)
            field_count = self.count_points_in_fields_3_to_5(map_points, corrected_line_points, corrected_direction)

            local_point3_points = self.points_in_oriented_rectangle(
                map_points,
                corrected_point3,
                corrected_direction,
                self.p.row_point_window_length,
                self.p.row_rectangle_width,
            )

            if len(exact_points) > 0 and field_count <= self.p.row_end_min_points_fields_3_to_5:
                result.ended = True
                result.end_point = previous_valid_point if previous_valid_point is not None else exact_points[-1]
                break

            if len(local_point3_points) > 0:
                row_point = np.mean(local_point3_points, axis=0)
                previous_valid_point = row_point
            else:
                row_point = np.array(corrected_point3, copy=True)

            exact_points.append(row_point)
            result.debug_segments.append(
                SegmentDebug(
                    line_points=np.array(corrected_line_points, copy=True),
                    direction=np.array(corrected_direction, copy=True),
                    big_rect_center=np.array(search_point3, copy=True),
                    big_rect_length=big_rect_length,
                    big_rect_width=self.p.row_rectangle_width,
                    point3_rect_center=np.array(corrected_point3, copy=True),
                    point3_rect_length=self.p.row_point_window_length,
                    point3_rect_width=self.p.row_rectangle_width,
                    support_count=len(fit_points),
                    end_field_count=field_count,
                )
            )

            direction = corrected_direction
            line_points = self.build_next_line_from_old_point3(corrected_point3, direction)

        result.points = np.asarray(exact_points, dtype=float) if exact_points else np.empty((0, 2), dtype=float)
        result.current_line_points = np.asarray(line_points, dtype=float)
        if result.ended and result.end_point is None and len(result.points) > 0:
            result.end_point = result.points[-1]
        return result

    def build_initial_line_from_point3(self, point3: np.ndarray, direction: np.ndarray) -> np.ndarray:
        spacing = self.p.row_segment_point_spacing
        direction = self.normalize(direction)
        return np.array([point3 + (idx - 2) * spacing * direction for idx in range(self.p.row_segment_point_count)], dtype=float)

    def build_next_line_from_old_point3(self, old_point3: np.ndarray, direction: np.ndarray) -> np.ndarray:
        spacing = self.p.row_segment_point_spacing
        direction = self.normalize(direction)
        point2 = np.asarray(old_point3, dtype=float)
        return np.array([point2 + (idx - 1) * spacing * direction for idx in range(self.p.row_segment_point_count)], dtype=float)

    def big_rectangle_length(self) -> float:
        return float(self.p.row_segment_point_count) * self.p.row_segment_point_spacing

    def count_points_in_fields_3_to_5(self, map_points: np.ndarray, line_points: np.ndarray, direction: np.ndarray) -> int:
        total = 0
        for idx in (2, 3, 4):
            total += len(
                self.points_in_oriented_rectangle(
                    map_points,
                    line_points[idx],
                    direction,
                    self.p.row_point_window_length,
                    self.p.row_rectangle_width,
                )
            )
        return total

    def fit_line(self, points: np.ndarray, fallback_direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fallback_direction = self.normalize(fallback_direction)
        fallback_origin = np.mean(points, axis=0) if len(points) > 0 else np.array([0.0, 0.0], dtype=float)
        if len(points) < 2:
            return fallback_origin, fallback_direction

        fit_points = points
        if len(points) > 2:
            best_inliers = np.ones(len(points), dtype=bool)
            best_count = -1
            threshold = max(0.05, 0.18 * self.p.row_rectangle_width)
            rng = np.random.default_rng(42)
            iterations = min(64, max(16, len(points) * 2))

            for _ in range(iterations):
                idx_a, idx_b = rng.choice(len(points), size=2, replace=False)
                a = points[idx_a]
                b = points[idx_b]
                candidate = b - a
                norm = float(np.linalg.norm(candidate))
                if norm < 1e-9:
                    continue
                candidate = candidate / norm
                if np.dot(candidate, fallback_direction) < 0.0:
                    candidate = -candidate
                perp = np.array([-candidate[1], candidate[0]], dtype=float)
                distances = np.abs((points - a) @ perp)
                inliers = distances <= threshold
                count = int(np.sum(inliers))
                if count > best_count:
                    best_count = count
                    best_inliers = inliers

            if best_count >= self.p.row_min_fit_points:
                fit_points = points[best_inliers]

        centered = fit_points - np.mean(fit_points, axis=0)
        cov = np.cov(centered.T)
        if np.ndim(cov) != 2:
            return np.mean(fit_points, axis=0), fallback_direction

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        direction = np.real(eigenvectors[:, int(np.argmax(eigenvalues))])
        direction = self.normalize(direction)
        if np.dot(direction, fallback_direction) < 0.0:
            direction = -direction

        # Light damping keeps the marching direction from swinging on sparse/noisy cells.
        direction = self.normalize(0.75 * fallback_direction + 0.25 * direction)
        origin = np.mean(fit_points, axis=0)
        return np.asarray(origin, dtype=float), direction

    def project_point_to_line(self, point: np.ndarray, line_origin: np.ndarray, line_direction: np.ndarray) -> np.ndarray:
        line_direction = self.normalize(line_direction)
        rel = np.asarray(point, dtype=float) - np.asarray(line_origin, dtype=float)
        return np.asarray(line_origin, dtype=float) + float(rel @ line_direction) * line_direction

    def points_in_oriented_rectangle(
        self,
        points: np.ndarray,
        center: np.ndarray,
        direction: np.ndarray,
        length: float,
        width: float,
    ) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=float)
        direction = self.normalize(direction)
        perp = np.array([-direction[1], direction[0]], dtype=float)
        rel = points - np.asarray(center, dtype=float)
        along = rel @ direction
        lateral = rel @ perp
        mask = (np.abs(along) <= 0.5 * length) & (np.abs(lateral) <= 0.5 * width)
        return points[mask]

    def build_midline(self, left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
        if len(left_points) == 0 or len(right_points) == 0:
            return np.empty((0, 2), dtype=float)
        count = min(len(left_points), len(right_points))
        return 0.5 * (left_points[:count] + right_points[:count])

    def point_at_polyline_distance(self, polyline: np.ndarray, start_idx: int, distance_ahead: float) -> Optional[np.ndarray]:
        if len(polyline) == 0:
            return None
        if len(polyline) == 1:
            return polyline[0]
        start_idx = int(np.clip(start_idx, 0, len(polyline) - 1))
        travelled = 0.0
        for idx in range(start_idx, len(polyline) - 1):
            segment = polyline[idx + 1] - polyline[idx]
            seg_len = float(np.linalg.norm(segment))
            if seg_len < 1e-9:
                continue
            if travelled + seg_len >= distance_ahead:
                ratio = (distance_ahead - travelled) / seg_len
                return polyline[idx] + ratio * segment
            travelled += seg_len
        return polyline[-1]

    def drive_to_point(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float)
        if self.last_target_point is None:
            filtered_target = target
        else:
            alpha = self.p.target_filter_alpha
            filtered_target = (1.0 - alpha) * self.last_target_point + alpha * target
        self.last_target_point = np.asarray(filtered_target, dtype=float)

        dx = float(filtered_target[0] - self.robot_pose.x)
        dy = float(filtered_target[1] - self.robot_pose.y)
        target_yaw = math.atan2(dy, dx)
        yaw_error = wrap_to_pi(target_yaw - self.robot_pose.yaw)

        cmd = Twist()
        speed_factor = float(np.clip(1.0 - abs(yaw_error) / 0.7, 0.25, 1.0))
        cmd.linear.x = self.p.follow_speed * speed_factor
        radius_limited_angular = abs(cmd.linear.x) / self.p.min_follow_turn_radius
        max_angular = min(self.p.max_angular_speed, radius_limited_angular)
        angular_raw = float(np.clip(self.p.yaw_kp * yaw_error, -max_angular, max_angular))

        dt = 1.0 / max(1e-6, self.p.control_frequency)
        max_delta = self.p.angular_rate_limit * dt
        angular_limited = float(np.clip(
            angular_raw,
            self.last_cmd_angular_z - max_delta,
            self.last_cmd_angular_z + max_delta,
        ))
        self.last_cmd_angular_z = angular_limited
        cmd.angular.z = angular_limited
        self.cmd_pub.publish(cmd)

    def get_map_points_in_roi(self, pose: Pose2D, size: float) -> np.ndarray:
        points = self.get_all_map_points()
        if len(points) == 0:
            return points
        center = np.array([pose.x, pose.y], dtype=float)
        half = 0.5 * float(size)
        rel = points - center
        mask = (np.abs(rel[:, 0]) <= half) & (np.abs(rel[:, 1]) <= half)
        return points[mask]

    def get_all_map_points(self) -> np.ndarray:
        if self.latest_map is None:
            return np.empty((0, 2), dtype=float)

        info = self.latest_map.info
        grid = np.asarray(self.latest_map.data, dtype=np.int16).reshape((info.height, info.width))
        occ_y, occ_x = np.where(grid >= self.p.occ_threshold)
        if len(occ_x) == 0:
            return np.empty((0, 2), dtype=float)

        world_x = (occ_x.astype(float) + 0.5) * info.resolution + info.origin.position.x
        world_y = (occ_y.astype(float) + 0.5) * info.resolution + info.origin.position.y
        return np.column_stack((world_x, world_y))

    def publish_visuals(self) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear_marker = Marker(header=Header(frame_id=self.p.map_frame, stamp=stamp))
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        marker_id = 1
        if self.left_row is not None:
            marker_id = self.add_row_markers(markers, marker_id, self.left_row, (0.1, 0.95, 0.2, 1.0), stamp)
        if self.right_row is not None:
            marker_id = self.add_row_markers(markers, marker_id, self.right_row, (1.0, 0.55, 0.05, 1.0), stamp)
        if len(self.midline) > 0:
            markers.markers.append(self.create_line_marker("march_midline", marker_id, self.midline, (0.15, 0.45, 1.0, 1.0), 0.055, stamp))
            marker_id += 1
        if self.last_target_point is not None:
            markers.markers.append(self.create_sphere_marker("filtered_target_point", marker_id, self.last_target_point, (1.0, 0.05, 1.0, 1.0), 0.18, stamp))
            marker_id += 1
            if self.robot_pose is not None:
                robot_point = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
                target_line = np.vstack((robot_point, self.last_target_point))
                markers.markers.append(self.create_line_marker("robot_to_filtered_target", marker_id, target_line, (1.0, 0.05, 1.0, 0.75), 0.025, stamp))
                marker_id += 1
        if self.row_end_point is not None:
            markers.markers.append(self.create_sphere_marker("row_end", marker_id, self.row_end_point, (1.0, 0.0, 0.0, 1.0), 0.22, stamp))

        self.marker_pub.publish(markers)

    def add_row_markers(
        self,
        markers: MarkerArray,
        start_id: int,
        model: RowMarchModel,
        color: Tuple[float, float, float, float],
        stamp,
    ) -> int:
        marker_id = start_id
        result = model.result
        if len(result.points) > 0:
            markers.markers.append(self.create_line_marker(f"{model.side}_row_curve", marker_id, result.points, color, 0.045, stamp))
            marker_id += 1
            markers.markers.append(self.create_points_marker(f"{model.side}_row_points", marker_id, result.points, color, 0.11, stamp))
            marker_id += 1

        if len(result.current_line_points) > 0:
            markers.markers.append(
                self.create_points_marker(
                    f"{model.side}_current_five_points",
                    marker_id,
                    result.current_line_points,
                    (1.0, 1.0, 0.1, 0.95),
                    0.08,
                    stamp,
                )
            )
            marker_id += 1
            markers.markers.append(
                self.create_line_marker(
                    f"{model.side}_current_line",
                    marker_id,
                    result.current_line_points,
                    (1.0, 1.0, 0.1, 0.85),
                    0.025,
                    stamp,
                )
            )
            marker_id += 1
            for idx, point in enumerate(result.current_line_points[: self.p.row_segment_point_count]):
                markers.markers.append(
                    self.create_text_marker(
                        f"{model.side}_current_point_numbers",
                        marker_id,
                        point,
                        str(idx + 1),
                        (1.0, 1.0, 1.0, 0.95),
                        stamp,
                    )
                )
                marker_id += 1

        if self.p.publish_debug:
            for segment in result.debug_segments[-25:]:
                markers.markers.append(
                    self.create_rectangle_marker(
                        f"{model.side}_fit_rectangles",
                        marker_id,
                        segment.big_rect_center,
                        segment.direction,
                        segment.big_rect_length,
                        segment.big_rect_width,
                        (0.05, 0.75, 1.0, 0.34),
                        stamp,
                    )
                )
                marker_id += 1
                markers.markers.append(
                    self.create_rectangle_marker(
                        f"{model.side}_point3_rectangles",
                        marker_id,
                        segment.point3_rect_center,
                        segment.direction,
                        segment.point3_rect_length,
                        segment.point3_rect_width,
                        (1.0, 0.1, 0.8, 0.48),
                        stamp,
                    )
                )
                marker_id += 1

        if result.end_point is not None:
            markers.markers.append(self.create_sphere_marker(f"{model.side}_end_point", marker_id, result.end_point, (1.0, 0.0, 0.0, 1.0), 0.18, stamp))
            marker_id += 1
        return marker_id

    def create_line_marker(
        self,
        namespace: str,
        marker_id: int,
        points: np.ndarray,
        color: Tuple[float, float, float, float],
        scale: float,
        stamp,
    ) -> Marker:
        marker = Marker(header=Header(frame_id=self.p.map_frame, stamp=stamp))
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = float(scale)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        for point in points:
            marker.points.append(Point(x=float(point[0]), y=float(point[1]), z=0.0))
        return marker

    def create_points_marker(
        self,
        namespace: str,
        marker_id: int,
        points: np.ndarray,
        color: Tuple[float, float, float, float],
        scale: float,
        stamp,
    ) -> Marker:
        marker = Marker(header=Header(frame_id=self.p.map_frame, stamp=stamp))
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = float(scale)
        marker.scale.y = float(scale)
        marker.scale.z = float(scale)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        for point in points:
            marker.points.append(Point(x=float(point[0]), y=float(point[1]), z=0.0))
        return marker

    def create_text_marker(
        self,
        namespace: str,
        marker_id: int,
        point: np.ndarray,
        text: str,
        color: Tuple[float, float, float, float],
        stamp,
    ) -> Marker:
        marker = Marker(header=Header(frame_id=self.p.map_frame, stamp=stamp))
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.18
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.25
        marker.pose.orientation.w = 1.0
        marker.text = text
        return marker

    def create_sphere_marker(
        self,
        namespace: str,
        marker_id: int,
        point: np.ndarray,
        color: Tuple[float, float, float, float],
        scale: float,
        stamp,
    ) -> Marker:
        marker = Marker(header=Header(frame_id=self.p.map_frame, stamp=stamp))
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = float(scale)
        marker.scale.y = float(scale)
        marker.scale.z = float(scale)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        return marker

    def create_rectangle_marker(
        self,
        namespace: str,
        marker_id: int,
        center: np.ndarray,
        direction: np.ndarray,
        length: float,
        width: float,
        color: Tuple[float, float, float, float],
        stamp,
    ) -> Marker:
        marker = Marker(header=Header(frame_id=self.p.map_frame, stamp=stamp))
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.025
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

        direction = self.normalize(direction)
        perp = np.array([-direction[1], direction[0]], dtype=float)
        center = np.asarray(center, dtype=float)
        half_length = 0.5 * float(length)
        half_width = 0.5 * float(width)
        corners = [
            center - half_length * direction - half_width * perp,
            center + half_length * direction - half_width * perp,
            center + half_length * direction + half_width * perp,
            center - half_length * direction + half_width * perp,
            center - half_length * direction - half_width * perp,
        ]
        for corner in corners:
            marker.points.append(Point(x=float(corner[0]), y=float(corner[1]), z=0.0))
        return marker

    def yaw_to_vector(self, yaw: float) -> np.ndarray:
        return np.array([math.cos(yaw), math.sin(yaw)], dtype=float)

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return vector / norm


def main(args=None) -> None:
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
