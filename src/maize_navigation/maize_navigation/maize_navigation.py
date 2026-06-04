#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist
from maize_navigation_interfaces.srv import StartNavigation
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
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
    point_directions: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    frozen_count: int = 0
    debug_segments: List[SegmentDebug] = field(default_factory=list)
    current_line_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    ended: bool = False
    end_point: Optional[np.ndarray] = None
    end_direction: Optional[np.ndarray] = None


@dataclass
class RowMarchModel:
    side: str
    initial_point1: np.ndarray
    initial_point2: np.ndarray
    initial_direction: np.ndarray
    row_number: int
    result: RowMarchResult = field(default_factory=RowMarchResult)
    frozen_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    frozen_directions: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))


@dataclass
class PatternStep:
    lane_shift: int
    direction: str


@dataclass
class EntrancePeak:
    lateral: float
    point: np.ndarray
    row_number: Optional[int] = None
    selected: bool = False


@dataclass
class LaserLineFit:
    valid: bool = False
    slope: float = 0.0
    intercept: float = 0.0
    inliers: int = 0
    visible_length: float = 0.0


@dataclass
class LaserFollowResult:
    valid: bool = False
    left_line: LaserLineFit = field(default_factory=LaserLineFit)
    right_line: LaserLineFit = field(default_factory=LaserLineFit)
    center_slope: float = 0.0
    center_intercept: float = 0.0
    confidence: float = 0.0
    weight: float = 0.0
    target_base: Optional[np.ndarray] = None
    reason: str = ""
    roi_centers: List[np.ndarray] = field(default_factory=list)
    roi_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=float))


class MissionState(Enum):
    IDLE = 0
    INITIALIZING = 1
    FOLLOW_ROW = 2
    FIND_NEXT_ROW_ENTRANCE = 3
    FINISHED = 4


@dataclass
class NavigatorParams:
    cmd_vel_topic: str = "/cmd_vel"
    base_frame: str = "base_link"
    map_frame: str = "map"
    map_topic: str = "/map"
    scan_topic: str = "/sensors/merged_scan"

    control_frequency: float = 30.0
    expected_row_width: float = 0.75
    min_lane_width: float = 0.55
    max_lane_width: float = 1.20

    hist_roi_size: float = 5.0
    hist_roi_depth: float = 5.0
    hist_roi_width: float = 5.0
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
    row_freeze_behind_distance: float = 0.0

    laser_follow_enabled: bool = True
    laser_scan_timeout: float = 0.50
    laser_roi_x_min: float = 0.25
    laser_roi_x_max: float = 1.80
    laser_roi_length: float = 1.55
    laser_roi_width: float = 0.32
    laser_roi_center_offset_limit: float = 0.25
    laser_ransac_iterations: int = 80
    laser_ransac_distance: float = 0.08
    laser_min_inliers: int = 5
    laser_min_visible_length: float = 0.35
    laser_max_abs_line_slope: float = 0.9
    laser_max_angle_to_map: float = 0.45
    laser_max_center_offset: float = 0.40
    laser_min_confidence: float = 0.25
    laser_full_confidence: float = 0.85
    laser_max_weight_both_sides: float = 0.80
    laser_max_weight_one_side: float = 0.40
    laser_tracker_alpha: float = 0.25

    follow_speed: float = 0.20
    slow_speed: float = 0.12
    lookahead_distance: float = 1.10
    turn_lookahead_distance: float = 0.45
    lookahead_curvature_gain: float = 1.5
    yaw_kp: float = 0.6
    pure_pursuit_gain: float = 1.0
    curve_speed_reduction_gain: float = 1.0
    max_angular_speed: float = 0.40
    min_follow_turn_radius: float = 0.37
    angular_rate_limit: float = 1.2
    target_filter_alpha: float = 0.35
    row_exit_extension_distance: float = 0.30
    row_end_goal_outward_distance: float = 0.30
    path_goal_xy_tolerance: float = 0.20
    maneuver_goal_xy_tolerance: float = 0.35
    maneuver_goal_yaw_tolerance: float = 0.45
    maneuver_heading_lookahead_distance: float = 0.70
    maneuver_speed: float = 0.16
    maneuver_slow_speed: float = 0.10
    maneuver_lookahead_distance: float = 0.80
    maneuver_route_spacing: float = 0.08
    maneuver_corner_radius: float = 0.50
    maneuver_entry_extension_distance: float = 0.80
    maneuver_max_route_deviation: float = 0.70
    maneuver_entry_lateral_tolerance: float = 0.25
    maneuver_entry_yaw_tolerance: float = 0.45
    row_map_output_directory: str = "~/.ros/maize_navigation"

    pattern: str = "1L 2R"
    starting_lane_number: int = 1
    row_numbers_increase_to: str = "left"
    publish_debug: bool = True


class LaserRowFollower:
    def __init__(self, params: NavigatorParams) -> None:
        self.p = params
        self.filtered_confidence = 0.0

    def reset(self) -> None:
        self.filtered_confidence = 0.0

    def reject(self, result: LaserFollowResult, reason: str) -> LaserFollowResult:
        self.filtered_confidence *= 1.0 - self.p.laser_tracker_alpha
        result.confidence = self.filtered_confidence
        result.reason = reason
        return result

    def process_scan(
        self,
        scan: Optional[LaserScan],
        map_slope: Optional[float],
        map_target_base: np.ndarray,
    ) -> LaserFollowResult:
        if scan is None:
            return self.reject(LaserFollowResult(), "no scan")

        points = self.scan_to_points(scan)
        roi_centers, roi_direction = self.build_rois(map_slope, map_target_base)
        left_line = self.fit_line_in_roi(points, roi_centers[0], roi_direction)
        right_line = self.fit_line_in_roi(points, roi_centers[1], roi_direction)
        result = LaserFollowResult(
            left_line=left_line,
            right_line=right_line,
            roi_centers=roi_centers,
            roi_direction=roi_direction,
        )
        target_x = float(np.clip(map_target_base[0], self.p.laser_roi_x_min, self.p.laser_roi_x_max))

        if left_line.valid and right_line.valid:
            left_y = left_line.slope * target_x + left_line.intercept
            right_y = right_line.slope * target_x + right_line.intercept
            width = left_y - right_y
            if not self.p.min_lane_width <= width <= self.p.max_lane_width:
                return self.reject(result, "invalid lane width")
            if abs(left_line.slope - right_line.slope) > 0.35:
                return self.reject(result, "side lines not parallel")
            result.center_slope = 0.5 * (left_line.slope + right_line.slope)
            result.center_intercept = 0.5 * (left_line.intercept + right_line.intercept)
            raw_confidence = 1.0
            max_weight = self.p.laser_max_weight_both_sides
        elif left_line.valid:
            result.center_slope = left_line.slope
            result.center_intercept = left_line.intercept - 0.5 * self.p.expected_row_width
            raw_confidence = 0.60
            max_weight = self.p.laser_max_weight_one_side
        elif right_line.valid:
            result.center_slope = right_line.slope
            result.center_intercept = right_line.intercept + 0.5 * self.p.expected_row_width
            raw_confidence = 0.60
            max_weight = self.p.laser_max_weight_one_side
        else:
            return self.reject(result, "no valid side line")

        laser_target_y = result.center_slope * target_x + result.center_intercept
        angle_error = 0.0 if map_slope is None else abs(wrap_to_pi(math.atan(result.center_slope) - math.atan(map_slope)))
        if map_slope is not None and angle_error > self.p.laser_max_angle_to_map:
            return self.reject(result, "angle differs from map")
        if abs(laser_target_y - float(map_target_base[1])) > self.p.laser_max_center_offset:
            return self.reject(result, "center differs from map")

        alpha = self.p.laser_tracker_alpha
        self.filtered_confidence = (1.0 - alpha) * self.filtered_confidence + alpha * raw_confidence
        confidence_range = max(1e-6, self.p.laser_full_confidence - self.p.laser_min_confidence)
        scale = (self.filtered_confidence - self.p.laser_min_confidence) / confidence_range
        result.valid = True
        result.confidence = self.filtered_confidence
        result.weight = float(np.clip(scale, 0.0, 1.0)) * max_weight
        result.target_base = np.array([target_x, laser_target_y], dtype=float)
        result.reason = "ok"
        return result

    def scan_to_points(self, scan: LaserScan) -> np.ndarray:
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges), dtype=float) * scan.angle_increment
        mask = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        return np.column_stack((ranges[mask] * np.cos(angles[mask]), ranges[mask] * np.sin(angles[mask])))

    def build_rois(
        self,
        map_slope: Optional[float],
        map_target_base: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        slope = 0.0 if map_slope is None else float(map_slope)
        angle = math.atan(slope)
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        perp = np.array([-direction[1], direction[0]], dtype=float)
        map_center_intercept = float(map_target_base[1] - slope * map_target_base[0])
        center_offset = float(np.clip(
            map_center_intercept,
            -self.p.laser_roi_center_offset_limit,
            self.p.laser_roi_center_offset_limit,
        ))
        along_center = self.p.laser_roi_x_min + 0.5 * self.p.laser_roi_length
        lane_center = np.array([0.0, center_offset], dtype=float) + along_center * direction
        half_lane = 0.5 * self.p.expected_row_width
        return [lane_center + half_lane * perp, lane_center - half_lane * perp], direction

    def points_in_oriented_roi(
        self,
        points: np.ndarray,
        center: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=float)
        perp = np.array([-direction[1], direction[0]], dtype=float)
        rel = points - np.asarray(center, dtype=float)
        mask = (
            (np.abs(rel @ direction) <= 0.5 * self.p.laser_roi_length)
            & (np.abs(rel @ perp) <= 0.5 * self.p.laser_roi_width)
        )
        return points[mask]

    def fit_line_in_roi(
        self,
        points: np.ndarray,
        center: np.ndarray,
        direction: np.ndarray,
    ) -> LaserLineFit:
        roi_points = self.points_in_oriented_roi(points, center, direction)
        perp = np.array([-direction[1], direction[0]], dtype=float)
        rel = roi_points - np.asarray(center, dtype=float)
        local_points = np.column_stack((rel @ direction, rel @ perp))
        local_fit = self.fit_line_ransac(local_points)
        if not local_fit.valid:
            return local_fit
        local_endpoints = np.array(
            [
                [-0.5 * self.p.laser_roi_length, local_fit.intercept - 0.5 * self.p.laser_roi_length * local_fit.slope],
                [0.5 * self.p.laser_roi_length, local_fit.intercept + 0.5 * self.p.laser_roi_length * local_fit.slope],
            ],
            dtype=float,
        )
        base_endpoints = center + np.outer(local_endpoints[:, 0], direction) + np.outer(local_endpoints[:, 1], perp)
        dx = float(base_endpoints[1, 0] - base_endpoints[0, 0])
        if abs(dx) < 1e-6:
            return LaserLineFit()
        slope = float((base_endpoints[1, 1] - base_endpoints[0, 1]) / dx)
        intercept = float(base_endpoints[0, 1] - slope * base_endpoints[0, 0])
        return LaserLineFit(True, slope, intercept, local_fit.inliers, local_fit.visible_length)

    def fit_line_ransac(self, points: np.ndarray) -> LaserLineFit:
        if len(points) < self.p.laser_min_inliers:
            return LaserLineFit()

        best_inliers = np.zeros(len(points), dtype=bool)
        rng = np.random.default_rng(42)
        for _ in range(self.p.laser_ransac_iterations):
            first, second = points[rng.choice(len(points), size=2, replace=False)]
            dx = float(second[0] - first[0])
            if abs(dx) < 1e-6:
                continue
            slope = float((second[1] - first[1]) / dx)
            if abs(slope) > self.p.laser_max_abs_line_slope:
                continue
            intercept = float(first[1] - slope * first[0])
            distances = np.abs(slope * points[:, 0] - points[:, 1] + intercept) / math.sqrt(slope * slope + 1.0)
            inliers = distances <= self.p.laser_ransac_distance
            if int(np.sum(inliers)) > int(np.sum(best_inliers)):
                best_inliers = inliers

        if int(np.sum(best_inliers)) < self.p.laser_min_inliers:
            return LaserLineFit()
        inlier_points = points[best_inliers]
        slope, intercept = np.polyfit(inlier_points[:, 0], inlier_points[:, 1], 1)
        visible_length = float(np.max(inlier_points[:, 0]) - np.min(inlier_points[:, 0]))
        valid = visible_length >= self.p.laser_min_visible_length and abs(slope) <= self.p.laser_max_abs_line_slope
        return LaserLineFit(valid, float(slope), float(intercept), int(np.sum(best_inliers)), visible_length)


class MaizeNavigator(Node):
    def __init__(self) -> None:
        super().__init__("maize_navigator")
        self.p = self.load_params()
        self.validate_params()

        self.state = MissionState.IDLE
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_received_ns: Optional[int] = None
        self.robot_pose: Optional[Pose2D] = None
        self.laser_follower = LaserRowFollower(self.p)
        self.laser_follow_result = LaserFollowResult(reason="not initialized")
        self.fused_target_point: Optional[np.ndarray] = None

        self.left_row: Optional[RowMarchModel] = None
        self.right_row: Optional[RowMarchModel] = None
        self.midline: np.ndarray = np.empty((0, 2), dtype=float)
        self.plant_row_end_point: Optional[np.ndarray] = None
        self.row_exit_goal: Optional[np.ndarray] = None
        self.row_exit_heading_goal: Optional[np.ndarray] = None
        self.row_end_direction: Optional[np.ndarray] = None
        self.last_cmd_angular_z: float = 0.0
        self.last_target_point: Optional[np.ndarray] = None
        self.current_lookahead_distance: float = self.p.lookahead_distance
        self.current_lookahead_curvature: float = 0.0
        self.initial_forward_direction: Optional[np.ndarray] = None
        self.row_number_increase_direction: Optional[np.ndarray] = None
        self.pattern_steps: List[PatternStep] = self.parse_pattern(self.p.pattern)
        self.pattern_index: int = 0
        self.finish_after_current_row: bool = False
        self.entrance_hist_center: Optional[np.ndarray] = None
        self.entrance_hist_direction: Optional[np.ndarray] = None
        self.entrance_hist_peaks: List[EntrancePeak] = []
        self.entrance_target: Optional[np.ndarray] = None
        self.entrance_target_direction: Optional[np.ndarray] = None
        self.entrance_waypoints: List[np.ndarray] = []
        self.entrance_heading_goal: Optional[np.ndarray] = None
        self.entrance_follow_path: np.ndarray = np.empty((0, 2), dtype=float)
        self.entrance_route_support: np.ndarray = np.empty((0, 2), dtype=float)
        self.entrance_route: np.ndarray = np.empty((0, 2), dtype=float)
        self.entrance_route_progress_index: int = 0
        self.entrance_route_projection: Optional[np.ndarray] = None
        self.entrance_route_target: Optional[np.ndarray] = None
        self.entrance_route_remaining_distance: float = 0.0
        self.entrance_route_provisional: bool = False
        self.entrance_active_step: Optional[PatternStep] = None
        self.entrance_traverse_outward: Optional[float] = None
        self.pending_target_peaks: Optional[Tuple[EntrancePeak, EntrancePeak]] = None
        self.stored_rows: Dict[int, np.ndarray] = {}
        self.row_end_directions_by_side: Dict[str, List[np.ndarray]] = {"forward": [], "backward": []}
        self.row_map_exported: bool = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(OccupancyGrid, self.p.map_topic, self.map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.p.scan_topic, self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.p.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "navigation_markers", 10)

        self.start_srv = self.create_service(StartNavigation, "start_navigation", self.start_cb)
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
        p.scan_topic = str(get_param("scan_topic", p.scan_topic))
        p.control_frequency = float(get_param("control_frequency", p.control_frequency))

        p.expected_row_width = float(get_param("expected_row_width", p.expected_row_width))
        p.min_lane_width = float(get_param("min_lane_width", p.min_lane_width))
        p.max_lane_width = float(get_param("max_lane_width", p.max_lane_width))

        p.hist_roi_size = float(get_param("hist_roi_size", p.hist_roi_size))
        p.hist_roi_depth = float(get_param("hist_roi_depth", p.hist_roi_size))
        p.hist_roi_width = float(get_param("hist_roi_width", p.hist_roi_size))
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
        p.row_freeze_behind_distance = float(get_param("row_freeze_behind_distance", p.row_freeze_behind_distance))

        p.laser_follow_enabled = bool(get_param("laser_follow_enabled", p.laser_follow_enabled))
        p.laser_scan_timeout = float(get_param("laser_scan_timeout", p.laser_scan_timeout))
        p.laser_roi_x_min = float(get_param("laser_roi_x_min", p.laser_roi_x_min))
        p.laser_roi_x_max = float(get_param("laser_roi_x_max", p.laser_roi_x_max))
        p.laser_roi_length = float(get_param("laser_roi_length", p.laser_roi_length))
        p.laser_roi_width = float(get_param("laser_roi_width", p.laser_roi_width))
        p.laser_roi_center_offset_limit = float(
            get_param("laser_roi_center_offset_limit", p.laser_roi_center_offset_limit)
        )
        p.laser_ransac_iterations = int(get_param("laser_ransac_iterations", p.laser_ransac_iterations))
        p.laser_ransac_distance = float(get_param("laser_ransac_distance", p.laser_ransac_distance))
        p.laser_min_inliers = int(get_param("laser_min_inliers", p.laser_min_inliers))
        p.laser_min_visible_length = float(get_param("laser_min_visible_length", p.laser_min_visible_length))
        p.laser_max_abs_line_slope = float(get_param("laser_max_abs_line_slope", p.laser_max_abs_line_slope))
        p.laser_max_angle_to_map = float(get_param("laser_max_angle_to_map", p.laser_max_angle_to_map))
        p.laser_max_center_offset = float(get_param("laser_max_center_offset", p.laser_max_center_offset))
        p.laser_min_confidence = float(get_param("laser_min_confidence", p.laser_min_confidence))
        p.laser_full_confidence = float(get_param("laser_full_confidence", p.laser_full_confidence))
        p.laser_max_weight_both_sides = float(
            get_param("laser_max_weight_both_sides", p.laser_max_weight_both_sides)
        )
        p.laser_max_weight_one_side = float(
            get_param("laser_max_weight_one_side", p.laser_max_weight_one_side)
        )
        p.laser_tracker_alpha = float(get_param("laser_tracker_alpha", p.laser_tracker_alpha))

        p.follow_speed = float(get_param("follow_speed", p.follow_speed))
        p.slow_speed = float(get_param("slow_speed", p.slow_speed))
        p.lookahead_distance = float(get_param("lookahead_distance", p.lookahead_distance))
        p.turn_lookahead_distance = float(get_param("turn_lookahead_distance", p.turn_lookahead_distance))
        p.lookahead_curvature_gain = float(get_param("lookahead_curvature_gain", p.lookahead_curvature_gain))
        p.yaw_kp = float(get_param("yaw_kp", p.yaw_kp))
        p.pure_pursuit_gain = float(get_param("pure_pursuit_gain", p.pure_pursuit_gain))
        p.curve_speed_reduction_gain = float(get_param("curve_speed_reduction_gain", p.curve_speed_reduction_gain))
        p.max_angular_speed = float(get_param("follow_max_angular_speed", p.max_angular_speed))
        p.min_follow_turn_radius = float(get_param("min_follow_turn_radius", p.min_follow_turn_radius))
        p.angular_rate_limit = float(get_param("angular_rate_limit", p.angular_rate_limit))
        p.target_filter_alpha = float(get_param("target_filter_alpha", p.target_filter_alpha))
        p.row_exit_extension_distance = float(get_param("row_exit_extension_distance", p.row_exit_extension_distance))
        p.row_end_goal_outward_distance = float(
            get_param("row_end_goal_outward_distance", p.row_end_goal_outward_distance)
        )
        p.path_goal_xy_tolerance = float(get_param("path_goal_xy_tolerance", p.path_goal_xy_tolerance))
        p.maneuver_goal_xy_tolerance = float(get_param("maneuver_goal_xy_tolerance", p.maneuver_goal_xy_tolerance))
        p.maneuver_goal_yaw_tolerance = float(get_param("maneuver_goal_yaw_tolerance", p.maneuver_goal_yaw_tolerance))
        p.maneuver_heading_lookahead_distance = float(
            get_param("maneuver_heading_lookahead_distance", p.maneuver_heading_lookahead_distance)
        )
        p.maneuver_speed = float(get_param("maneuver_speed", p.maneuver_speed))
        p.maneuver_slow_speed = float(get_param("maneuver_slow_speed", p.maneuver_slow_speed))
        p.maneuver_lookahead_distance = float(get_param("maneuver_lookahead_distance", p.maneuver_lookahead_distance))
        p.maneuver_route_spacing = float(get_param("maneuver_route_spacing", p.maneuver_route_spacing))
        p.maneuver_corner_radius = float(get_param("maneuver_corner_radius", p.maneuver_corner_radius))
        p.maneuver_entry_extension_distance = float(
            get_param("maneuver_entry_extension_distance", p.maneuver_entry_extension_distance)
        )
        p.maneuver_max_route_deviation = float(
            get_param("maneuver_max_route_deviation", p.maneuver_max_route_deviation)
        )
        p.maneuver_entry_lateral_tolerance = float(
            get_param("maneuver_entry_lateral_tolerance", p.maneuver_entry_lateral_tolerance)
        )
        p.maneuver_entry_yaw_tolerance = float(
            get_param("maneuver_entry_yaw_tolerance", p.maneuver_entry_yaw_tolerance)
        )
        p.row_map_output_directory = str(get_param("row_map_output_directory", p.row_map_output_directory))

        p.pattern = str(get_param("pattern", p.pattern))
        p.starting_lane_number = int(get_param("starting_lane_number", p.starting_lane_number))
        p.row_numbers_increase_to = str(get_param("row_numbers_increase_to", p.row_numbers_increase_to))
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
        self.p.hist_roi_depth = max(self.p.hist_bin_size, self.p.hist_roi_depth)
        self.p.hist_roi_width = max(self.p.hist_bin_size, self.p.hist_roi_width)
        self.p.row_rectangle_width = max(0.05, self.p.row_rectangle_width)
        self.p.row_point_window_length = max(0.05, self.p.row_point_window_length)
        self.p.row_max_march_steps = max(1, self.p.row_max_march_steps)
        self.p.row_freeze_behind_distance = max(0.0, self.p.row_freeze_behind_distance)
        self.p.laser_scan_timeout = max(0.05, self.p.laser_scan_timeout)
        self.p.laser_roi_length = max(0.05, self.p.laser_roi_length)
        self.p.laser_roi_x_max = self.p.laser_roi_x_min + self.p.laser_roi_length
        self.p.laser_roi_width = max(0.05, self.p.laser_roi_width)
        self.p.laser_roi_center_offset_limit = max(0.0, self.p.laser_roi_center_offset_limit)
        self.p.laser_ransac_iterations = max(1, self.p.laser_ransac_iterations)
        self.p.laser_ransac_distance = max(0.01, self.p.laser_ransac_distance)
        self.p.laser_min_inliers = max(2, self.p.laser_min_inliers)
        self.p.laser_min_visible_length = max(0.05, self.p.laser_min_visible_length)
        self.p.laser_max_abs_line_slope = max(0.05, self.p.laser_max_abs_line_slope)
        self.p.laser_max_angle_to_map = max(0.05, self.p.laser_max_angle_to_map)
        self.p.laser_max_center_offset = max(0.05, self.p.laser_max_center_offset)
        self.p.laser_min_confidence = float(np.clip(self.p.laser_min_confidence, 0.0, 0.99))
        self.p.laser_full_confidence = float(np.clip(self.p.laser_full_confidence, self.p.laser_min_confidence + 0.01, 1.0))
        self.p.laser_max_weight_both_sides = float(np.clip(self.p.laser_max_weight_both_sides, 0.0, 1.0))
        self.p.laser_max_weight_one_side = float(np.clip(self.p.laser_max_weight_one_side, 0.0, 1.0))
        self.p.laser_tracker_alpha = float(np.clip(self.p.laser_tracker_alpha, 0.01, 1.0))
        self.p.min_follow_turn_radius = max(0.1, self.p.min_follow_turn_radius)
        self.p.angular_rate_limit = max(0.01, self.p.angular_rate_limit)
        self.p.target_filter_alpha = float(np.clip(self.p.target_filter_alpha, 0.01, 1.0))
        self.p.row_exit_extension_distance = max(0.0, self.p.row_exit_extension_distance)
        self.p.row_end_goal_outward_distance = max(0.0, self.p.row_end_goal_outward_distance)
        self.p.pure_pursuit_gain = max(0.05, self.p.pure_pursuit_gain)
        self.p.lookahead_distance = max(0.05, self.p.lookahead_distance)
        self.p.turn_lookahead_distance = float(
            np.clip(self.p.turn_lookahead_distance, 0.05, self.p.lookahead_distance)
        )
        self.p.lookahead_curvature_gain = max(0.0, self.p.lookahead_curvature_gain)
        self.p.slow_speed = float(np.clip(self.p.slow_speed, 0.0, self.p.follow_speed))
        self.p.curve_speed_reduction_gain = max(0.0, self.p.curve_speed_reduction_gain)
        self.p.maneuver_goal_xy_tolerance = max(0.05, self.p.maneuver_goal_xy_tolerance)
        self.p.maneuver_goal_yaw_tolerance = max(0.05, self.p.maneuver_goal_yaw_tolerance)
        self.p.maneuver_heading_lookahead_distance = max(0.05, self.p.maneuver_heading_lookahead_distance)
        self.p.maneuver_speed = max(0.01, self.p.maneuver_speed)
        self.p.maneuver_slow_speed = float(np.clip(self.p.maneuver_slow_speed, 0.01, self.p.maneuver_speed))
        self.p.maneuver_lookahead_distance = max(0.05, self.p.maneuver_lookahead_distance)
        self.p.maneuver_route_spacing = max(0.02, self.p.maneuver_route_spacing)
        self.p.maneuver_corner_radius = max(0.0, self.p.maneuver_corner_radius)
        self.p.maneuver_entry_extension_distance = max(0.10, self.p.maneuver_entry_extension_distance)
        self.p.maneuver_max_route_deviation = max(0.10, self.p.maneuver_max_route_deviation)
        self.p.maneuver_entry_lateral_tolerance = max(0.05, self.p.maneuver_entry_lateral_tolerance)
        self.p.maneuver_entry_yaw_tolerance = max(0.05, self.p.maneuver_entry_yaw_tolerance)
        self.p.starting_lane_number = max(1, self.p.starting_lane_number)
        self.p.row_numbers_increase_to = self.p.row_numbers_increase_to.lower()
        if self.p.row_numbers_increase_to not in ("left", "right"):
            self.get_logger().warn("row_numbers_increase_to must be 'left' or 'right'; using 'left'")
            self.p.row_numbers_increase_to = "left"

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.latest_scan_received_ns = self.get_clock().now().nanoseconds

    def start_cb(self, req, res):
        pattern = req.pattern.strip()
        invalid_tokens = [token for token in pattern.split() if re.fullmatch(r"([1-9][0-9]*)([LlRr])", token) is None]
        if not pattern or invalid_tokens:
            res.success = False
            res.message = "Invalid pattern. Use space-separated steps such as '1L 2R'."
            return res

        self.p.pattern = pattern
        self.pattern_steps = self.parse_pattern(pattern)
        self.left_row = None
        self.right_row = None
        self.midline = np.empty((0, 2), dtype=float)
        self.plant_row_end_point = None
        self.row_exit_goal = None
        self.row_exit_heading_goal = None
        self.row_end_direction = None
        self.initial_forward_direction = None
        self.row_number_increase_direction = None
        self.pattern_index = 0
        self.finish_after_current_row = len(self.pattern_steps) == 0
        self.stored_rows = {}
        self.row_end_directions_by_side = {"forward": [], "backward": []}
        self.row_map_exported = False
        self.reset_entrance_state()
        self.reset_controller_state()
        self.laser_follower.reset()
        self.laser_follow_result = LaserFollowResult(reason="navigation started")
        self.fused_target_point = None
        self.state = MissionState.INITIALIZING
        res.success = True
        res.message = f"Navigation started with pattern: {pattern}"
        return res

    def stop_cb(self, req, res):
        self.store_current_rows()
        export_path = self.export_row_map()
        self.state = MissionState.IDLE
        self.reset_entrance_state()
        self.reset_controller_state()
        self.cmd_pub.publish(Twist())
        res.success = True
        res.message = "Navigation stopped"
        if export_path is not None:
            res.message += f"; row map saved to {export_path}"
        return res

    def reset_controller_state(self) -> None:
        self.last_cmd_angular_z = 0.0
        self.last_target_point = None
        self.fused_target_point = None

    def reset_entrance_state(self) -> None:
        self.entrance_hist_center = None
        self.entrance_hist_direction = None
        self.entrance_hist_peaks = []
        self.entrance_target = None
        self.entrance_target_direction = None
        self.entrance_waypoints = []
        self.entrance_heading_goal = None
        self.entrance_follow_path = np.empty((0, 2), dtype=float)
        self.entrance_route_support = np.empty((0, 2), dtype=float)
        self.entrance_route = np.empty((0, 2), dtype=float)
        self.entrance_route_progress_index = 0
        self.entrance_route_projection = None
        self.entrance_route_target = None
        self.entrance_route_remaining_distance = 0.0
        self.entrance_route_provisional = False
        self.entrance_active_step = None
        self.entrance_traverse_outward = None
        self.pending_target_peaks = None

    def parse_pattern(self, pattern: str) -> List[PatternStep]:
        steps: List[PatternStep] = []
        for token in pattern.split():
            match = re.fullmatch(r"([1-9][0-9]*)([LlRr])", token)
            if match is None:
                self.get_logger().warn(f"Ignoring invalid pattern token: {token!r}")
                continue
            steps.append(PatternStep(int(match.group(1)), match.group(2).upper()))
        return steps

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
        elif self.state == MissionState.FIND_NEXT_ROW_ENTRANCE:
            self.handle_find_next_row_entrance()
        elif self.state == MissionState.FINISHED:
            self.cmd_pub.publish(Twist())

        self.publish_visuals()

    def handle_initializing(self) -> None:
        points = self.get_map_points_in_hist_roi(self.robot_pose)
        if len(points) < 5:
            self.get_logger().info(f"Not enough occupied map points for histogram: {len(points)}", throttle_duration_sec=2.0)
            return

        peak_pair = self.find_start_peak_pair(points, self.robot_pose)
        if peak_pair is None:
            self.get_logger().info("No valid left/right histogram peak pair found", throttle_duration_sec=2.0)
            return

        left_peak, right_peak = peak_pair
        forward = self.yaw_to_vector(self.robot_pose.yaw)

        left_point1 = self.start_point1_from_peak(points, self.robot_pose, left_peak)
        right_point1 = self.start_point1_from_peak(points, self.robot_pose, right_peak)
        left_point2 = self.initial_point2_from_sector(points, self.robot_pose, left_point1, left_peak)
        right_point2 = self.initial_point2_from_sector(points, self.robot_pose, right_point1, right_peak)

        left_direction = self.initial_direction_from_points(left_point1, left_point2, forward)
        right_direction = self.initial_direction_from_points(right_point1, right_point2, forward)
        if self.initial_forward_direction is None:
            self.initial_forward_direction = np.asarray(forward, dtype=float)
            initial_left = np.array([-forward[1], forward[0]], dtype=float)
            increase_sign = 1.0 if self.p.row_numbers_increase_to == "left" else -1.0
            self.row_number_increase_direction = increase_sign * initial_left

        lane_number = self.p.starting_lane_number
        if self.p.row_numbers_increase_to == "left":
            left_number, right_number = lane_number + 1, lane_number
        else:
            left_number, right_number = lane_number, lane_number + 1
        self.left_row = RowMarchModel("left", left_point1, left_point2, left_direction, left_number)
        self.right_row = RowMarchModel("right", right_point1, right_point2, right_direction, right_number)

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
        if self.row_exit_goal is not None and self.row_end_direction is not None:
            if self.pose_goal_reached(self.row_exit_goal, self.row_end_direction):
                self.store_current_rows()
                self.record_current_row_end_direction()
                self.reset_controller_state()
                self.cmd_pub.publish(Twist())
                if self.finish_after_current_row:
                    self.get_logger().info("Last pattern row completed. Mission finished.")
                    self.export_row_map()
                    self.state = MissionState.FINISHED
                else:
                    self.get_logger().info("Row exit reached. Finding the next row entrance.")
                    self.reset_entrance_state()
                    self.state = MissionState.FIND_NEXT_ROW_ENTRANCE
                return

        lookahead_distance = self.dynamic_follow_lookahead(self.midline, robot_xy)
        target = self.lookahead_point_from_polyline_projection(self.midline, robot_xy, lookahead_distance)
        if target is None:
            target = self.midline[-1]
        target = np.asarray(target, dtype=float)
        self.fused_target_point = self.fuse_follow_target_with_laser(target)
        self.drive_to_point(self.fused_target_point)

    def fuse_follow_target_with_laser(self, map_target: np.ndarray) -> np.ndarray:
        self.laser_follow_result = LaserFollowResult(reason="laser following disabled")
        if not self.p.laser_follow_enabled:
            return np.asarray(map_target, dtype=float)

        scan = self.latest_scan
        if self.latest_scan_received_ns is None:
            scan = None
        else:
            age = (self.get_clock().now().nanoseconds - self.latest_scan_received_ns) * 1e-9
            if age > self.p.laser_scan_timeout:
                scan = None

        map_target_base = self.map_point_to_base(map_target)
        tangent_map = self.polyline_tangent_at_point(self.midline, np.array([self.robot_pose.x, self.robot_pose.y]))
        tangent_base = self.map_direction_to_base(tangent_map)
        if abs(float(tangent_base[0])) < 1e-6:
            map_slope = math.copysign(self.p.laser_max_abs_line_slope, float(tangent_base[1]))
        else:
            map_slope = float(tangent_base[1] / tangent_base[0])

        self.laser_follow_result = self.laser_follower.process_scan(scan, map_slope, map_target_base)
        result = self.laser_follow_result
        if not result.valid or result.target_base is None or result.weight <= 0.0:
            return np.asarray(map_target, dtype=float)
        fused_base = (1.0 - result.weight) * map_target_base + result.weight * result.target_base
        return self.base_point_to_map(fused_base)

    def map_point_to_base(self, point: np.ndarray) -> np.ndarray:
        rel = np.asarray(point, dtype=float) - np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
        cosine = math.cos(self.robot_pose.yaw)
        sine = math.sin(self.robot_pose.yaw)
        return np.array([cosine * rel[0] + sine * rel[1], -sine * rel[0] + cosine * rel[1]], dtype=float)

    def base_point_to_map(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        cosine = math.cos(self.robot_pose.yaw)
        sine = math.sin(self.robot_pose.yaw)
        return np.array(
            [
                self.robot_pose.x + cosine * point[0] - sine * point[1],
                self.robot_pose.y + sine * point[0] + cosine * point[1],
            ],
            dtype=float,
        )

    def map_direction_to_base(self, direction: np.ndarray) -> np.ndarray:
        direction = self.normalize(direction)
        cosine = math.cos(self.robot_pose.yaw)
        sine = math.sin(self.robot_pose.yaw)
        return np.array([cosine * direction[0] + sine * direction[1], -sine * direction[0] + cosine * direction[1]])

    def base_direction_to_map(self, direction: np.ndarray) -> np.ndarray:
        direction = self.normalize(direction)
        cosine = math.cos(self.robot_pose.yaw)
        sine = math.sin(self.robot_pose.yaw)
        return np.array([cosine * direction[0] - sine * direction[1], sine * direction[0] + cosine * direction[1]])

    def polyline_tangent_at_point(self, polyline: np.ndarray, point: np.ndarray) -> np.ndarray:
        if len(polyline) < 2:
            return self.yaw_to_vector(self.robot_pose.yaw)
        point = np.asarray(point, dtype=float)
        best_direction = np.asarray(polyline[1] - polyline[0], dtype=float)
        best_distance = float("inf")
        for start, end in zip(polyline[:-1], polyline[1:]):
            segment = end - start
            length_sq = float(segment @ segment)
            if length_sq < 1e-12:
                continue
            ratio = float(np.clip(((point - start) @ segment) / length_sq, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (start + ratio * segment)))
            if distance < best_distance:
                best_distance = distance
                best_direction = segment
        return self.normalize(best_direction)

    def store_current_rows(self) -> None:
        for model in (self.left_row, self.right_row):
            if model is None or len(model.result.points) < 2:
                continue
            points = self.orient_row_points(model.result.points)
            existing = self.stored_rows.get(model.row_number)
            if existing is None or self.polyline_length(points) > self.polyline_length(existing):
                self.stored_rows[model.row_number] = np.array(points, copy=True)

    def orient_row_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if len(points) < 2 or self.initial_forward_direction is None:
            return np.array(points, copy=True)
        if float((points[-1] - points[0]) @ self.initial_forward_direction) < 0.0:
            return np.array(points[::-1], copy=True)
        return np.array(points, copy=True)

    def polyline_length(self, points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def build_rounded_route(self, support_points: np.ndarray) -> np.ndarray:
        points = [np.asarray(point, dtype=float) for point in support_points]
        if len(points) < 2:
            return np.asarray(points, dtype=float)
        route: List[np.ndarray] = [points[0]]
        for idx in range(1, len(points) - 1):
            previous = points[idx - 1]
            corner = points[idx]
            following = points[idx + 1]
            incoming = corner - previous
            outgoing = following - corner
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length < 1e-6 or outgoing_length < 1e-6:
                continue
            radius = min(
                self.p.maneuver_corner_radius,
                0.50 * incoming_length,
                0.50 * outgoing_length,
            )
            before = corner - radius * incoming / incoming_length
            after = corner + radius * outgoing / outgoing_length
            self.append_sampled_line(route, route[-1], before)
            sample_count = max(3, int(math.ceil((2.0 * radius) / self.p.maneuver_route_spacing)))
            for ratio in np.linspace(0.0, 1.0, sample_count)[1:]:
                one_minus = 1.0 - float(ratio)
                route.append(one_minus * one_minus * before + 2.0 * one_minus * ratio * corner + ratio * ratio * after)
        self.append_sampled_line(route, route[-1], points[-1])
        return np.asarray(route, dtype=float)

    def append_sampled_line(self, route: List[np.ndarray], start: np.ndarray, end: np.ndarray) -> None:
        distance = float(np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float)))
        sample_count = max(1, int(math.ceil(distance / self.p.maneuver_route_spacing)))
        for ratio in np.linspace(0.0, 1.0, sample_count + 1)[1:]:
            route.append((1.0 - ratio) * np.asarray(start, dtype=float) + ratio * np.asarray(end, dtype=float))

    def project_onto_route_forward(
        self,
        route: np.ndarray,
        point: np.ndarray,
        start_idx: int,
    ) -> Tuple[np.ndarray, int, float]:
        point = np.asarray(point, dtype=float)
        best_projection = np.asarray(route[-1], dtype=float)
        best_idx = min(max(0, start_idx), max(0, len(route) - 2))
        best_distance = float("inf")
        for idx in range(best_idx, len(route) - 1):
            segment = route[idx + 1] - route[idx]
            length_sq = float(segment @ segment)
            if length_sq < 1e-12:
                continue
            ratio = float(np.clip(((point - route[idx]) @ segment) / length_sq, 0.0, 1.0))
            projection = route[idx] + ratio * segment
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_projection = projection
                best_idx = idx
                best_distance = distance
        return np.asarray(best_projection, dtype=float), best_idx, best_distance

    def polyline_distance_from_projection(self, route: np.ndarray, projection: np.ndarray, start_idx: int) -> float:
        if len(route) < 2:
            return 0.0
        start_idx = int(np.clip(start_idx, 0, len(route) - 2))
        return float(np.linalg.norm(route[start_idx + 1] - projection)) + self.polyline_length(route[start_idx + 1:])

    def point_at_route_distance(
        self,
        route: np.ndarray,
        projection: np.ndarray,
        start_idx: int,
        distance_ahead: float,
    ) -> np.ndarray:
        start_idx = int(np.clip(start_idx, 0, len(route) - 2))
        travelled = 0.0
        segment_start = np.asarray(projection, dtype=float)
        for idx in range(start_idx, len(route) - 1):
            segment_end = np.asarray(route[idx + 1], dtype=float)
            segment = segment_end - segment_start
            length = float(np.linalg.norm(segment))
            if travelled + length >= distance_ahead and length > 1e-9:
                return segment_start + ((distance_ahead - travelled) / length) * segment
            travelled += length
            segment_start = segment_end
        return np.asarray(route[-1], dtype=float)

    def export_row_map(self) -> Optional[str]:
        if self.row_map_exported or len(self.stored_rows) == 0:
            return None
        output_directory = os.path.abspath(os.path.expanduser(self.p.row_map_output_directory))
        filename = f"maize_row_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join(output_directory, filename)
        try:
            os.makedirs(output_directory, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(("row_number", "point_index", "x", "y", "map_frame"))
                for row_number in sorted(self.stored_rows):
                    for point_index, point in enumerate(self.stored_rows[row_number]):
                        writer.writerow(
                            (
                                row_number,
                                point_index,
                                f"{float(point[0]):.6f}",
                                f"{float(point[1]):.6f}",
                                self.p.map_frame,
                            )
                        )
            self.row_map_exported = True
            self.get_logger().info(f"Saved maize row map CSV: {output_path}")
            return output_path
        except OSError as exc:
            self.get_logger().error(f"Could not save maize row map CSV to {output_path}: {exc}")
            return None

    def handle_find_next_row_entrance(self) -> None:
        if not self.ensure_provisional_entrance_route():
            self.cmd_pub.publish(Twist())
            return
        if not self.lock_next_row_entrance() and self.entrance_target is None:
            self.rebuild_entrance_route(None)

        if (
            self.entrance_target_direction is not None
            and self.entry_line_reached(self.entrance_target, self.entrance_target_direction)
        ):
            self.initialize_selected_entrance_rows()
            return

        target = self.update_entrance_route_target()
        if target is None:
            self.cmd_pub.publish(Twist())
            return
        if (
            self.entrance_route_provisional
            and self.entrance_route_remaining_distance <= self.p.maneuver_goal_xy_tolerance
        ):
            self.cmd_pub.publish(Twist())
            return
        self.drive_to_point(target, self.p.maneuver_speed, self.p.maneuver_slow_speed)

    def entry_line_reached(self, goal: np.ndarray, direction: np.ndarray) -> bool:
        direction = self.normalize(direction)
        lateral_direction = np.array([-direction[1], direction[0]], dtype=float)
        robot_xy = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
        rel = robot_xy - np.asarray(goal, dtype=float)
        along = float(rel @ direction)
        lateral = abs(float(rel @ lateral_direction))
        desired_yaw = math.atan2(float(direction[1]), float(direction[0]))
        yaw_error = abs(wrap_to_pi(desired_yaw - self.robot_pose.yaw))
        return (
            along >= 0.0
            and lateral <= self.p.maneuver_entry_lateral_tolerance
            and yaw_error <= self.p.maneuver_entry_yaw_tolerance
        )

    def update_entrance_route_target(self) -> Optional[np.ndarray]:
        if len(self.entrance_route) < 2:
            self.get_logger().warn("Cannot follow headland maneuver without an entrance route")
            return None
        robot_xy = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
        projection, segment_idx, deviation = self.project_onto_route_forward(
            self.entrance_route,
            robot_xy,
            self.entrance_route_progress_index,
        )
        self.entrance_route_projection = projection
        self.entrance_route_progress_index = max(self.entrance_route_progress_index, segment_idx)
        if deviation > self.p.maneuver_max_route_deviation:
            self.get_logger().warn(
                f"Headland route deviation too large: {deviation:.3f} m",
                throttle_duration_sec=1.0,
            )
            self.entrance_route_target = None
            return None
        self.entrance_route_remaining_distance = self.polyline_distance_from_projection(
            self.entrance_route,
            projection,
            self.entrance_route_progress_index,
        )
        lookahead = min(
            self.p.maneuver_lookahead_distance,
            max(0.10, 0.60 * self.entrance_route_remaining_distance),
        )
        self.entrance_route_target = self.point_at_route_distance(
            self.entrance_route,
            projection,
            self.entrance_route_progress_index,
            lookahead,
        )
        return self.entrance_route_target

    def pose_goal_reached(self, goal: np.ndarray, direction: np.ndarray) -> bool:
        direction = self.normalize(direction)
        lateral_direction = np.array([-direction[1], direction[0]], dtype=float)
        robot_xy = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
        rel = robot_xy - np.asarray(goal, dtype=float)
        along = float(rel @ direction)
        lateral = abs(float(rel @ lateral_direction))
        distance = float(np.linalg.norm(rel))
        desired_yaw = math.atan2(float(direction[1]), float(direction[0]))
        yaw_error = abs(wrap_to_pi(desired_yaw - self.robot_pose.yaw))
        position_reached = distance <= self.p.maneuver_goal_xy_tolerance or (
            along >= 0.0 and lateral <= self.p.maneuver_goal_xy_tolerance
        )
        return position_reached and yaw_error <= self.p.maneuver_goal_yaw_tolerance

    def lock_next_row_entrance(self) -> bool:
        if (
            self.plant_row_end_point is None
            or self.row_end_direction is None
            or self.left_row is None
            or self.right_row is None
            or self.entrance_active_step is None
        ):
            self.get_logger().warn("Cannot find next entrance without completed row end data and a pending pattern step")
            return False

        center = np.asarray(self.plant_row_end_point, dtype=float)
        outgoing = self.average_row_end_direction_for_side(self.row_end_direction)
        roi_points = self.points_in_oriented_rectangle(
            self.get_all_map_points(),
            center,
            outgoing,
            self.p.hist_roi_depth,
            self.p.hist_roi_width,
        )
        self.entrance_hist_center = center
        self.entrance_hist_direction = outgoing
        self.entrance_hist_peaks = self.find_entrance_histogram_peaks(roi_points, center, outgoing)
        if len(self.entrance_hist_peaks) < 2:
            self.get_logger().info("Not enough headland histogram peaks for row entrance", throttle_duration_sec=2.0)
            return False

        current_indices = self.associate_known_rows_with_entrance_peaks(center, outgoing)
        if current_indices is None:
            self.get_logger().info("Could not associate known plant rows with headland peaks", throttle_duration_sec=2.0)
            return False

        step = self.entrance_active_step
        shift = step.lane_shift if step.direction == "L" else -step.lane_shift
        target_indices = (current_indices[0] + shift, current_indices[1] + shift)
        if min(target_indices) < 0 or max(target_indices) >= len(self.entrance_hist_peaks):
            self.get_logger().info(
                f"Pattern step {step.lane_shift}{step.direction} needs peaks outside detected range",
                throttle_duration_sec=2.0,
            )
            return False

        first = self.entrance_hist_peaks[target_indices[0]]
        second = self.entrance_hist_peaks[target_indices[1]]
        first.selected = True
        second.selected = True
        target_row_end = 0.5 * (first.point + second.point)
        new_target = target_row_end + self.p.row_end_goal_outward_distance * outgoing
        first_lock = self.entrance_route_provisional
        target_shift = (
            float("inf")
            if self.entrance_target is None
            else float(np.linalg.norm(new_target - self.entrance_target))
        )
        new_target_lateral = float((new_target - center) @ np.array([-outgoing[1], outgoing[0]], dtype=float))
        new_traverse_outward = self.compute_headland_traverse_outward(center, outgoing, new_target_lateral)
        traverse_shift = (
            float("inf")
            if self.entrance_traverse_outward is None
            else abs(new_traverse_outward - self.entrance_traverse_outward)
        )
        replan_threshold = max(0.15, 0.25 * self.p.expected_row_width)
        should_replan = (
            first_lock
            or len(self.entrance_route) < 2
            or target_shift >= replan_threshold
            or traverse_shift >= replan_threshold
        )
        if should_replan:
            self.pending_target_peaks = (first, second)
            self.entrance_target = new_target
            self.entrance_target_direction = -outgoing
            self.entrance_heading_goal = (
                self.entrance_target
                + self.p.maneuver_entry_extension_distance * self.entrance_target_direction
            )
            self.entrance_follow_path = self.build_entrance_follow_path()
            self.rebuild_entrance_route(self.entrance_target)
        if first_lock:
            self.pattern_index += 1
            self.finish_after_current_row = self.pattern_index >= len(self.pattern_steps)
            self.reset_controller_state()
            self.get_logger().info(
                f"Locked next entrance for pattern step {step.lane_shift}{step.direction}: "
                f"rows {first.row_number}/{second.row_number}"
            )
        return True

    def ensure_provisional_entrance_route(self) -> bool:
        if self.entrance_active_step is not None and len(self.entrance_route) >= 2:
            return True
        if (
            self.plant_row_end_point is None
            or self.row_end_direction is None
            or self.pattern_index >= len(self.pattern_steps)
        ):
            return False

        self.entrance_active_step = self.pattern_steps[self.pattern_index]
        self.rebuild_entrance_route(None)
        self.reset_controller_state()
        self.get_logger().info(
            f"Headland peaks not ready; following provisional route for pattern step "
            f"{self.entrance_active_step.lane_shift}{self.entrance_active_step.direction}",
            throttle_duration_sec=2.0,
        )
        return True

    def rebuild_entrance_route(self, target: Optional[np.ndarray]) -> None:
        if self.plant_row_end_point is None or self.row_end_direction is None or self.entrance_active_step is None:
            return
        center = np.asarray(self.plant_row_end_point, dtype=float)
        outgoing = self.average_row_end_direction_for_side(self.row_end_direction)
        lateral_direction = np.array([-outgoing[1], outgoing[0]], dtype=float)
        step = self.entrance_active_step
        signed_shift = step.lane_shift if step.direction == "L" else -step.lane_shift
        expected_lateral = signed_shift * self.p.expected_row_width
        target_lateral = expected_lateral if target is None else float((np.asarray(target) - center) @ lateral_direction)
        traverse_outward = self.compute_headland_traverse_outward(center, outgoing, target_lateral)

        robot_xy = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
        incoming = -outgoing
        target_lateral_error = (
            float("inf")
            if target is None
            else abs(float((robot_xy - np.asarray(target, dtype=float)) @ lateral_direction))
        )
        entry_capture_lateral = max(self.p.maneuver_entry_lateral_tolerance, 0.5 * self.p.expected_row_width)
        if (
            target is not None
            and float((robot_xy - np.asarray(target, dtype=float)) @ incoming) >= 0.0
            and target_lateral_error <= entry_capture_lateral
        ):
            forward_path = [
                point
                for point in self.entrance_follow_path
                if float((np.asarray(point, dtype=float) - robot_xy) @ incoming) > self.p.maneuver_route_spacing
            ]
            if not forward_path:
                forward_path = [robot_xy + self.p.maneuver_entry_extension_distance * incoming]
            self.entrance_waypoints = []
            self.set_entrance_route(np.vstack((robot_xy, *forward_path)), False)
            return

        route_start = self.headland_route_anchor(center, outgoing)
        route, support_points, waypoints = self.build_headland_route(
            route_start,
            center,
            outgoing,
            lateral_direction,
            target_lateral,
            traverse_outward,
            target,
        )
        self.entrance_waypoints = waypoints
        self.entrance_route_support = support_points
        self.entrance_route = route
        self.entrance_route_progress_index = 0
        self.entrance_route_projection = None
        self.entrance_route_target = None
        self.entrance_route_remaining_distance = self.polyline_length(self.entrance_route)
        self.entrance_route_provisional = target is None
        self.entrance_traverse_outward = traverse_outward

    def headland_route_anchor(self, center: np.ndarray, outgoing: np.ndarray) -> np.ndarray:
        if self.row_exit_goal is not None:
            return np.asarray(self.row_exit_goal, dtype=float)
        return np.asarray(center, dtype=float) + self.p.row_end_goal_outward_distance * self.normalize(outgoing)

    def compute_headland_traverse_outward(
        self,
        center: np.ndarray,
        outgoing: np.ndarray,
        target_lateral: float,
    ) -> float:
        lower_lateral, upper_lateral = sorted((0.0, target_lateral))
        lateral_margin = self.p.expected_row_width
        route_peaks = [
            peak
            for peak in self.entrance_hist_peaks
            if lower_lateral - lateral_margin <= peak.lateral <= upper_lateral + lateral_margin
        ]
        outermost = max(
            [0.0] + [float((np.asarray(peak.point) - center) @ outgoing) for peak in route_peaks]
        )
        clearance = max(self.p.row_exit_extension_distance, self.p.row_end_goal_outward_distance)
        return outermost + clearance

    def build_headland_route(
        self,
        start: np.ndarray,
        center: np.ndarray,
        outgoing: np.ndarray,
        lateral_direction: np.ndarray,
        target_lateral: float,
        traverse_outward: float,
        target: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        start = np.asarray(start, dtype=float)
        center = np.asarray(center, dtype=float)
        outgoing = self.normalize(outgoing)
        lateral_direction = self.normalize(lateral_direction)
        start_lateral = float((start - center) @ lateral_direction)
        start_outward = float((start - center) @ outgoing)
        target_outward = start_outward if target is None else float((np.asarray(target) - center) @ outgoing)
        delta_lateral = target_lateral - start_lateral
        turn_sign = 1.0 if delta_lateral >= 0.0 else -1.0
        abs_delta_lateral = abs(delta_lateral)
        desired_radius = self.p.maneuver_corner_radius
        if abs_delta_lateral < 1e-6:
            desired_radius = 0.0
        radius = min(desired_radius, 0.5 * abs_delta_lateral) if desired_radius > 0.0 else 0.0
        traverse_outward = max(traverse_outward, start_outward + radius, target_outward + radius)

        route: List[np.ndarray] = [start]
        support: List[np.ndarray] = [start]
        waypoints: List[np.ndarray] = []

        if radius <= 1e-6:
            straight_point = center + traverse_outward * outgoing + target_lateral * lateral_direction
            self.append_sampled_line(route, start, straight_point)
            support.append(straight_point)
            waypoints.append(straight_point)
        else:
            first_arc_start = center + (traverse_outward - radius) * outgoing + start_lateral * lateral_direction
            first_straight_start = (
                center
                + traverse_outward * outgoing
                + (start_lateral + turn_sign * radius) * lateral_direction
            )
            second_straight_end = (
                center
                + traverse_outward * outgoing
                + (target_lateral - turn_sign * radius) * lateral_direction
            )
            second_arc_end = center + (traverse_outward - radius) * outgoing + target_lateral * lateral_direction

            self.append_sampled_line(route, route[-1], first_arc_start)
            self.append_headland_arc(route, first_arc_start, outgoing, lateral_direction, turn_sign, radius, first=True)
            if target is None:
                provisional_end = center + traverse_outward * outgoing + target_lateral * lateral_direction
                self.append_sampled_line(route, route[-1], provisional_end)
                support.extend((first_arc_start, first_straight_start, provisional_end))
                waypoints.extend((first_straight_start, provisional_end))
            else:
                self.append_sampled_line(route, route[-1], second_straight_end)
                self.append_headland_arc(route, second_straight_end, outgoing, lateral_direction, turn_sign, radius, first=False)
                support.extend((first_arc_start, first_straight_start, second_straight_end, second_arc_end))
                waypoints.extend((first_straight_start, second_straight_end))

        if target is not None:
            target = np.asarray(target, dtype=float)
            self.append_sampled_line(route, route[-1], target)
            support.append(target)
            for point in self.entrance_follow_path:
                self.append_sampled_line(route, route[-1], np.asarray(point, dtype=float))
                support.append(np.asarray(point, dtype=float))

        return np.asarray(route, dtype=float), np.asarray(support, dtype=float), waypoints

    def append_headland_arc(
        self,
        route: List[np.ndarray],
        arc_start: np.ndarray,
        outgoing: np.ndarray,
        lateral_direction: np.ndarray,
        turn_sign: float,
        radius: float,
        first: bool,
    ) -> None:
        sample_count = max(3, int(math.ceil((0.5 * math.pi * radius) / self.p.maneuver_route_spacing)))
        for theta in np.linspace(0.0, 0.5 * math.pi, sample_count + 1)[1:]:
            if first:
                point = (
                    arc_start
                    + radius * math.sin(float(theta)) * outgoing
                    + turn_sign * radius * (1.0 - math.cos(float(theta))) * lateral_direction
                )
            else:
                point = (
                    arc_start
                    + turn_sign * radius * math.sin(float(theta)) * lateral_direction
                    - radius * (1.0 - math.cos(float(theta))) * outgoing
                )
            route.append(np.asarray(point, dtype=float))

    def set_entrance_route(
        self,
        support_points: np.ndarray,
        provisional: bool,
        marker_support_points: Optional[np.ndarray] = None,
    ) -> None:
        if marker_support_points is None:
            marker_support_points = support_points
        self.entrance_route_support = np.asarray(marker_support_points, dtype=float)
        self.entrance_route = self.build_rounded_route(support_points)
        self.entrance_route_progress_index = 0
        self.entrance_route_projection = None
        self.entrance_route_target = None
        self.entrance_route_remaining_distance = self.polyline_length(self.entrance_route)
        self.entrance_route_provisional = provisional

    def build_entrance_follow_path(self) -> np.ndarray:
        models = self.build_selected_entrance_models()
        if models is None or self.entrance_target is None or self.entrance_target_direction is None:
            return np.empty((0, 2), dtype=float)
        incoming = self.normalize(self.entrance_target_direction)
        fallback = self.entrance_target + self.p.maneuver_entry_extension_distance * incoming
        left_model, right_model = models
        map_points = self.get_all_map_points()
        left_result = self.march_row(left_model, map_points)
        right_result = self.march_row(right_model, map_points)
        midline = self.build_midline(left_result.points, right_result.points)
        path = [np.asarray(fallback, dtype=float)]
        minimum_along = self.p.maneuver_entry_extension_distance + self.p.maneuver_route_spacing
        for point in midline:
            if float((np.asarray(point, dtype=float) - self.entrance_target) @ incoming) > minimum_along:
                path.append(np.asarray(point, dtype=float))
        return np.asarray(path, dtype=float)

    def find_entrance_histogram_peaks(
        self,
        points: np.ndarray,
        center: np.ndarray,
        outgoing_direction: np.ndarray,
    ) -> List[EntrancePeak]:
        if len(points) == 0:
            return []
        lateral_direction = np.array([-outgoing_direction[1], outgoing_direction[0]], dtype=float)
        local_y = (points - center) @ lateral_direction
        half_roi = 0.5 * self.p.hist_roi_width
        bins = np.arange(-half_roi, half_roi + self.p.hist_bin_size, self.p.hist_bin_size)
        if len(bins) < 3:
            return []
        hist, bin_edges = np.histogram(local_y, bins=bins)
        smoothed_hist = np.convolve(hist, np.array([1, 1, 1], dtype=int), mode="same")
        candidates: List[Tuple[int, EntrancePeak]] = []
        for idx in range(1, len(hist) - 1):
            if (
                smoothed_hist[idx] >= self.p.hist_peak_min_points
                and smoothed_hist[idx] >= smoothed_hist[idx - 1]
                and smoothed_hist[idx] >= smoothed_hist[idx + 1]
                and hist[idx - 1] + hist[idx] + hist[idx + 1] > 0
            ):
                window_counts = hist[idx - 1 : idx + 2]
                window_centers = 0.5 * (bin_edges[idx - 1 : idx + 2] + bin_edges[idx : idx + 3])
                lateral = float(np.average(window_centers, weights=window_counts))
                point = self.actual_row_end_from_peak(points, center, outgoing_direction, lateral)
                candidates.append((int(smoothed_hist[idx]), EntrancePeak(lateral, point)))

        # Broad occupied bands often form several adjacent local maxima. Count them
        # as one plant row before applying pattern-relative peak offsets.
        min_peak_spacing = max(2.0 * self.p.hist_bin_size, 0.5 * self.p.min_lane_width)
        peaks: List[EntrancePeak] = []
        for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(abs(candidate.lateral - peak.lateral) >= min_peak_spacing for peak in peaks):
                peaks.append(candidate)
        peaks.sort(key=lambda peak: peak.lateral)
        self.get_logger().info(
            f"Headland histogram peaks: {[round(peak.lateral, 3) for peak in peaks]}",
            throttle_duration_sec=2.0,
        )
        return peaks

    def regularize_entrance_peak_spacing(self, peaks: List[EntrancePeak]) -> List[EntrancePeak]:
        regularized: List[EntrancePeak] = []
        for peak in sorted(peaks, key=lambda item: item.lateral):
            if not regularized or abs(peak.lateral - regularized[-1].lateral) >= self.p.min_lane_width:
                regularized.append(peak)
        return regularized

    def actual_row_end_from_peak(
        self,
        points: np.ndarray,
        center: np.ndarray,
        outgoing_direction: np.ndarray,
        peak_lateral: float,
    ) -> np.ndarray:
        lateral_direction = np.array([-outgoing_direction[1], outgoing_direction[0]], dtype=float)
        rel = points - np.asarray(center, dtype=float)
        local_x = rel @ outgoing_direction
        local_y = rel @ lateral_direction
        half_width = max(self.p.hist_bin_size, 0.5 * self.p.row_rectangle_width)
        band_mask = np.abs(local_y - peak_lateral) <= half_width
        if not np.any(band_mask):
            return np.asarray(center, dtype=float) + peak_lateral * lateral_direction
        band_points = points[band_mask]
        band_x = local_x[band_mask]
        outermost_x = float(np.max(band_x))
        end_points = band_points[band_x >= outermost_x - self.p.row_point_window_length]
        return np.asarray(np.mean(end_points, axis=0), dtype=float)

    def row_end_side(self, direction: np.ndarray) -> str:
        if self.initial_forward_direction is None:
            return "forward"
        return "forward" if float(self.normalize(direction) @ self.initial_forward_direction) >= 0.0 else "backward"

    def record_current_row_end_direction(self) -> None:
        if self.row_end_direction is None:
            return
        direction = self.normalize(self.row_end_direction)
        side = self.row_end_side(direction)
        directions = self.row_end_directions_by_side[side]
        if directions and float(direction @ directions[-1]) < 0.0:
            direction = -direction
        directions.append(direction)

    def average_row_end_direction_for_side(self, current_direction: np.ndarray) -> np.ndarray:
        current_direction = self.normalize(current_direction)
        directions = self.row_end_directions_by_side[self.row_end_side(current_direction)]
        if not directions:
            return current_direction
        aligned = [direction if float(direction @ current_direction) >= 0.0 else -direction for direction in directions]
        return self.normalize(np.sum(aligned, axis=0))

    def associate_known_rows_with_entrance_peaks(
        self,
        center: np.ndarray,
        outgoing_direction: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        if self.left_row is None or self.right_row is None or self.row_number_increase_direction is None:
            return None
        left_end = self.left_row.result.end_point
        right_end = self.right_row.result.end_point
        if left_end is None or right_end is None:
            return None

        lateral_direction = np.array([-outgoing_direction[1], outgoing_direction[0]], dtype=float)
        known_laterals = [
            float((np.asarray(left_end, dtype=float) - center) @ lateral_direction),
            float((np.asarray(right_end, dtype=float) - center) @ lateral_direction),
        ]
        matched_indices = [
            int(np.argmin([abs(peak.lateral - known_laterals[0]) for peak in self.entrance_hist_peaks])),
            int(np.argmin([abs(peak.lateral - known_laterals[1]) for peak in self.entrance_hist_peaks])),
        ]
        matched_distances = [
            abs(self.entrance_hist_peaks[matched_indices[0]].lateral - known_laterals[0]),
            abs(self.entrance_hist_peaks[matched_indices[1]].lateral - known_laterals[1]),
        ]
        if (
            matched_indices[0] == matched_indices[1]
            or abs(matched_indices[0] - matched_indices[1]) != 1
            or max(matched_distances) > self.p.max_lane_width
        ):
            return None

        left_idx, right_idx = matched_indices
        self.entrance_hist_peaks[left_idx].row_number = self.left_row.row_number
        self.entrance_hist_peaks[right_idx].row_number = self.right_row.row_number
        number_step = 1 if float(lateral_direction @ self.row_number_increase_direction) >= 0.0 else -1
        anchor_idx = left_idx
        anchor_number = self.left_row.row_number
        for idx, peak in enumerate(self.entrance_hist_peaks):
            peak.row_number = anchor_number + (idx - anchor_idx) * number_step
        if self.entrance_hist_peaks[right_idx].row_number != self.right_row.row_number:
            return None
        return tuple(sorted((left_idx, right_idx)))

    def initialize_selected_entrance_rows(self) -> None:
        models = self.build_selected_entrance_models()
        if models is None:
            self.cmd_pub.publish(Twist())
            return

        first, second = sorted(self.pending_target_peaks, key=lambda peak: peak.lateral)
        self.left_row, self.right_row = models
        self.midline = np.empty((0, 2), dtype=float)
        self.plant_row_end_point = None
        self.row_exit_goal = None
        self.row_exit_heading_goal = None
        self.row_end_direction = None
        self.reset_controller_state()
        self.laser_follower.reset()
        self.reset_entrance_state()
        self.state = MissionState.FOLLOW_ROW
        self.get_logger().info(
            f"Entered new lane between plant rows {first.row_number} and {second.row_number}; following row"
        )

    def build_selected_entrance_models(self) -> Optional[Tuple[RowMarchModel, RowMarchModel]]:
        if (
            self.pending_target_peaks is None
            or self.entrance_hist_center is None
            or self.entrance_hist_direction is None
        ):
            return None

        outgoing = self.normalize(self.entrance_hist_direction)
        incoming = -outgoing
        roi_points = self.points_in_oriented_rectangle(
            self.get_all_map_points(),
            self.entrance_hist_center,
            outgoing,
            self.p.hist_roi_depth,
            self.p.hist_roi_width,
        )
        first, second = sorted(self.pending_target_peaks, key=lambda peak: peak.lateral)
        # Seen in the new incoming direction, the lower outgoing-lateral peak is the left row.
        left_point1 = self.entrance_point1_from_peak(roi_points, self.entrance_hist_center, outgoing, first.lateral)
        right_point1 = self.entrance_point1_from_peak(roi_points, self.entrance_hist_center, outgoing, second.lateral)
        left_point2 = self.point2_from_sector(roi_points, left_point1, incoming, f"row {first.row_number}")
        right_point2 = self.point2_from_sector(roi_points, right_point1, incoming, f"row {second.row_number}")
        left_direction = self.initial_direction_from_points(left_point1, left_point2, incoming)
        right_direction = self.initial_direction_from_points(right_point1, right_point2, incoming)
        return (
            RowMarchModel("left", left_point1, left_point2, left_direction, int(first.row_number)),
            RowMarchModel("right", right_point1, right_point2, right_direction, int(second.row_number)),
        )

    def find_start_peak_pair(self, points: np.ndarray, pose: Pose2D) -> Optional[Tuple[float, float]]:
        forward = self.yaw_to_vector(pose.yaw)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        robot_xy = np.array([pose.x, pose.y], dtype=float)
        rel = points - robot_xy
        local_y = rel @ lateral

        half_roi = 0.5 * self.p.hist_roi_width
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

    def start_point1_from_peak(self, points: np.ndarray, pose: Pose2D, peak_y: float) -> np.ndarray:
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

        point1 = np.mean(start_points, axis=0)
        self.get_logger().info(
            f"Initial point 1 for peak {peak_y:.3f}: first_x={first_x:.2f}, n={len(start_points)}",
            throttle_duration_sec=2.0,
        )
        return np.asarray(point1, dtype=float)

    def initial_point2_from_sector(self, points: np.ndarray, pose: Pose2D, point1: np.ndarray, peak_y: float) -> np.ndarray:
        forward = self.yaw_to_vector(pose.yaw)
        return self.point2_from_sector(points, point1, forward, f"peak {peak_y:.3f}")

    def point2_from_sector(
        self,
        points: np.ndarray,
        point1: np.ndarray,
        forward: np.ndarray,
        label: str,
    ) -> np.ndarray:
        expected_point2 = np.asarray(point1, dtype=float) + self.p.row_segment_point_spacing * forward
        sector_points = self.points_in_oriented_rectangle(
            points,
            expected_point2,
            forward,
            self.p.row_point_window_length,
            self.p.row_rectangle_width,
        )
        if len(sector_points) == 0:
            self.get_logger().info(
                f"No occupied second-sector points for {label}; using expected point 2",
                throttle_duration_sec=2.0,
            )
            return expected_point2

        point2 = np.mean(sector_points, axis=0)
        self.get_logger().info(
            f"Initial point 2 for {label}: n={len(sector_points)}",
            throttle_duration_sec=2.0,
        )
        return np.asarray(point2, dtype=float)

    def entrance_point1_from_peak(
        self,
        points: np.ndarray,
        center: np.ndarray,
        outgoing_direction: np.ndarray,
        peak_y: float,
    ) -> np.ndarray:
        outgoing_direction = self.normalize(outgoing_direction)
        lateral_direction = np.array([-outgoing_direction[1], outgoing_direction[0]], dtype=float)
        fallback = np.asarray(center, dtype=float) + peak_y * lateral_direction
        if len(points) == 0:
            return fallback

        rel = points - np.asarray(center, dtype=float)
        local_x = rel @ outgoing_direction
        local_y = rel @ lateral_direction
        peak_half_width = max(self.p.hist_bin_size, 0.5 * self.p.row_rectangle_width)
        band_mask = np.abs(local_y - peak_y) <= peak_half_width
        band_points = points[band_mask]
        band_x = local_x[band_mask]
        if len(band_points) == 0:
            return fallback

        outermost_x = float(np.max(band_x))
        point1_points = band_points[band_x >= outermost_x - self.p.row_point_window_length]
        if len(point1_points) == 0:
            return fallback
        return np.asarray(np.mean(point1_points, axis=0), dtype=float)

    def initial_direction_from_points(self, point1: np.ndarray, point2: np.ndarray, fallback_direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(point2, dtype=float) - np.asarray(point1, dtype=float)
        if float(np.linalg.norm(direction)) < 1e-6:
            return self.normalize(fallback_direction)
        fallback_direction = self.normalize(fallback_direction)
        direction = self.normalize(direction)
        if np.dot(direction, fallback_direction) < 0.0:
            direction = -direction
        return direction

    def recompute_rows(self) -> None:
        if self.left_row is None or self.right_row is None:
            return
        map_points = self.get_all_map_points()
        if len(map_points) == 0:
            self.midline = np.empty((0, 2), dtype=float)
            self.plant_row_end_point = None
            self.row_exit_goal = None
            self.row_exit_heading_goal = None
            self.row_end_direction = None
            return

        self.left_row.result = self.march_row(self.left_row, map_points)
        self.right_row.result = self.march_row(self.right_row, map_points)
        self.update_frozen_prefix(self.left_row)
        self.update_frozen_prefix(self.right_row)
        self.midline = self.build_midline(self.left_row.result.points, self.right_row.result.points)

        if self.left_row.result.ended and self.right_row.result.ended:
            left_end = self.left_row.result.end_point
            right_end = self.right_row.result.end_point
            left_direction = self.left_row.result.end_direction
            right_direction = self.right_row.result.end_direction
            if left_end is not None and right_end is not None and left_direction is not None and right_direction is not None and len(self.midline) > 0:
                self.row_end_direction = self.mean_direction(left_direction, right_direction)
                farther_end = left_end if float(left_end @ self.row_end_direction) >= float(right_end @ self.row_end_direction) else right_end
                center_anchor = np.asarray(self.midline[-1], dtype=float)
                self.plant_row_end_point = self.project_point_to_line(farther_end, center_anchor, self.row_end_direction)
                self.row_exit_goal = self.plant_row_end_point + self.p.row_end_goal_outward_distance * self.row_end_direction
                self.row_exit_heading_goal = (
                    self.row_exit_goal
                    + self.p.maneuver_heading_lookahead_distance * self.row_end_direction
                )
                self.midline = self.append_polyline_point(self.midline, self.plant_row_end_point)
                self.midline = self.append_polyline_point(self.midline, self.row_exit_goal)
                self.midline = self.append_polyline_point(self.midline, self.row_exit_heading_goal)
            elif len(self.midline) > 0:
                self.plant_row_end_point = np.asarray(self.midline[-1], dtype=float)
                self.row_exit_goal = np.asarray(self.midline[-1], dtype=float)
                self.row_exit_heading_goal = np.asarray(self.midline[-1], dtype=float)
        else:
            self.plant_row_end_point = None
            self.row_exit_goal = None
            self.row_exit_heading_goal = None
            self.row_end_direction = None

    def update_frozen_prefix(self, model: RowMarchModel) -> None:
        if self.robot_pose is None or len(model.result.points) < 2:
            return
        robot_xy = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
        direction = self.normalize(model.initial_direction)
        along = (model.result.points - robot_xy) @ direction
        eligible = np.where(along <= -self.p.row_freeze_behind_distance)[0]
        if len(eligible) == 0:
            return
        freeze_count = int(eligible[-1]) + 1
        if freeze_count > len(model.frozen_points):
            model.frozen_points = np.array(model.result.points[:freeze_count], copy=True)
            if len(model.result.point_directions) >= freeze_count:
                model.frozen_directions = np.array(model.result.point_directions[:freeze_count], copy=True)
        model.result.frozen_count = len(model.frozen_points)

    def march_row(self, model: RowMarchModel, map_points: np.ndarray) -> RowMarchResult:
        result = RowMarchResult()
        if len(model.frozen_points) >= 2:
            exact_points = [np.asarray(point, dtype=float) for point in model.frozen_points]
            point1 = exact_points[-2]
            point2 = exact_points[-1]
            if len(model.frozen_directions) == len(model.frozen_points):
                exact_directions = [np.asarray(direction, dtype=float) for direction in model.frozen_directions]
                direction = self.normalize(exact_directions[-1])
            else:
                direction = self.initial_direction_from_points(point1, point2, model.initial_direction)
                exact_directions = [np.asarray(direction, dtype=float) for _ in exact_points]
            line_points = self.build_next_line_from_old_point3(point2, direction)
        else:
            point1 = np.asarray(model.initial_point1, dtype=float)
            point2 = np.asarray(model.initial_point2, dtype=float)
            direction = self.normalize(model.initial_direction)
            exact_points = [point1, point2]
            exact_directions = [np.asarray(direction, dtype=float), np.asarray(direction, dtype=float)]
            line_points = self.build_initial_line_from_point1_and_point2(point1, point2, direction)
        result.frozen_count = len(model.frozen_points)
        previous_valid_point: Optional[np.ndarray] = np.asarray(point2, dtype=float)
        last_valid_direction = np.asarray(direction, dtype=float)

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
                result.end_direction = np.asarray(last_valid_direction, dtype=float)
                break

            if len(local_point3_points) > 0:
                row_point = np.mean(local_point3_points, axis=0)
                previous_valid_point = row_point
            else:
                row_point = np.array(corrected_point3, copy=True)

            exact_points.append(row_point)
            exact_directions.append(np.asarray(corrected_direction, dtype=float))
            last_valid_direction = np.asarray(corrected_direction, dtype=float)
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
        result.point_directions = (
            np.asarray(exact_directions, dtype=float) if exact_directions else np.empty((0, 2), dtype=float)
        )
        result.current_line_points = np.asarray(line_points, dtype=float)
        if result.ended and result.end_point is None and len(result.points) > 0:
            result.end_point = result.points[-1]
        if result.end_direction is None:
            result.end_direction = np.asarray(last_valid_direction, dtype=float)
        return result

    def build_initial_line_from_point3(self, point3: np.ndarray, direction: np.ndarray) -> np.ndarray:
        spacing = self.p.row_segment_point_spacing
        direction = self.normalize(direction)
        return np.array([point3 + (idx - 2) * spacing * direction for idx in range(self.p.row_segment_point_count)], dtype=float)

    def build_initial_line_from_point1_and_point2(self, point1: np.ndarray, point2: np.ndarray, direction: np.ndarray) -> np.ndarray:
        spacing = self.p.row_segment_point_spacing
        direction = self.initial_direction_from_points(point1, point2, direction)
        point1 = np.asarray(point1, dtype=float)
        point2 = np.asarray(point2, dtype=float)
        points = [point1, point2]
        while len(points) < self.p.row_segment_point_count:
            points.append(points[-1] + spacing * direction)
        return np.asarray(points, dtype=float)

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

    def mean_direction(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first = self.normalize(first)
        second = self.normalize(second)
        if np.dot(first, second) < 0.0:
            second = -second
        return self.normalize(first + second)

    def append_polyline_point(self, polyline: np.ndarray, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        if len(polyline) == 0:
            return np.array([point], dtype=float)
        if float(np.linalg.norm(np.asarray(polyline[-1], dtype=float) - point)) < 1e-6:
            return polyline
        return np.vstack((polyline, point))

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

    def project_onto_polyline(self, polyline: np.ndarray, point: np.ndarray) -> Tuple[np.ndarray, int]:
        point = np.asarray(point, dtype=float)
        if len(polyline) < 2:
            return point, 0

        best_projection = np.asarray(polyline[0], dtype=float)
        best_segment_idx = 0
        best_distance = float("inf")
        for idx in range(len(polyline) - 1):
            segment = polyline[idx + 1] - polyline[idx]
            seg_len_sq = float(segment @ segment)
            if seg_len_sq < 1e-12:
                continue
            ratio = float(np.clip(((point - polyline[idx]) @ segment) / seg_len_sq, 0.0, 1.0))
            projection = polyline[idx] + ratio * segment
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_distance = distance
                best_projection = projection
                best_segment_idx = idx
        return np.asarray(best_projection, dtype=float), best_segment_idx

    def point_at_polyline_distance_from_projection(
        self,
        polyline: np.ndarray,
        projection: np.ndarray,
        start_idx: int,
        distance_ahead: float,
    ) -> np.ndarray:
        if len(polyline) < 2:
            return np.asarray(projection, dtype=float)

        start_idx = int(np.clip(start_idx, 0, len(polyline) - 2))
        travelled = 0.0
        segment_start = np.asarray(projection, dtype=float)
        for idx in range(start_idx, len(polyline) - 1):
            segment_end = np.asarray(polyline[idx + 1], dtype=float)
            segment = segment_end - segment_start
            seg_len = float(np.linalg.norm(segment))
            if seg_len >= 1e-9:
                if travelled + seg_len >= distance_ahead:
                    ratio = (distance_ahead - travelled) / seg_len
                    return segment_start + ratio * segment
                travelled += seg_len
            segment_start = segment_end
        return np.asarray(polyline[-1], dtype=float)

    def estimate_polyline_curvature_ahead(self, polyline: np.ndarray, point: np.ndarray) -> float:
        if len(polyline) < 3:
            return 0.0

        projection, segment_idx = self.project_onto_polyline(polyline, point)
        mid_distance = max(self.p.turn_lookahead_distance, 0.5 * self.p.lookahead_distance)
        p0 = projection
        p1 = self.point_at_polyline_distance_from_projection(polyline, projection, segment_idx, mid_distance)
        p2 = self.point_at_polyline_distance_from_projection(polyline, projection, segment_idx, self.p.lookahead_distance)

        a = float(np.linalg.norm(p1 - p0))
        b = float(np.linalg.norm(p2 - p1))
        c = float(np.linalg.norm(p2 - p0))
        denominator = a * b * c
        if denominator < 1e-9:
            return 0.0
        first = p1 - p0
        second = p2 - p0
        cross = float(first[0] * second[1] - first[1] * second[0])
        return 2.0 * cross / denominator

    def dynamic_follow_lookahead(self, polyline: np.ndarray, point: np.ndarray) -> float:
        curvature = self.estimate_polyline_curvature_ahead(polyline, point)
        lookahead = self.p.lookahead_distance / (1.0 + self.p.lookahead_curvature_gain * abs(curvature))
        lookahead = float(np.clip(lookahead, self.p.turn_lookahead_distance, self.p.lookahead_distance))
        self.current_lookahead_distance = lookahead
        self.current_lookahead_curvature = curvature
        return lookahead

    def lookahead_point_from_polyline_projection(
        self,
        polyline: np.ndarray,
        point: np.ndarray,
        distance_ahead: float,
    ) -> Optional[np.ndarray]:
        if len(polyline) == 0:
            return None
        if len(polyline) == 1:
            return np.asarray(polyline[0], dtype=float)

        point = np.asarray(point, dtype=float)
        best_projection: Optional[np.ndarray] = None
        best_segment_idx = 0
        best_distance = float("inf")
        for idx in range(len(polyline) - 1):
            segment = polyline[idx + 1] - polyline[idx]
            seg_len_sq = float(segment @ segment)
            if seg_len_sq < 1e-12:
                continue
            ratio = float(np.clip(((point - polyline[idx]) @ segment) / seg_len_sq, 0.0, 1.0))
            projection = polyline[idx] + ratio * segment
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_distance = distance
                best_projection = projection
                best_segment_idx = idx

        if best_projection is None:
            return np.asarray(polyline[-1], dtype=float)

        travelled = 0.0
        segment_start = best_projection
        for idx in range(best_segment_idx, len(polyline) - 1):
            segment_end = polyline[idx + 1]
            segment = segment_end - segment_start
            seg_len = float(np.linalg.norm(segment))
            if seg_len >= 1e-9:
                if travelled + seg_len >= distance_ahead:
                    ratio = (distance_ahead - travelled) / seg_len
                    return segment_start + ratio * segment
                travelled += seg_len
            segment_start = segment_end
        return np.asarray(polyline[-1], dtype=float)

    def drive_to_point(
        self,
        target: np.ndarray,
        max_speed: Optional[float] = None,
        min_speed: Optional[float] = None,
    ) -> None:
        target = np.asarray(target, dtype=float)
        if self.last_target_point is None:
            filtered_target = target
        else:
            alpha = self.p.target_filter_alpha
            filtered_target = (1.0 - alpha) * self.last_target_point + alpha * target
        self.last_target_point = np.asarray(filtered_target, dtype=float)

        dx = float(filtered_target[0] - self.robot_pose.x)
        dy = float(filtered_target[1] - self.robot_pose.y)
        target_distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_error = wrap_to_pi(target_yaw - self.robot_pose.yaw)

        cmd = Twist()
        max_speed = self.p.follow_speed if max_speed is None else max_speed
        min_speed = self.p.slow_speed if min_speed is None else min_speed
        pursuit_distance = max(0.10, target_distance)
        curvature = self.p.pure_pursuit_gain * 2.0 * math.sin(yaw_error) / pursuit_distance
        cmd.linear.x = float(np.clip(
            max_speed / (1.0 + self.p.curve_speed_reduction_gain * abs(curvature)),
            min_speed,
            max_speed,
        ))
        radius_limited_angular = abs(cmd.linear.x) / self.p.min_follow_turn_radius
        max_angular = min(self.p.max_angular_speed, radius_limited_angular)
        # Pure-pursuit curvature is calmer around the midline than a direct
        # proportional yaw correction and still works for headland waypoints.
        pursuit_angular = cmd.linear.x * curvature
        angular_raw = float(np.clip(pursuit_angular, -max_angular, max_angular))

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

    def get_map_points_in_hist_roi(self, pose: Pose2D) -> np.ndarray:
        points = self.get_all_map_points()
        if len(points) == 0:
            return points
        center = np.array([pose.x, pose.y], dtype=float)
        forward = self.yaw_to_vector(pose.yaw)
        return self.points_in_oriented_rectangle(
            points,
            center,
            forward,
            self.p.hist_roi_depth,
            self.p.hist_roi_width,
        )

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
        for row_number in sorted(self.stored_rows):
            markers.markers.append(
                self.create_line_marker(
                    f"stored_row_{row_number}",
                    marker_id,
                    self.stored_rows[row_number],
                    (0.65, 0.75, 0.82, 0.65),
                    0.025,
                    stamp,
                )
            )
            marker_id += 1
        if self.left_row is not None:
            marker_id = self.add_row_markers(markers, marker_id, self.left_row, (0.1, 0.95, 0.2, 1.0), stamp)
        if self.right_row is not None:
            marker_id = self.add_row_markers(markers, marker_id, self.right_row, (1.0, 0.55, 0.05, 1.0), stamp)
        if len(self.midline) > 0:
            markers.markers.append(self.create_line_marker("march_midline", marker_id, self.midline, (0.15, 0.45, 1.0, 1.0), 0.055, stamp))
            marker_id += 1
        if self.robot_pose is not None and self.p.laser_follow_enabled:
            marker_id = self.add_laser_follow_markers(markers, marker_id, stamp)
        if self.fused_target_point is not None:
            markers.markers.append(self.create_sphere_marker("fused_target_point", marker_id, self.fused_target_point, (0.95, 0.25, 0.95, 1.0), 0.14, stamp))
            marker_id += 1
        if self.last_target_point is not None:
            markers.markers.append(self.create_sphere_marker("filtered_target_point", marker_id, self.last_target_point, (1.0, 0.05, 1.0, 1.0), 0.18, stamp))
            marker_id += 1
            if self.robot_pose is not None:
                robot_point = np.array([self.robot_pose.x, self.robot_pose.y], dtype=float)
                target_line = np.vstack((robot_point, self.last_target_point))
                markers.markers.append(self.create_line_marker("robot_to_filtered_target", marker_id, target_line, (1.0, 0.05, 1.0, 0.75), 0.025, stamp))
                marker_id += 1
        if self.plant_row_end_point is not None:
            markers.markers.append(self.create_sphere_marker("plant_row_end", marker_id, self.plant_row_end_point, (1.0, 0.0, 0.0, 1.0), 0.22, stamp))
            marker_id += 1
        if self.row_exit_goal is not None:
            markers.markers.append(self.create_sphere_marker("row_exit_goal", marker_id, self.row_exit_goal, (0.7, 0.0, 1.0, 1.0), 0.24, stamp))
            marker_id += 1
        if self.row_exit_heading_goal is not None:
            markers.markers.append(
                self.create_sphere_marker("row_exit_heading_goal", marker_id, self.row_exit_heading_goal, (0.7, 0.0, 1.0, 0.55), 0.13, stamp)
            )
            marker_id += 1
        if self.entrance_hist_center is not None and self.entrance_hist_direction is not None:
            markers.markers.append(
                self.create_rectangle_marker(
                    "entrance_histogram_roi",
                    marker_id,
                    self.entrance_hist_center,
                    self.entrance_hist_direction,
                    self.p.hist_roi_depth,
                    self.p.hist_roi_width,
                    (0.1, 1.0, 0.9, 0.8),
                    stamp,
                )
            )
            marker_id += 1
            markers.markers.append(
                self.create_arrow_marker(
                    "entrance_histogram_direction",
                    marker_id,
                    self.entrance_hist_center,
                    self.entrance_hist_direction,
                    (0.1, 1.0, 0.9, 1.0),
                    stamp,
                )
            )
            marker_id += 1
        for peak in self.entrance_hist_peaks:
            color = (0.0, 1.0, 0.25, 1.0) if peak.selected else (0.0, 0.85, 1.0, 0.9)
            scale = 0.24 if peak.selected else 0.16
            markers.markers.append(self.create_sphere_marker("entrance_histogram_peaks", marker_id, peak.point, color, scale, stamp))
            marker_id += 1
            label = "?" if peak.row_number is None else str(peak.row_number)
            markers.markers.append(self.create_text_marker("entrance_histogram_row_numbers", marker_id, peak.point, label, color, stamp))
            marker_id += 1
        for waypoint in self.entrance_waypoints:
            markers.markers.append(
                self.create_sphere_marker("entrance_waypoints", marker_id, waypoint, (1.0, 0.9, 0.1, 1.0), 0.24, stamp)
            )
            marker_id += 1
        if len(self.entrance_follow_path) > 0:
            markers.markers.append(
                self.create_line_marker("entrance_follow_path", marker_id, self.entrance_follow_path, (0.1, 1.0, 0.35, 0.9), 0.06, stamp)
            )
            marker_id += 1
        if len(self.entrance_route_support) > 1:
            markers.markers.append(
                self.create_line_marker(
                    "entrance_route_support",
                    marker_id,
                    self.entrance_route_support,
                    (1.0, 0.95, 0.1, 0.55),
                    0.035,
                    stamp,
                )
            )
            marker_id += 1
        if len(self.entrance_route) > 0:
            markers.markers.append(
                self.create_line_marker("entrance_turn_route", marker_id, self.entrance_route, (1.0, 0.75, 0.0, 0.9), 0.055, stamp)
            )
            marker_id += 1
        if self.entrance_route_projection is not None:
            markers.markers.append(
                self.create_sphere_marker("entrance_route_projection", marker_id, self.entrance_route_projection, (0.1, 1.0, 0.9, 1.0), 0.15, stamp)
            )
            marker_id += 1
        if self.entrance_route_target is not None:
            markers.markers.append(
                self.create_sphere_marker("entrance_route_lookahead", marker_id, self.entrance_route_target, (1.0, 0.2, 0.8, 1.0), 0.17, stamp)
            )
            marker_id += 1
            markers.markers.append(
                self.create_text_marker(
                    "entrance_route_status",
                    marker_id,
                    self.entrance_route_target,
                    f"idx={self.entrance_route_progress_index} remaining={self.entrance_route_remaining_distance:.2f}",
                    (1.0, 1.0, 1.0, 1.0),
                    stamp,
                )
            )
            marker_id += 1
        if self.entrance_target is not None:
            markers.markers.append(self.create_sphere_marker("entrance_target", marker_id, self.entrance_target, (0.9, 0.1, 1.0, 1.0), 0.26, stamp))
            marker_id += 1
            if self.entrance_heading_goal is not None:
                markers.markers.append(
                    self.create_sphere_marker("entrance_heading_goal", marker_id, self.entrance_heading_goal, (0.9, 0.1, 1.0, 0.55), 0.13, stamp)
                )
                marker_id += 1
            if self.entrance_target_direction is not None:
                markers.markers.append(
                    self.create_arrow_marker(
                        "entrance_target_direction",
                        marker_id,
                        self.entrance_target,
                        self.entrance_target_direction,
                        (0.9, 0.1, 1.0, 1.0),
                        stamp,
                    )
                )

        self.marker_pub.publish(markers)

    def add_laser_follow_markers(self, markers: MarkerArray, marker_id: int, stamp) -> int:
        x_min = self.p.laser_roi_x_min
        x_max = self.p.laser_roi_x_max
        result = self.laser_follow_result
        roi_centers = result.roi_centers
        roi_direction = result.roi_direction
        if len(roi_centers) != 2:
            roi_centers, roi_direction = self.laser_follower.build_rois(None, np.array([x_max, 0.0], dtype=float))
        for center_base, namespace in zip(roi_centers, ("laser_left_roi", "laser_right_roi")):
            markers.markers.append(
                self.create_rectangle_marker(
                    namespace,
                    marker_id,
                    self.base_point_to_map(center_base),
                    self.base_direction_to_map(roi_direction),
                    self.p.laser_roi_length,
                    self.p.laser_roi_width,
                    (0.2, 0.95, 0.95, 0.65),
                    stamp,
                )
            )
            marker_id += 1

        for line, namespace, color in (
            (result.left_line, "laser_left_line", (0.0, 1.0, 0.25, 1.0)),
            (result.right_line, "laser_right_line", (1.0, 0.55, 0.0, 1.0)),
        ):
            if not line.valid:
                continue
            line_base = np.array(
                [[x_min, line.slope * x_min + line.intercept], [x_max, line.slope * x_max + line.intercept]],
                dtype=float,
            )
            markers.markers.append(self.create_line_marker(namespace, marker_id, self.base_points_to_map(line_base), color, 0.045, stamp))
            marker_id += 1

        if result.valid:
            centerline_base = np.array(
                [
                    [x_min, result.center_slope * x_min + result.center_intercept],
                    [x_max, result.center_slope * x_max + result.center_intercept],
                ],
                dtype=float,
            )
            markers.markers.append(
                self.create_line_marker(
                    "laser_centerline",
                    marker_id,
                    self.base_points_to_map(centerline_base),
                    (0.95, 0.15, 0.95, 1.0),
                    0.055,
                    stamp,
                )
            )
            marker_id += 1

        status_point = self.base_point_to_map(np.array([0.35, 0.0], dtype=float))
        status = f"laser conf={result.confidence:.2f} weight={result.weight:.2f} {result.reason}"
        markers.markers.append(self.create_text_marker("laser_follow_status", marker_id, status_point, status, (1.0, 1.0, 1.0, 1.0), stamp))
        return marker_id + 1

    def base_points_to_map(self, points: np.ndarray) -> np.ndarray:
        return np.asarray([self.base_point_to_map(point) for point in points], dtype=float)

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
            if result.frozen_count > 0:
                markers.markers.append(
                    self.create_line_marker(
                        f"{model.side}_row_frozen",
                        marker_id,
                        result.points[: result.frozen_count],
                        (0.72, 0.72, 0.82, 0.95),
                        0.065,
                        stamp,
                    )
                )
                marker_id += 1
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

    def create_arrow_marker(
        self,
        namespace: str,
        marker_id: int,
        point: np.ndarray,
        direction: np.ndarray,
        color: Tuple[float, float, float, float],
        stamp,
    ) -> Marker:
        marker = Marker(header=Header(frame_id=self.p.map_frame, stamp=stamp))
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.55
        marker.scale.y = 0.10
        marker.scale.z = 0.10
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.orientation.z = math.sin(0.5 * math.atan2(float(direction[1]), float(direction[0])))
        marker.pose.orientation.w = math.cos(0.5 * math.atan2(float(direction[1]), float(direction[0])))
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
