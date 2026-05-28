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
    def __init__(self, points: np.ndarray, heading_yaw: float = 0.0, s: float = 0.15):
        """
        points: (N, 2) array of (x, y) coordinates in map frame
        heading_yaw: Approximate current heading yaw of the robot to align spline direction
        s: Balanced smoothing factor to allow tracking real curves while filtering high-frequency noise
        """
        self.valid = False
        self.t_min = 0.0
        self.t_max = 0.0
        self.points = points
        
        if len(points) < 4:
            return
            
        # PCA to find main direction
        self.mean = np.mean(points, axis=0)
        centered = points - self.mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        self.main_dir = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Align spline direction with the approximate robot heading
        heading_vec = np.array([math.cos(heading_yaw), math.sin(heading_yaw)])
        if np.dot(self.main_dir, heading_vec) < 0:
            self.main_dir = -self.main_dir
            
        # Project points onto main direction to sort them
        projections = centered @ self.main_dir
        sort_idx = np.argsort(projections)
        self.points = points[sort_idx]
        
        self.t = projections[sort_idx]
        # Remove duplicate t values for spline fitting
        unique_t, unique_idx = np.unique(self.t, return_index=True)
        if len(unique_t) < 4:
            return
            
        # Adaptiver Splinegrad wie in der Referenz: bei wenigen Punkten keinen zu hohen Grad erzwingen.
        # Außerdem adaptives Glätten: s wird skaliert mit Anzahl der Punkte und räumlicher Streuung,
        # damit Splines bei dichterem / lauterem Rauschen stärker geglättet werden.
        try:
            k = min(3, len(unique_t) - 1)
            spread = float(np.mean(np.std(self.points[unique_idx], axis=0)))
            s_used = float(s) * max(1.0, len(unique_t) / 8.0) * (1.0 + spread)
            self.x_spline = UnivariateSpline(unique_t, self.points[unique_idx, 0], k=k, s=s_used)
            self.y_spline = UnivariateSpline(unique_t, self.points[unique_idx, 1], k=k, s=s_used)
            self.valid = True
            self.t_min = unique_t[0]
            self.t_max = unique_t[-1]
        except Exception:
            self.valid = False
            return

    def evaluate(self, t: float) -> Tuple[float, float]:
        return float(self.x_spline(t)), float(self.y_spline(t))

    def get_direction(self, t: float) -> float:
        dx = self.x_spline.derivative()(t)
        dy = self.y_spline.derivative()(t)
        return math.atan2(dy, dx)

    def get_global_direction(self) -> float:
        """Global row trend from PCA main_dir"""
        return float(math.atan2(self.main_dir[1], self.main_dir[0]))

    def get_points(self, num: int = 50) -> np.ndarray:
        t_vals = np.linspace(self.t_min, self.t_max, num)
        return np.column_stack((self.x_spline(t_vals), self.y_spline(t_vals)))
    
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
    
    control_frequency: float = 20.0
    expected_row_width: float = 0.75
    
    # Histogram params
    hist_roi_size: float = 5.0
    hist_bin_size: float = 0.05
    occ_threshold: int = 50
    
    # Driving params
    follow_speed: float = 0.35
    turn_speed: float = 0.25
    exit_distance: float = 0.5
    min_turn_radius: float = 0.5
    lookahead_dist: float = 0.6
    
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
        
        left_peaks = [p for p in peaks if p > 0.15]
        right_peaks = [p for p in peaks if p < -0.15]
        
        if not left_peaks or not right_peaks: 
            self.get_logger().info(f"Missing side rows. Left peaks: {left_peaks}, Right peaks: {right_peaks}", throttle_duration_sec=2.0)
            return
            
        best_left, best_right = min(left_peaks), max(right_peaks)
        
        self.left_row_points = points[np.abs(local_y - best_left) < 0.25]
        self.right_row_points = points[np.abs(local_y - best_right) < 0.25]
        
        self.left_row_spline = PlantSpline(self.left_row_points, heading_yaw=self.robot_pose.yaw)
        self.right_row_spline = PlantSpline(self.right_row_points, heading_yaw=self.robot_pose.yaw)
        
        if self.left_row_spline.valid and self.right_row_spline.valid:
            self.row_entry_pose = self.robot_pose
            self.state = MissionState.FOLLOW_ROW
            self.get_logger().info(f"Rows found. Width: {best_left - best_right:.2f}m. Following...")

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
        
        new_pts_l = self.find_points_along_tangent(end_val_l, weighted_yaw_l, max_dist=2.2, width=0.45)
        if len(new_pts_l) > 0:
            self.left_row_points = self.accumulate_unique_points(self.left_row_points, new_pts_l)
            temp_spline = PlantSpline(self.left_row_points, heading_yaw=self.robot_pose.yaw)
            if temp_spline.valid:
                self.left_row_spline = temp_spline

        # For Right Spline:
        end_val_r = np.array(self.right_row_spline.evaluate(self.right_row_spline.t_max))
        local_tangent_r = self.right_row_spline.get_direction(self.right_row_spline.t_max)
        global_trend_r = self.right_row_spline.get_global_direction()
        weighted_yaw_r = wrap_to_pi(0.4 * local_tangent_r + 0.6 * global_trend_r)
        
        new_pts_r = self.find_points_along_tangent(end_val_r, weighted_yaw_r, max_dist=2.2, width=0.45)
        if len(new_pts_r) > 0:
            self.right_row_points = self.accumulate_unique_points(self.right_row_points, new_pts_r)
            temp_spline = PlantSpline(self.right_row_points, heading_yaw=self.robot_pose.yaw)
            if temp_spline.valid:
                self.right_row_spline = temp_spline

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

    def find_points_along_tangent(self, start_pos: np.ndarray, yaw: float, max_dist: float, width: float) -> np.ndarray:
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
        return points[mask]

    def accumulate_unique_points(self, current_pts: np.ndarray, new_pts: np.ndarray) -> np.ndarray:
        """Filters new points to only append those that aren't already represented in current_pts."""
        # Simple spatial proximity filter: keep points that are at least 0.08m away from any existing point
        kept = []
        for pt in new_pts:
            dists = np.linalg.norm(current_pts - pt, axis=1)
            if np.min(dists) > 0.08:
                kept.append(pt)
        if len(kept) > 0:
            return np.vstack((current_pts, np.array(kept)))
        return current_pts

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

        self.get_logger().info("Hallooooooooooooooooooooo")
        
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
        if self.left_row_spline and self.left_row_spline.valid:
            markers.markers.append(self.create_spline_marker(self.left_row_spline, 0, [0.0, 1.0, 0.0]))
        if self.right_row_spline and self.right_row_spline.valid:
            markers.markers.append(self.create_spline_marker(self.right_row_spline, 1, [0.0, 1.0, 0.0]))
        self.marker_pub.publish(markers)

    def create_spline_marker(self, spline: PlantSpline, id: int, color: list) -> Marker:
        m = Marker(header=Header(frame_id=self.p.map_frame, stamp=self.get_clock().now().to_msg()))
        m.ns, m.id, m.type, m.action = "splines", id, Marker.LINE_STRIP, Marker.ADD
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], 1.0
        for p in spline.get_points():
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
