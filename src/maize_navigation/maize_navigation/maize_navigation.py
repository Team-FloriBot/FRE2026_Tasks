#!/usr/bin/env python3
"""
maize_navigation.py

Ein einzelner ROS-2-Node fuer Maisreihen-Navigation.

Architektur innerhalb dieses Nodes:
    LaserScan
      -> RowPerception
      -> RowTracker
      -> MissionStateMachine
      -> LocalPlanner
      -> PathFollower
      -> SafetySupervisor
      -> cmd_vel

Keine eigenen ROS-Messages notwendig.
Debug-Ausgaben:
    /debug/state
    /debug/row_confidence
    /debug/end_probability
    /debug/local_path
    /debug/row_markers

Wichtige Annahmen:
    - LaserScan liegt geometrisch in einem Frame, der base_link entspricht,
      oder wurde vorher bereits korrekt gemerged/transformiert.
    - +x zeigt nach vorne, +y nach links.
    - Fuer Wendepfade wird odom -> base_link per TF benoetigt.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Point, PoseStamped, Quaternion, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


# =============================================================================
# Kleine Hilfsfunktionen
# =============================================================================


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(0.5 * yaw)
    q.w = math.cos(0.5 * yaw)
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    # yaw aus Quaternion, ausreichend fuer planaren Roboter
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_point_to_base(px: float, py: float, robot_pose_odom: "Pose2D") -> Tuple[float, float]:
    """Punkt aus odom/map in base_link transformieren."""
    dx = px - robot_pose_odom.x
    dy = py - robot_pose_odom.y
    c = math.cos(-robot_pose_odom.yaw)
    s = math.sin(-robot_pose_odom.yaw)
    bx = c * dx - s * dy
    by = s * dx + c * dy
    return bx, by


def transform_point_from_base(px: float, py: float, robot_pose_odom: "Pose2D") -> Tuple[float, float]:
    """Punkt aus base_link in odom/map transformieren."""
    c = math.cos(robot_pose_odom.yaw)
    s = math.sin(robot_pose_odom.yaw)
    ox = robot_pose_odom.x + c * px - s * py
    oy = robot_pose_odom.y + s * px + c * py
    return ox, oy


# =============================================================================
# Datenklassen
# =============================================================================


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class LineFit:
    valid: bool = False
    a: float = 0.0       # y = a*x + b
    b: float = 0.0
    inliers: int = 0
    length_x: float = 0.0
    rmse: float = 0.0


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

    points_left: List[Point2D] = field(default_factory=list)
    points_right: List[Point2D] = field(default_factory=list)
    points_all: List[Point2D] = field(default_factory=list)


@dataclass
class RowModel:
    valid: bool = False
    confidence: float = 0.0

    center_a: float = 0.0
    center_b: float = 0.0
    row_yaw: float = 0.0
    row_width: float = 0.75

    end_probability: float = 0.0
    end_detected: bool = False
    missing_frames: int = 0


@dataclass
class PathPoint:
    x: float
    y: float
    yaw: float
    v: float


@dataclass
class LocalPath:
    points: List[PathPoint] = field(default_factory=list)
    valid: bool = False
    frame_id: str = "base_link"   # "base_link" oder odom_frame
    reason: str = ""


class MissionState(Enum):
    IDLE = 0
    FOLLOW_ROW = 1
    EXIT_ROW = 2
    PLAN_TURN = 3
    EXECUTE_TURN = 4
    ENTER_ROW = 5
    FINISHED = 6


@dataclass
class PatternStep:
    count: int
    direction: int  # +1 links, -1 rechts


# =============================================================================
# Parameter
# =============================================================================


@dataclass
class NavParams:
    # Frames / Topics
    scan_topic: str = "/sensors/merged_scan"
    cmd_vel_topic: str = "/cmd_vel"
    base_frame: str = "base_link"
    odom_frame: str = "odom"
    control_rate_hz: float = 20.0

    # Mission
    pattern: str = "1L-1R-2L-3R"
    auto_start: bool = False

    # Perception ROI
    roi_x_min: float = 0.15
    roi_x_max: float = 4.0
    roi_y_abs_min: float = 0.12
    roi_y_abs_max: float = 1.50
    front_density_x_min: float = 0.5
    front_density_x_max: float = 2.5
    front_density_y_abs: float = 0.35

    # RANSAC / Reihenfit
    ransac_iterations: int = 80
    ransac_distance: float = 0.06
    min_inliers: int = 5
    min_visible_length: float = 0.50
    expected_row_width: float = 0.75
    min_row_width: float = 0.45
    max_row_width: float = 1.10
    max_row_angle_diff_deg: float = 15.0

    # Tracking
    tracker_alpha: float = 0.25
    confidence_decay_no_measurement: float = 0.95
    min_detection_confidence_update: float = 0.25
    end_detect_threshold: float = 0.75

    # Lokaler Planer
    row_lookahead_distance: float = 3.0
    row_path_points: int = 40
    nominal_speed: float = 0.35
    slow_speed: float = 0.12

    # Vorgewende / Turn
    exit_distance: float = 0.80
    turn_forward_distance: float = 2.20
    enter_distance: float = 1.20
    row_spacing: float = 0.75
    turn_speed: float = 0.20
    enter_speed: float = 0.18
    min_enter_confidence: float = 0.45
    enter_stable_frames_required: int = 8

    # Controller
    lookahead_distance: float = 0.80
    max_linear_speed: float = 0.45
    max_angular_speed: float = 1.20
    max_linear_accel: float = 0.50
    max_angular_accel: float = 1.80
    path_goal_tolerance_xy: float = 0.12
    path_goal_tolerance_yaw_deg: float = 12.0

    # Safety
    obstacle_stop_distance: float = 0.25
    obstacle_slow_distance: float = 0.45
    front_obstacle_angle_deg: float = 25.0
    min_follow_confidence_stop: float = 0.10
    min_follow_confidence_slow: float = 0.25


# =============================================================================
# Row Perception
# =============================================================================


class RowPerception:
    def __init__(self, params: NavParams):
        self.p = params
        self.rng = random.Random(42)

    def process_scan(self, scan: LaserScan) -> RowDetection:
        points = self.scan_to_points(scan)
        points = self.filter_roi(points)

        left = [pt for pt in points if pt.y > self.p.roi_y_abs_min]
        right = [pt for pt in points if pt.y < -self.p.roi_y_abs_min]

        left_fit = self.fit_line_ransac(left)
        right_fit = self.fit_line_ransac(right)

        det = RowDetection(
            points_left=left,
            points_right=right,
            points_all=points,
        )

        if left_fit.valid:
            det.left_valid = True
            det.left_a = left_fit.a
            det.left_b = left_fit.b

        if right_fit.valid:
            det.right_valid = True
            det.right_a = right_fit.a
            det.right_b = right_fit.b

        self.compute_centerline(det)
        self.compute_confidence(det, left_fit, right_fit)
        self.compute_end_probability(det)
        return det

    def scan_to_points(self, scan: LaserScan) -> List[Point2D]:
        points: List[Point2D] = []
        angle = scan.angle_min

        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append(Point2D(x, y))
            angle += scan.angle_increment

        return points

    def filter_roi(self, points: Sequence[Point2D]) -> List[Point2D]:
        out: List[Point2D] = []
        for pt in points:
            if pt.x < self.p.roi_x_min or pt.x > self.p.roi_x_max:
                continue
            ay = abs(pt.y)
            if ay < self.p.roi_y_abs_min or ay > self.p.roi_y_abs_max:
                continue
            out.append(pt)
        return out

    def fit_line_ransac(self, points: Sequence[Point2D]) -> LineFit:
        if len(points) < self.p.min_inliers:
            return LineFit(valid=False)

        pts = list(points)
        best_inliers: List[Point2D] = []
        best_a = 0.0
        best_b = 0.0

        for _ in range(self.p.ransac_iterations):
            p1, p2 = self.rng.sample(pts, 2)
            dx = p2.x - p1.x
            if abs(dx) < 1e-3:
                continue

            a = (p2.y - p1.y) / dx
            b = p1.y - a * p1.x

            inliers = []
            denom = math.sqrt(a * a + 1.0)
            for p in pts:
                dist = abs(a * p.x - p.y + b) / denom
                if dist <= self.p.ransac_distance:
                    inliers.append(p)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_a = a
                best_b = b

        if len(best_inliers) < self.p.min_inliers:
            return LineFit(valid=False)

        xs = np.array([p.x for p in best_inliers], dtype=float)
        ys = np.array([p.y for p in best_inliers], dtype=float)
        length_x = float(np.max(xs) - np.min(xs)) if len(xs) > 1 else 0.0
        if length_x < self.p.min_visible_length:
            return LineFit(valid=False)

        # finaler Least-Squares-Fit auf Inliern
        a, b = np.polyfit(xs, ys, 1)
        residuals = ys - (a * xs + b)
        rmse = float(np.sqrt(np.mean(residuals * residuals)))

        return LineFit(
            valid=True,
            a=float(a),
            b=float(b),
            inliers=len(best_inliers),
            length_x=length_x,
            rmse=rmse,
        )

    def compute_centerline(self, det: RowDetection) -> None:
        expected = self.p.expected_row_width

        if det.left_valid and det.right_valid:
            det.center_a = 0.5 * (det.left_a + det.right_a)
            det.center_b = 0.5 * (det.left_b + det.right_b)
            det.lane_width = det.left_b - det.right_b
            return

        if det.left_valid and not det.right_valid:
            det.center_a = det.left_a
            det.center_b = det.left_b - 0.5 * expected
            det.lane_width = expected
            return

        if det.right_valid and not det.left_valid:
            det.center_a = det.right_a
            det.center_b = det.right_b + 0.5 * expected
            det.lane_width = expected
            return

        det.center_a = 0.0
        det.center_b = 0.0
        det.lane_width = 0.0

    def compute_confidence(self, det: RowDetection, left_fit: LineFit, right_fit: LineFit) -> None:
        confidence = 0.0

        if det.left_valid:
            confidence += 0.30
        if det.right_valid:
            confidence += 0.30

        if det.left_valid and det.right_valid:
            width_ok = self.p.min_row_width <= det.lane_width <= self.p.max_row_width
            angle_diff = abs(math.atan(det.left_a) - math.atan(det.right_a))
            angle_ok = angle_diff <= math.radians(self.p.max_row_angle_diff_deg)

            if width_ok:
                confidence += 0.20
            if angle_ok:
                confidence += 0.10
        else:
            # Eine Seite ist besser als nichts, aber weniger sicher.
            confidence += 0.10

        length_bonus = 0.0
        if left_fit.valid:
            length_bonus += clamp(left_fit.length_x / self.p.roi_x_max, 0.0, 0.10)
        if right_fit.valid:
            length_bonus += clamp(right_fit.length_x / self.p.roi_x_max, 0.0, 0.10)
        confidence += length_bonus

        det.confidence = clamp(confidence, 0.0, 1.0)

    def compute_end_probability(self, det: RowDetection) -> None:
        front_points = [
            pt for pt in det.points_all
            if self.p.front_density_x_min <= pt.x <= self.p.front_density_x_max
            and abs(pt.y) <= self.p.front_density_y_abs
        ]

        front_density = len(front_points)
        side_count = len(det.points_left) + len(det.points_right)

        score = 0.0
        if front_density < 4:
            score += 0.35
        if not det.left_valid and not det.right_valid:
            score += 0.30
        elif det.left_valid != det.right_valid:
            score += 0.15
        if side_count < 8:
            score += 0.20
        if det.confidence < 0.35:
            score += 0.15

        det.end_probability = clamp(score, 0.0, 1.0)


# =============================================================================
# Row Tracker
# =============================================================================


class RowTracker:
    def __init__(self, params: NavParams):
        self.p = params
        self.model = RowModel(row_width=params.expected_row_width)

    def update(self, det: RowDetection) -> RowModel:
        if det.confidence >= self.p.min_detection_confidence_update:
            if not self.model.valid:
                self.model.valid = True
                self.model.center_a = det.center_a
                self.model.center_b = det.center_b
                self.model.row_width = det.lane_width if det.lane_width > 0.0 else self.p.expected_row_width
                self.model.confidence = det.confidence
            else:
                a = self.p.tracker_alpha
                self.model.center_a = (1.0 - a) * self.model.center_a + a * det.center_a
                self.model.center_b = (1.0 - a) * self.model.center_b + a * det.center_b
                if det.lane_width > 0.0:
                    self.model.row_width = (1.0 - a) * self.model.row_width + a * det.lane_width
                self.model.confidence = clamp(
                    0.80 * self.model.confidence + 0.20 * det.confidence,
                    0.0,
                    1.0,
                )

            self.model.row_yaw = math.atan(self.model.center_a)
            self.model.missing_frames = 0
        else:
            self.model.missing_frames += 1
            self.model.confidence *= self.p.confidence_decay_no_measurement

        self.model.end_probability = clamp(
            0.85 * self.model.end_probability + 0.15 * det.end_probability,
            0.0,
            1.0,
        )
        self.model.end_detected = self.model.end_probability >= self.p.end_detect_threshold
        return self.model


# =============================================================================
# Mission State Machine
# =============================================================================


class MissionStateMachine:
    def __init__(self, params: NavParams):
        self.p = params
        self.state = MissionState.FOLLOW_ROW if params.auto_start else MissionState.IDLE
        self.pattern = self.parse_pattern(params.pattern)
        self.pattern_index = 0
        self.active_path: Optional[LocalPath] = None
        self.enter_stable_frames = 0
        self.last_transition_reason = "init"

    def parse_pattern(self, pattern: str) -> List[PatternStep]:
        steps: List[PatternStep] = []
        if not pattern:
            return steps

        for token in pattern.split("-"):
            token = token.strip().upper()
            if len(token) < 2:
                continue
            direction_char = token[-1]
            number_text = token[:-1]
            try:
                count = int(number_text)
            except ValueError:
                continue

            if direction_char == "L":
                direction = +1
            elif direction_char == "R":
                direction = -1
            else:
                continue

            if count > 0:
                steps.append(PatternStep(count=count, direction=direction))
        return steps

    def start(self) -> None:
        self.pattern_index = 0
        self.enter_stable_frames = 0
        self.active_path = None
        self.transition(MissionState.FOLLOW_ROW, "start service")

    def stop(self) -> None:
        self.active_path = None
        self.transition(MissionState.IDLE, "stop service")

    def current_step(self) -> Optional[PatternStep]:
        if self.pattern_index >= len(self.pattern):
            return None
        return self.pattern[self.pattern_index]

    def advance_step(self) -> None:
        self.pattern_index += 1

    def transition(self, new_state: MissionState, reason: str) -> None:
        if new_state != self.state:
            self.state = new_state
            self.last_transition_reason = reason


# =============================================================================
# Local Planner
# =============================================================================


class LocalPlanner:
    def __init__(self, params: NavParams):
        self.p = params

    def plan_follow_row(self, row: RowModel) -> LocalPath:
        if not row.valid or row.confidence < 0.05:
            return LocalPath(valid=False, frame_id="base_link", reason="row model invalid")

        n = max(2, self.p.row_path_points)
        points: List[PathPoint] = []

        speed = self.p.nominal_speed
        if row.confidence < self.p.min_follow_confidence_slow:
            speed = self.p.slow_speed

        for x in np.linspace(0.25, self.p.row_lookahead_distance, n):
            y = row.center_a * float(x) + row.center_b
            yaw = math.atan(row.center_a)
            points.append(PathPoint(float(x), float(y), yaw, speed))

        return LocalPath(points=points, valid=True, frame_id="base_link")

    def plan_exit_path(self, pose: Pose2D) -> LocalPath:
        points: List[PathPoint] = []
        n = 18
        for d in np.linspace(0.0, self.p.exit_distance, n):
            x = pose.x + float(d) * math.cos(pose.yaw)
            y = pose.y + float(d) * math.sin(pose.yaw)
            points.append(PathPoint(x, y, pose.yaw, self.p.turn_speed))
        return LocalPath(points=points, valid=True, frame_id=self.p.odom_frame)

    def plan_turn_path(self, pose: Pose2D, step: PatternStep) -> LocalPath:
        """
        Erzeugt eine feste S-Kurven-Primitive im odom-Frame.

        Interpretation:
            direction +1: Zielreihe links vom Roboter
            direction -1: Zielreihe rechts vom Roboter

        Der Pfad ist bewusst einfach:
            - vorwaerts im Vorgewende
            - lateraler Versatz mit glatter kubischer Kurve
            - Ausrichtung bleibt lokal nach vorne; fuer echte 180-Grad-Umkehr
              kann target_yaw = pose.yaw + pi verwendet und eine Dubins-
              oder Reeds-Shepp-Primitive ergaenzt werden.

        Fuer viele FRE-Setups mit parallelen Reihen und vorwaerts gerichteter
        Einfahrt ist diese Primitive ein guter erster Test. Falls der Roboter
        nach dem Vorgewende in Gegenrichtung in die naechste Reihe fahren muss,
        siehe TODO im Code unten.
        """
        lateral_shift = step.direction * step.count * self.p.row_spacing
        forward = self.p.turn_forward_distance
        points: List[PathPoint] = []

        # Start im Roboterframe, dann nach odom transformieren.
        local_points: List[PathPoint] = []

        for x in np.linspace(0.0, 0.4, 8):
            local_points.append(PathPoint(float(x), 0.0, 0.0, self.p.turn_speed))

        # glatte S-Kurve y(t) = shift * (3t^2 - 2t^3)
        for t in np.linspace(0.0, 1.0, 50):
            x = 0.4 + forward * float(t)
            y = lateral_shift * (3.0 * t * t - 2.0 * t * t * t)
            dy_dt = lateral_shift * (6.0 * t - 6.0 * t * t)
            dx_dt = forward
            yaw = math.atan2(dy_dt, dx_dt)
            local_points.append(PathPoint(x, y, yaw, self.p.turn_speed))

        for x in np.linspace(0.4 + forward, 0.4 + forward + self.p.enter_distance, 16):
            local_points.append(PathPoint(float(x), lateral_shift, 0.0, self.p.enter_speed))

        for lp in local_points:
            ox, oy = transform_point_from_base(lp.x, lp.y, pose)
            oyaw = wrap_to_pi(pose.yaw + lp.yaw)
            points.append(PathPoint(ox, oy, oyaw, lp.v))

        return LocalPath(points=points, valid=True, frame_id=self.p.odom_frame)

    def plan_enter_row(self) -> LocalPath:
        points: List[PathPoint] = []
        for x in np.linspace(0.2, self.p.enter_distance, 20):
            points.append(PathPoint(float(x), 0.0, 0.0, self.p.enter_speed))
        return LocalPath(points=points, valid=True, frame_id="base_link")


# =============================================================================
# Path Follower
# =============================================================================


class PathFollower:
    def __init__(self, params: NavParams):
        self.p = params
        self.last_cmd = Twist()

    def compute_cmd(self, path: LocalPath, pose_odom: Optional[Pose2D], dt: float) -> Twist:
        cmd = Twist()
        if not path.valid or len(path.points) == 0:
            self.last_cmd = self.limit_accel(cmd, dt)
            return self.last_cmd

        target = self.find_lookahead_point(path, pose_odom)
        if target is None:
            self.last_cmd = self.limit_accel(cmd, dt)
            return self.last_cmd

        # Zielpunkt ist nach find_lookahead_point immer im base_link-Frame.
        alpha = math.atan2(target.y, target.x)
        curvature = 2.0 * math.sin(alpha) / max(self.p.lookahead_distance, 1e-3)

        v = clamp(target.v, 0.0, self.p.max_linear_speed)
        if abs(curvature) > 1.0:
            v *= 0.6
        if abs(curvature) > 1.8:
            v *= 0.4

        w = clamp(v * curvature, -self.p.max_angular_speed, self.p.max_angular_speed)

        cmd.linear.x = v
        cmd.angular.z = w
        cmd = self.limit_accel(cmd, dt)
        self.last_cmd = cmd
        return cmd

    def find_lookahead_point(self, path: LocalPath, pose_odom: Optional[Pose2D]) -> Optional[PathPoint]:
        candidates: List[PathPoint] = []

        for p in path.points:
            if path.frame_id == "base_link":
                bx, by = p.x, p.y
                byaw = p.yaw
            else:
                if pose_odom is None:
                    return None
                bx, by = transform_point_to_base(p.x, p.y, pose_odom)
                byaw = wrap_to_pi(p.yaw - pose_odom.yaw)

            if bx < -0.10:
                continue
            d = math.hypot(bx, by)
            candidates.append(PathPoint(bx, by, byaw, p.v))
            if d >= self.p.lookahead_distance:
                return candidates[-1]

        if candidates:
            return candidates[-1]
        return None

    def path_goal_reached(self, path: LocalPath, pose_odom: Optional[Pose2D]) -> bool:
        if not path.valid or len(path.points) == 0:
            return False

        goal = path.points[-1]
        if path.frame_id == "base_link":
            dist = math.hypot(goal.x, goal.y)
            yaw_err = abs(wrap_to_pi(goal.yaw))
        else:
            if pose_odom is None:
                return False
            dist = math.hypot(goal.x - pose_odom.x, goal.y - pose_odom.y)
            yaw_err = abs(wrap_to_pi(goal.yaw - pose_odom.yaw))

        return (
            dist <= self.p.path_goal_tolerance_xy
            and yaw_err <= math.radians(self.p.path_goal_tolerance_yaw_deg)
        )

    def limit_accel(self, desired: Twist, dt: float) -> Twist:
        if dt <= 0.0:
            return desired

        out = Twist()
        dv = desired.linear.x - self.last_cmd.linear.x
        dw = desired.angular.z - self.last_cmd.angular.z

        max_dv = self.p.max_linear_accel * dt
        max_dw = self.p.max_angular_accel * dt

        out.linear.x = self.last_cmd.linear.x + clamp(dv, -max_dv, max_dv)
        out.angular.z = self.last_cmd.angular.z + clamp(dw, -max_dw, max_dw)
        return out

    def reset(self) -> None:
        self.last_cmd = Twist()


# =============================================================================
# Safety
# =============================================================================


class SafetySupervisor:
    def __init__(self, params: NavParams):
        self.p = params

    def filter_cmd(
        self,
        cmd: Twist,
        scan: Optional[LaserScan],
        row: RowModel,
        state: MissionState,
    ) -> Twist:
        safe = Twist()
        safe.linear.x = cmd.linear.x
        safe.angular.z = cmd.angular.z

        min_front = self.min_front_distance(scan)
        if min_front is not None:
            if min_front < self.p.obstacle_stop_distance:
                return Twist()
            if min_front < self.p.obstacle_slow_distance:
                scale = clamp(
                    (min_front - self.p.obstacle_stop_distance)
                    / max(self.p.obstacle_slow_distance - self.p.obstacle_stop_distance, 1e-3),
                    0.0,
                    1.0,
                )
                safe.linear.x *= scale

        # Geringe Reihen-Konfidenz soll nur in der Reihe hart bremsen.
        # Im Vorgewende darf die fehlende Reihe nicht zum Blockieren fuehren.
        if state == MissionState.FOLLOW_ROW:
            if row.confidence < self.p.min_follow_confidence_stop:
                return Twist()
            if row.confidence < self.p.min_follow_confidence_slow:
                safe.linear.x *= 0.4

        return safe

    def min_front_distance(self, scan: Optional[LaserScan]) -> Optional[float]:
        if scan is None:
            return None

        half_angle = math.radians(self.p.front_obstacle_angle_deg)
        angle = scan.angle_min
        values: List[float] = []

        for r in scan.ranges:
            if -half_angle <= angle <= half_angle:
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    values.append(float(r))
            angle += scan.angle_increment

        if not values:
            return None
        return min(values)


# =============================================================================
# Haupt-Node
# =============================================================================


class MaizeNavigator(Node):
    def __init__(self) -> None:
        super().__init__("maize_navigator")

        self.params = self.load_params()

        self.perception = RowPerception(self.params)
        self.tracker = RowTracker(self.params)
        self.mission = MissionStateMachine(self.params)
        self.planner = LocalPlanner(self.params)
        self.controller = PathFollower(self.params)
        self.safety = SafetySupervisor(self.params)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_row_detection = RowDetection()
        self.latest_row_model = RowModel()
        self.latest_path = LocalPath()

        self.last_loop_time = self.get_clock().now()

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.params.scan_topic,
            self.scan_callback,
            10,
        )
        self.cmd_pub = self.create_publisher(Twist, self.params.cmd_vel_topic, 10)

        self.debug_state_pub = self.create_publisher(String, "/debug/state", 10)
        self.debug_conf_pub = self.create_publisher(Float32, "/debug/row_confidence", 10)
        self.debug_end_pub = self.create_publisher(Float32, "/debug/end_probability", 10)
        self.debug_path_pub = self.create_publisher(Path, "/debug/local_path", 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, "/debug/row_markers", 10)

        self.start_srv = self.create_service(Trigger, "/start_navigation", self.handle_start)
        self.stop_srv = self.create_service(Trigger, "/stop_navigation", self.handle_stop)

        timer_period = 1.0 / max(self.params.control_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info("maize_navigator initialized")
        self.get_logger().info(f"scan_topic={self.params.scan_topic}")
        self.get_logger().info(f"pattern={self.params.pattern}")

    def load_params(self) -> NavParams:
        p = NavParams()

        # Parameter deklarieren
        for name, default in p.__dict__.items():
            self.declare_parameter(name, default)

        # Parameter aus ROS laden
        kwargs = {}
        for name in p.__dict__.keys():
            kwargs[name] = self.get_parameter(name).value

        return NavParams(**kwargs)

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def handle_start(self, request, response):
        del request
        self.mission.start()
        self.controller.reset()
        response.success = True
        response.message = "navigation started"
        self.get_logger().info("Navigation started")
        return response

    def handle_stop(self, request, response):
        del request
        self.mission.stop()
        self.controller.reset()
        self.publish_stop()
        response.success = True
        response.message = "navigation stopped"
        self.get_logger().info("Navigation stopped")
        return response

    def control_loop(self) -> None:
        now = self.get_clock().now()
        dt = max((now - self.last_loop_time).nanoseconds * 1e-9, 1e-3)
        self.last_loop_time = now

        pose_odom = self.lookup_robot_pose()

        if self.latest_scan is not None:
            self.latest_row_detection = self.perception.process_scan(self.latest_scan)
            self.latest_row_model = self.tracker.update(self.latest_row_detection)

        row = self.latest_row_model
        state = self.mission.state

        path = self.update_mission_and_plan(state, row, pose_odom)
        self.latest_path = path

        if self.mission.state in (MissionState.IDLE, MissionState.FINISHED):
            self.publish_stop()
            self.publish_debug(row, self.latest_path)
            return

        cmd = self.controller.compute_cmd(path, pose_odom, dt)
        safe_cmd = self.safety.filter_cmd(cmd, self.latest_scan, row, self.mission.state)
        self.cmd_pub.publish(safe_cmd)

        self.publish_debug(row, path)

    def update_mission_and_plan(
        self,
        state: MissionState,
        row: RowModel,
        pose_odom: Optional[Pose2D],
    ) -> LocalPath:
        # IDLE / FINISHED
        if state == MissionState.IDLE:
            return LocalPath(valid=False, reason="idle")
        if state == MissionState.FINISHED:
            return LocalPath(valid=False, reason="finished")

        # FOLLOW_ROW: kontinuierlich neuen lokalen Pfad aus Reihenmodell bilden.
        if state == MissionState.FOLLOW_ROW:
            if row.end_detected:
                if self.mission.current_step() is None:
                    self.mission.transition(MissionState.FINISHED, "pattern complete")
                    return LocalPath(valid=False, reason="finished")
                if pose_odom is None:
                    return LocalPath(valid=False, reason="no odom pose for exit")
                self.mission.active_path = self.planner.plan_exit_path(pose_odom)
                self.mission.transition(MissionState.EXIT_ROW, "row end detected")
                self.get_logger().info("State transition: FOLLOW_ROW -> EXIT_ROW")
                return self.mission.active_path

            return self.planner.plan_follow_row(row)

        # EXIT_ROW: festen Exit-Pfad verfolgen.
        if state == MissionState.EXIT_ROW:
            if self.mission.active_path is None:
                if pose_odom is None:
                    return LocalPath(valid=False, reason="no odom pose")
                self.mission.active_path = self.planner.plan_exit_path(pose_odom)

            if self.controller.path_goal_reached(self.mission.active_path, pose_odom):
                self.mission.transition(MissionState.PLAN_TURN, "exit path reached")
                self.get_logger().info("State transition: EXIT_ROW -> PLAN_TURN")

            return self.mission.active_path

        # PLAN_TURN: nur einmal planen, dann EXECUTE_TURN.
        if state == MissionState.PLAN_TURN:
            step = self.mission.current_step()
            if step is None:
                self.mission.transition(MissionState.FINISHED, "no remaining pattern step")
                return LocalPath(valid=False, reason="finished")
            if pose_odom is None:
                return LocalPath(valid=False, reason="no odom pose for turn")

            self.mission.active_path = self.planner.plan_turn_path(pose_odom, step)
            self.mission.transition(MissionState.EXECUTE_TURN, "turn path planned")
            self.get_logger().info(
                f"State transition: PLAN_TURN -> EXECUTE_TURN, step={step.count}{'L' if step.direction > 0 else 'R'}"
            )
            return self.mission.active_path

        # EXECUTE_TURN: festen Turn-Pfad verfolgen. Reihen-Konfidenz darf hier fehlen.
        if state == MissionState.EXECUTE_TURN:
            if self.mission.active_path is None:
                self.mission.transition(MissionState.PLAN_TURN, "missing turn path")
                return LocalPath(valid=False, reason="missing turn path")

            if self.controller.path_goal_reached(self.mission.active_path, pose_odom):
                self.mission.enter_stable_frames = 0
                self.mission.transition(MissionState.ENTER_ROW, "turn path reached")
                self.get_logger().info("State transition: EXECUTE_TURN -> ENTER_ROW")

            return self.mission.active_path

        # ENTER_ROW: langsam geradeaus bzw. mit aktuellem Reihenmodell in Reihe einfahren.
        if state == MissionState.ENTER_ROW:
            if row.valid and row.confidence >= self.params.min_enter_confidence:
                self.mission.enter_stable_frames += 1
            else:
                self.mission.enter_stable_frames = 0

            if self.mission.enter_stable_frames >= self.params.enter_stable_frames_required:
                self.mission.advance_step()
                self.mission.active_path = None
                self.mission.enter_stable_frames = 0
                row.end_probability = 0.0
                row.end_detected = False
                self.mission.transition(MissionState.FOLLOW_ROW, "target row acquired")
                self.get_logger().info("State transition: ENTER_ROW -> FOLLOW_ROW")
                return self.planner.plan_follow_row(row)

            if row.valid and row.confidence > 0.25:
                path = self.planner.plan_follow_row(row)
                # beim Einfahren langsamer fahren
                for pt in path.points:
                    pt.v = min(pt.v, self.params.enter_speed)
                return path

            return self.planner.plan_enter_row()

        return LocalPath(valid=False, reason="unhandled state")

    def lookup_robot_pose(self) -> Optional[Pose2D]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.params.odom_frame,
                self.params.base_frame,
                Time(),
                timeout=Duration(seconds=0.02),
            )
        except Exception:
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        return Pose2D(float(t.x), float(t.y), quaternion_to_yaw(q))

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def publish_debug(self, row: RowModel, path: LocalPath) -> None:
        state_msg = String()
        state_msg.data = self.mission.state.name
        self.debug_state_pub.publish(state_msg)

        conf_msg = Float32()
        conf_msg.data = float(row.confidence)
        self.debug_conf_pub.publish(conf_msg)

        end_msg = Float32()
        end_msg.data = float(row.end_probability)
        self.debug_end_pub.publish(end_msg)

        self.publish_debug_path(path)
        self.publish_row_markers(row)

    def publish_debug_path(self, path: LocalPath) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = path.frame_id if path.frame_id else self.params.base_frame

        if path.valid:
            for p in path.points:
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position.x = float(p.x)
                pose.pose.position.y = float(p.y)
                pose.pose.position.z = 0.0
                pose.pose.orientation = yaw_to_quaternion(p.yaw)
                msg.poses.append(pose)

        self.debug_path_pub.publish(msg)

    def publish_row_markers(self, row: RowModel) -> None:
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Centerline im base_link-Frame
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = self.params.base_frame
        marker.ns = "row_model"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 1.0

        if row.valid:
            for x in np.linspace(0.2, self.params.row_lookahead_distance, 20):
                pt = Point()
                pt.x = float(x)
                pt.y = float(row.center_a * x + row.center_b)
                pt.z = 0.05
                marker.points.append(pt)
        markers.markers.append(marker)

        # Text mit Confidence
        text = Marker()
        text.header.stamp = now
        text.header.frame_id = self.params.base_frame
        text.ns = "row_model"
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = 0.5
        text.pose.position.y = 0.0
        text.pose.position.z = 0.7
        text.scale.z = 0.18
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = (
            f"state={self.mission.state.name}\n"
            f"conf={row.confidence:.2f}\n"
            f"end={row.end_probability:.2f}"
        )
        markers.markers.append(text)

        self.debug_marker_pub.publish(markers)


# =============================================================================
# main
# =============================================================================


def main(args=None) -> None:
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
