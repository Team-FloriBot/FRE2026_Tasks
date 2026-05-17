import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Twist, Point32
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger

import tf2_ros

import numpy as np
from enum import Enum
from dataclasses import dataclass
import re


# ============================================================
# >>> HILFSFUNKTIONEN
# ============================================================

def normalize_angle(angle: float) -> float:
    """
    Normiert einen Winkel auf den Bereich [-pi, pi].
    """
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_from_quaternion(q) -> float:
    """
    Berechnet Yaw aus Quaternion.
    Keine zusätzliche tf_transformations-Dependency nötig.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def distance_2d(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


# ============================================================
# >>> DATENCONTAINER: PERCEPTION OUTPUT
# ============================================================

@dataclass
class PerceptionData:
    left_dist: float = np.inf
    right_dist: float = np.inf
    center_error: float = 0.0
    row_end_detected: bool = False

    x_mean: float = np.inf
    y_mean: float = np.inf

    min_dist: float = np.inf
    num_points_in_box: int = 0
    filtered_points: list = None

    # Lokale Reihenausrichtung aus Laserpunktwolke.
    # Positiv: Reihenmittelpunkt läuft nach links, wenn x nach vorne zunimmt.
    row_heading_error: float = 0.0
    row_heading_valid: bool = False

    # Nahbereich für Kollisionsvermeidung in der Reihe.
    left_near: float = np.inf
    right_near: float = np.inf
    min_side_clearance: float = np.inf

    # Zusatz für schnelle S-Kurven.
    near_center_error: float = 0.0
    forward_center_error: float = 0.0
    forward_center_valid: bool = False

    # Reihenerkennung beim seitlichen Vorbeifahren im Vorgewende.
    row_candidate_y: float = np.inf
    row_candidate_valid: bool = False
    row_candidate_strength: int = 0
    row_candidate_width: float = np.inf


# ============================================================
# >>> DATENCONTAINER: ROBOTERPOSE AUS SLAM / TF
# ============================================================

@dataclass
class RobotPose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    valid: bool = False


# ============================================================
# >>> DATENCONTAINER: CONTROLLER OUTPUT
# ============================================================

@dataclass
class ControlCommand:
    linear: float
    angular: float


# ============================================================
# >>> STATE MACHINE ZUSTÄNDE
# ============================================================

class State(Enum):
    ROBOT_STOP = 0
    DRIVE_IN_ROW = 1
    EXIT_ROW = 2
    SHIFT_TO_NEXT_ROW = 3
    ALIGN_TO_NEXT_ROW = 4
    ENTER_ROW = 5


# ============================================================
# >>> PATTERN, z.B. "1L-2R"
# ============================================================

class Pattern:
    def __init__(self, pattern_str):
        self.steps = self.parse(pattern_str)
        self.index = 0

    def parse(self, pattern_str):
        tokens = pattern_str.split("-")
        result = []

        for token in tokens:
            match = re.match(r"(\d+)([LR])", token.strip())
            if match:
                count = int(match.group(1))
                direction = match.group(2)
                result.append((count, direction))

        return result

    def current(self):
        if self.index < len(self.steps):
            return self.steps[self.index]
        return None

    def next(self):
        self.index += 1

    def reset(self):
        self.index = 0


# ============================================================
# >>> PERCEPTION
# ============================================================

class Perception:
    def __init__(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes

        # Tiefpassfilter gegen Zittern durch einzelne Laser-Ausreißer.
        self.center_error_filtered = 0.0
        self.forward_center_error_filtered = 0.0
        self.heading_error_filtered = 0.0
        self.filter_alpha = 0.25

    def process(self, scan_msg, current_state, pattern_direction) -> PerceptionData:
        data = PerceptionData()
        points = []

        state_key_map = {
            State.DRIVE_IN_ROW: "drive_in_row",
            State.EXIT_ROW: "turn_and_exit",
            State.SHIFT_TO_NEXT_ROW: "counting_rows",
            State.ALIGN_TO_NEXT_ROW: "turn_and_exit",
            State.ENTER_ROW: "turn_to_row",
        }

        box_key = state_key_map.get(current_state, "drive_in_row")
        box = self.bounding_boxes[box_key]

        if current_state in (State.DRIVE_IN_ROW, State.ENTER_ROW):
            filter_dir = "both"
        else:
            filter_dir = pattern_direction

        min_distance = np.inf

        for i, dist in enumerate(scan_msg.ranges):
            if (
                dist < scan_msg.range_min
                or dist > scan_msg.range_max
                or np.isinf(dist)
                or np.isnan(dist)
            ):
                continue

            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            x = dist * np.cos(angle)
            y = dist * np.sin(angle)

            if dist < min_distance:
                min_distance = dist

            if self.is_in_box(x, y, box, filter_dir):
                points.append(Point32(x=float(x), y=float(y), z=0.0))

        data.min_dist = min_distance
        data.num_points_in_box = len(points)

        near_points = [p for p in points if 0.00 <= p.x <= 0.70]
        forward_points = [p for p in points if 0.70 < p.x <= 1.60]

        (
            data.left_dist,
            data.right_dist,
            data.left_near,
            data.right_near,
            data.near_center_error,
            near_valid,
        ) = self.compute_side_distances(near_points)

        data.min_side_clearance = min(data.left_near, data.right_near)

        if near_valid:
            data.center_error = data.near_center_error
            data.row_end_detected = False
        else:
            (
                data.left_dist,
                data.right_dist,
                data.left_near,
                data.right_near,
                data.center_error,
                all_valid,
            ) = self.compute_side_distances(points)

            data.min_side_clearance = min(data.left_near, data.right_near)
            data.row_end_detected = not all_valid

        (
            _forward_left,
            _forward_right,
            _forward_left_near,
            _forward_right_near,
            data.forward_center_error,
            data.forward_center_valid,
        ) = self.compute_side_distances(forward_points)

        points_x = [p.x for p in points]
        points_y = [p.y for p in points]

        data.x_mean = np.mean(points_x) if len(points_x) > 0 else np.inf
        data.y_mean = np.mean(points_y) if len(points_y) > 0 else np.inf

        if current_state == State.SHIFT_TO_NEXT_ROW:
            (
                data.row_candidate_y,
                data.row_candidate_valid,
                data.row_candidate_strength,
                data.row_candidate_width,
            ) = self.detect_lateral_row_candidate(points)

        left_points = [p for p in points if p.y > 0.0]
        right_points = [p for p in points if p.y <= 0.0]

        slopes = []
        left_slope = self.fit_side_slope(left_points)
        right_slope = self.fit_side_slope(right_points)

        if left_slope is not None:
            slopes.append(left_slope)
        if right_slope is not None:
            slopes.append(right_slope)

        if slopes:
            center_slope = float(np.mean(slopes))
            data.row_heading_error = float(np.arctan(center_slope))
            data.row_heading_valid = True

        # Tiefpassfilter gegen Reglerzittern.
        if np.isfinite(data.center_error):
            self.center_error_filtered = (
                self.filter_alpha * data.center_error
                + (1.0 - self.filter_alpha) * self.center_error_filtered
            )
            data.center_error = self.center_error_filtered

        if data.forward_center_valid and np.isfinite(data.forward_center_error):
            self.forward_center_error_filtered = (
                self.filter_alpha * data.forward_center_error
                + (1.0 - self.filter_alpha) * self.forward_center_error_filtered
            )
            data.forward_center_error = self.forward_center_error_filtered

        if data.row_heading_valid and np.isfinite(data.row_heading_error):
            self.heading_error_filtered = (
                self.filter_alpha * data.row_heading_error
                + (1.0 - self.filter_alpha) * self.heading_error_filtered
            )
            data.row_heading_error = self.heading_error_filtered

        data.filtered_points = points

        return data

    def detect_lateral_row_candidate(
        self,
        points,
        bin_size=0.05,
        min_points=5,
        max_cluster_width=0.45,
    ):
        """
        Detektiert eine Maisreihe beim seitlichen Vorbeifahren.

        Rückgabe:
            candidate_y, valid, strength, width
        """
        if len(points) < min_points:
            return np.inf, False, 0, np.inf

        ys = np.array([p.y for p in points], dtype=float)

        if len(ys) < min_points or np.all(~np.isfinite(ys)):
            return np.inf, False, 0, np.inf

        y_min = float(np.min(ys))
        y_max = float(np.max(ys))

        if y_max - y_min < bin_size:
            width = float(y_max - y_min)
            return float(np.median(ys)), True, int(len(ys)), width

        bins = np.arange(y_min, y_max + 2.0 * bin_size, bin_size)

        if len(bins) < 3:
            return np.inf, False, 0, np.inf

        hist, edges = np.histogram(ys, bins=bins)
        hist_smooth = np.convolve(hist, np.ones(3) / 3.0, mode="same")

        peak_idx = int(np.argmax(hist_smooth))
        peak_value = hist_smooth[peak_idx]

        if peak_value < min_points:
            return np.inf, False, int(round(float(peak_value))), np.inf

        peak_center = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
        cluster_mask = np.abs(ys - peak_center) <= max_cluster_width / 2.0
        cluster_ys = ys[cluster_mask]

        if len(cluster_ys) < min_points:
            return np.inf, False, int(len(cluster_ys)), np.inf

        width = float(np.ptp(cluster_ys)) if len(cluster_ys) > 1 else 0.0

        if width > max_cluster_width:
            return float(np.median(cluster_ys)), False, int(len(cluster_ys)), width

        return float(np.median(cluster_ys)), True, int(len(cluster_ys)), width

    def compute_side_distances(self, points):
        """
        Robuste Seitenabstände für linke und rechte Pflanzenreihe.
        """
        left_y = np.array([abs(p.y) for p in points if p.y > 0.0], dtype=float)
        right_y = np.array([abs(p.y) for p in points if p.y <= 0.0], dtype=float)

        if len(left_y) > 0:
            left_dist = float(np.percentile(left_y, 25.0))
            left_near = float(np.min(left_y))
        else:
            left_dist = np.inf
            left_near = np.inf

        if len(right_y) > 0:
            right_dist = float(np.percentile(right_y, 25.0))
            right_near = float(np.min(right_y))
        else:
            right_dist = np.inf
            right_near = np.inf

        if np.isinf(left_dist) or np.isinf(right_dist):
            center_error = 0.0
            valid = False
        else:
            center_error = (right_dist - left_dist) / 2.0
            valid = True

        return left_dist, right_dist, left_near, right_near, center_error, valid

    def is_in_box(self, x, y, box, direction):
        x_min, x_max = box["x_min"], box["x_max"]
        y_min, y_max = box["y_min"], box["y_max"]

        if not (x_min < x < x_max):
            return False

        if direction == "both":
            return y_min < abs(y) < y_max

        if direction == "L":
            return y_min < y < y_max

        if direction == "R":
            return -y_max < y < -y_min

        return False

    def fit_side_slope(self, points, min_points=4):
        """
        Fit y = m*x + b für eine Pflanzenseite im Roboterkoordinatensystem.
        """
        if len(points) < min_points:
            return None

        xs = np.array([p.x for p in points], dtype=float)
        ys = np.array([p.y for p in points], dtype=float)

        if np.ptp(xs) < 0.20:
            return None

        m, _ = np.polyfit(xs, ys, 1)
        return float(m)


# ============================================================
# >>> STATE MACHINE
# ============================================================

class StateMachine:
    def __init__(self, pattern, node):
        self.state = State.ROBOT_STOP
        self.pattern = pattern
        self.node = node

        self.navigation_triggered = False

        self.exit_target_pose = None
        self.shift_target_pose = None
        self.align_target_pose = None
        self.shift_start_pose = None

        self.row_end_counter = 0
        self.enter_row_seen_counter = 0

        self.shift_row_count = 0
        self.shift_row_visible_counter = 0
        self.shift_row_free_counter = 0
        self.shift_ready_for_next_row = True
        self.last_shift_row_y = np.inf

        self.tf_missing_counter = 0
        self.tf_warning_printed = False

    def get_current_direction(self):
        step = self.pattern.current()
        if step:
            return step[1]
        return "L"

    def reset_mission_state(self):
        self.pattern.reset()

        self.exit_target_pose = None
        self.shift_target_pose = None
        self.align_target_pose = None
        self.shift_start_pose = None

        self.row_end_counter = 0
        self.enter_row_seen_counter = 0
        self.reset_shift_row_detection()

        self.tf_missing_counter = 0
        self.tf_warning_printed = False

    def reset_shift_row_detection(self):
        self.shift_row_count = 0
        self.shift_row_visible_counter = 0
        self.shift_row_free_counter = 0
        self.shift_ready_for_next_row = True
        self.last_shift_row_y = np.inf

    def is_tf_available_or_stop(self, robot_pose: RobotPose, params, state_name: str) -> bool:
        """
        TF-Ausfälle werden toleriert.
        Erst nach tf.missing_max_cycles wird die Mission gestoppt.
        """
        if robot_pose.valid:
            self.tf_missing_counter = 0
            return True

        self.tf_missing_counter += 1

        if self.tf_missing_counter >= params["tf_missing_max_cycles"]:
            self.node.get_logger().warn(
                f"{state_name} abgebrochen: TF fehlt seit "
                f"{self.tf_missing_counter} Zyklen."
            )
            self.state = State.ROBOT_STOP
            return False

        return False

    def compute_headland_targets(self, robot_pose: RobotPose, params):
        """
        Berechnet drei Zielposen im map-Frame.
        """
        step = self.pattern.current()
        if step is None:
            return None, None, None

        row_count, direction = step

        side = 1.0 if direction == "L" else -1.0

        yaw_old = robot_pose.yaw
        yaw_new = normalize_angle(yaw_old + np.pi)

        forward_x = np.cos(yaw_old)
        forward_y = np.sin(yaw_old)

        left_x = -np.sin(yaw_old)
        left_y = np.cos(yaw_old)

        exit_dist = params["turn_headland_exit_dist"]
        row_shift = side * row_count * params["row_width"]
        pre_entry_offset = params["turn_pre_entry_offset"]

        exit_pose = RobotPose(
            x=robot_pose.x + forward_x * exit_dist,
            y=robot_pose.y + forward_y * exit_dist,
            yaw=yaw_old,
            valid=True,
        )

        row_entry_pose_x = exit_pose.x + left_x * row_shift
        row_entry_pose_y = exit_pose.y + left_y * row_shift

        shift_pose = RobotPose(
            x=row_entry_pose_x + forward_x * pre_entry_offset,
            y=row_entry_pose_y + forward_y * pre_entry_offset,
            yaw=yaw_old,
            valid=True,
        )

        align_pose = RobotPose(
            x=shift_pose.x,
            y=shift_pose.y,
            yaw=yaw_new,
            valid=True,
        )

        return exit_pose, shift_pose, align_pose

    def update(self, perception: PerceptionData, params, robot_pose: RobotPose):
        old_state = self.state

        # ====================================================
        # >>> STATE: ROBOT_STOP
        # ====================================================

        if self.state == State.ROBOT_STOP:
            if self.navigation_triggered:
                self.reset_mission_state()
                self.navigation_triggered = False
                self.state = State.DRIVE_IN_ROW

        # ====================================================
        # >>> STATE: DRIVE_IN_ROW
        # ====================================================

        elif self.state == State.DRIVE_IN_ROW:
            if perception.row_end_detected:
                self.row_end_counter += 1
            else:
                self.row_end_counter = 0
                self.tf_warning_printed = False

            if self.row_end_counter >= params["row_end_confirm_cycles"]:
                if robot_pose.valid:
                    (
                        self.exit_target_pose,
                        self.shift_target_pose,
                        self.align_target_pose,
                    ) = self.compute_headland_targets(robot_pose, params)

                    if (
                        self.exit_target_pose is not None
                        and self.shift_target_pose is not None
                        and self.align_target_pose is not None
                    ):
                        self.enter_row_seen_counter = 0
                        self.reset_shift_row_detection()
                        self.shift_start_pose = None
                        self.tf_missing_counter = 0
                        self.state = State.EXIT_ROW
                else:
                    if not self.tf_warning_printed:
                        self.node.get_logger().warn(
                            "Reihenende bestätigt, aber keine gültige SLAM-Pose verfügbar. "
                            "Bleibe in DRIVE_IN_ROW."
                        )
                        self.tf_warning_printed = True

        # ====================================================
        # >>> STATE: EXIT_ROW
        # ====================================================

        elif self.state == State.EXIT_ROW:
            if self.exit_target_pose is None:
                self.node.get_logger().warn("EXIT_ROW abgebrochen: keine exit_target_pose.")
                self.state = State.ROBOT_STOP
            elif self.is_tf_available_or_stop(robot_pose, params, "EXIT_ROW"):
                dist_error = distance_2d(robot_pose, self.exit_target_pose)

                if dist_error < params["turn_exit_pos_tolerance"]:
                    self.shift_start_pose = RobotPose(
                        x=robot_pose.x,
                        y=robot_pose.y,
                        yaw=robot_pose.yaw,
                        valid=True,
                    )
                    self.reset_shift_row_detection()
                    self.state = State.SHIFT_TO_NEXT_ROW

        # ====================================================
        # >>> STATE: SHIFT_TO_NEXT_ROW
        # ====================================================

        elif self.state == State.SHIFT_TO_NEXT_ROW:
            if self.shift_target_pose is None:
                self.node.get_logger().warn("SHIFT_TO_NEXT_ROW abgebrochen: keine shift_target_pose.")
                self.state = State.ROBOT_STOP

            elif self.is_tf_available_or_stop(robot_pose, params, "SHIFT_TO_NEXT_ROW"):
                step = self.pattern.current()
                target_row_count = step[0] if step is not None else 1

                row_candidate_ok = (
                    perception.row_candidate_valid
                    and perception.row_candidate_strength >= params["row_pass_min_strength"]
                    and perception.row_candidate_width <= params["row_pass_max_width"]
                )

                if row_candidate_ok:
                    self.shift_row_visible_counter += 1
                    self.shift_row_free_counter = 0
                else:
                    self.shift_row_visible_counter = 0
                    self.shift_row_free_counter += 1

                    if self.shift_row_free_counter >= params["row_pass_free_cycles"]:
                        self.shift_ready_for_next_row = True

                if (
                    row_candidate_ok
                    and self.shift_ready_for_next_row
                    and self.shift_row_visible_counter >= params["row_pass_confirm_cycles"]
                ):
                    self.shift_row_count += 1
                    self.last_shift_row_y = perception.row_candidate_y
                    self.shift_ready_for_next_row = False

                    self.node.get_logger().info(
                        f"Reihe beim Vorbeifahren bestätigt: "
                        f"{self.shift_row_count}/{target_row_count}, "
                        f"y={perception.row_candidate_y:.2f} m, "
                        f"points={perception.row_candidate_strength}, "
                        f"width={perception.row_candidate_width:.2f} m"
                    )

                dist_error = distance_2d(robot_pose, self.shift_target_pose)

                max_shift_distance = (
                    target_row_count * params["row_width"]
                    + params["row_pass_max_extra_shift"]
                )

                if self.shift_start_pose is not None:
                    shifted_distance = distance_2d(robot_pose, self.shift_start_pose)

                    if shifted_distance > max_shift_distance:
                        self.node.get_logger().warn(
                            f"SHIFT_TO_NEXT_ROW Sicherheitslimit erreicht: "
                            f"{shifted_distance:.2f} m > {max_shift_distance:.2f} m. "
                            f"Wechsle zu ALIGN_TO_NEXT_ROW."
                        )
                        self.state = State.ALIGN_TO_NEXT_ROW

                enough_rows_seen = self.shift_row_count >= target_row_count
                slam_target_reached = dist_error < params["turn_shift_pos_tolerance"]

                # SLAM-Fallback nur erlauben, wenn mindestens eine Reihe gesehen wurde.
                # Dadurch fährt der Roboter nicht blind sehr weit aus dem Feld heraus.
                guarded_slam_fallback = (
                    slam_target_reached
                    and params["row_pass_fallback_to_slam"]
                    and self.shift_row_count > 0
                )

                if enough_rows_seen or guarded_slam_fallback:
                    self.state = State.ALIGN_TO_NEXT_ROW

        # ====================================================
        # >>> STATE: ALIGN_TO_NEXT_ROW
        # ====================================================

        elif self.state == State.ALIGN_TO_NEXT_ROW:
            if self.align_target_pose is None:
                self.node.get_logger().warn("ALIGN_TO_NEXT_ROW abgebrochen: keine align_target_pose.")
                self.state = State.ROBOT_STOP
            elif self.is_tf_available_or_stop(robot_pose, params, "ALIGN_TO_NEXT_ROW"):
                yaw_error = abs(normalize_angle(self.align_target_pose.yaw - robot_pose.yaw))

                if yaw_error < params["turn_yaw_tolerance"]:
                    self.enter_row_seen_counter = 0
                    self.state = State.ENTER_ROW

        # ====================================================
        # >>> STATE: ENTER_ROW
        # ====================================================

        elif self.state == State.ENTER_ROW:
            row_visible = (
                not perception.row_end_detected
                and not np.isinf(perception.left_dist)
                and not np.isinf(perception.right_dist)
                and abs(perception.center_error) < params["enter_row_max_center_error"]
            )

            if row_visible:
                self.enter_row_seen_counter += 1
            else:
                self.enter_row_seen_counter = 0

            if self.enter_row_seen_counter >= params["enter_row_confirm_cycles"]:
                self.pattern.next()

                self.row_end_counter = 0
                self.enter_row_seen_counter = 0
                self.reset_shift_row_detection()
                self.shift_start_pose = None
                self.tf_missing_counter = 0

                self.exit_target_pose = None
                self.shift_target_pose = None
                self.align_target_pose = None

                if self.pattern.current() is None:
                    self.state = State.ROBOT_STOP
                    self.node.get_logger().info("PATTERN COMPLETED. Stopping Robot.")
                else:
                    self.state = State.DRIVE_IN_ROW

        if self.state != old_state:
            self.node.get_logger().info(f"State transition: {old_state.name} -> {self.state.name}")

        return self.state


# ============================================================
# >>> CONTROLLER
# ============================================================

class Controller:
    def compute(self, state, perception, direction, params, node, robot_pose, state_machine):
        cmd = ControlCommand(linear=0.0, angular=0.0)

        if state == State.ROBOT_STOP:
            return cmd

        # ====================================================
        # >>> STATE: DRIVE_IN_ROW
        # ====================================================

        if state == State.DRIVE_IN_ROW:
            if not perception.row_end_detected:
                center_term = -params["row_center_kp"] * perception.center_error

                if perception.row_heading_valid:
                    heading_term = params["row_heading_kp"] * perception.row_heading_error
                else:
                    heading_term = 0.0

                if perception.forward_center_valid:
                    forward_term = -params["row_forward_kp"] * perception.forward_center_error
                else:
                    forward_term = 0.0

                cmd.angular = center_term + heading_term + forward_term

                if np.isfinite(perception.left_near) and np.isfinite(perception.right_near):
                    clearance_error = perception.right_near - perception.left_near

                    if perception.min_side_clearance < params["row_slow_clearance"]:
                        clearance_term = -params["row_clearance_kp"] * clearance_error
                        cmd.angular += clearance_term

                if np.isfinite(perception.right_near):
                    if perception.right_near < params["row_guard_clearance"]:
                        cmd.angular = max(cmd.angular, params["row_guard_min_angular"])

                if np.isfinite(perception.left_near):
                    if perception.left_near < params["row_guard_clearance"]:
                        cmd.angular = min(cmd.angular, -params["row_guard_min_angular"])

                cmd.angular = float(
                    np.clip(
                        cmd.angular,
                        -params["row_max_angular"],
                        params["row_max_angular"],
                    )
                )

                center_ratio = abs(perception.center_error) / params["max_dist_in_row"]
                angular_ratio = abs(cmd.angular) / max(params["row_max_angular"], 1e-6)
                curve_ratio = max(center_ratio, angular_ratio)

                speed_factor = 1.0 - params["row_curve_slowdown_gain"] * curve_ratio
                speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

                if np.isfinite(perception.min_side_clearance):
                    clearance_speed_factor = (
                        perception.min_side_clearance - params["row_emergency_clearance"]
                    ) / max(
                        params["row_slow_clearance"] - params["row_emergency_clearance"],
                        1e-6,
                    )

                    clearance_speed_factor = float(np.clip(clearance_speed_factor, 0.0, 1.0))
                    speed_factor *= clearance_speed_factor

                cmd.linear = params["vel_linear_drive"] * speed_factor

                if perception.min_side_clearance <= params["row_emergency_clearance"]:
                    cmd.linear = 0.0
                else:
                    cmd.linear = float(
                        np.clip(
                            cmd.linear,
                            params["row_min_linear"],
                            params["vel_linear_drive"],
                        )
                    )

        # ====================================================
        # >>> STATE: EXIT_ROW
        # ====================================================

        elif state == State.EXIT_ROW:
            cmd = self.compute_forward_with_yaw_hold(
                robot_pose=robot_pose,
                target_pose=state_machine.exit_target_pose,
                params=params,
                linear_speed=params["vel_linear_drive"],
                pos_tolerance=params["turn_exit_pos_tolerance"],
            )

        # ====================================================
        # >>> STATE: SHIFT_TO_NEXT_ROW
        # ====================================================

        elif state == State.SHIFT_TO_NEXT_ROW:
            cmd = self.compute_pose_position_control(
                robot_pose=robot_pose,
                target_pose=state_machine.shift_target_pose,
                params=params,
                max_linear=params["vel_linear_turn"],
                pos_tolerance=params["turn_shift_pos_tolerance"],
            )

        # ====================================================
        # >>> STATE: ALIGN_TO_NEXT_ROW
        # ====================================================

        elif state == State.ALIGN_TO_NEXT_ROW:
            cmd = self.compute_yaw_control(
                robot_pose=robot_pose,
                target_pose=state_machine.align_target_pose,
                params=params,
            )

        # ====================================================
        # >>> STATE: ENTER_ROW
        # ====================================================

        elif state == State.ENTER_ROW:
            cmd.linear = params["vel_linear_enter_row"]

            if not perception.row_end_detected:
                cmd.angular = -perception.center_error * 4.0 * params["vel_linear_enter_row"]
            else:
                cmd.angular = 0.0

            cmd.angular = float(
                np.clip(
                    cmd.angular,
                    -params["turn_max_angular"],
                    params["turn_max_angular"],
                )
            )

        return cmd

    def compute_forward_with_yaw_hold(
        self,
        robot_pose: RobotPose,
        target_pose: RobotPose,
        params,
        linear_speed: float,
        pos_tolerance: float,
    ) -> ControlCommand:
        cmd = ControlCommand(linear=0.0, angular=0.0)

        if not robot_pose.valid or target_pose is None or not target_pose.valid:
            return cmd

        dist_error = distance_2d(robot_pose, target_pose)

        if dist_error < pos_tolerance:
            return cmd

        yaw_error = normalize_angle(target_pose.yaw - robot_pose.yaw)

        linear_scale = np.clip(dist_error / params["turn_slowdown_distance"], 0.0, 1.0)

        cmd.linear = linear_speed * linear_scale
        cmd.angular = params["turn_k_yaw"] * yaw_error

        cmd.angular = float(
            np.clip(
                cmd.angular,
                -params["turn_max_angular"],
                params["turn_max_angular"],
            )
        )

        return cmd

    def compute_pose_position_control(
        self,
        robot_pose: RobotPose,
        target_pose: RobotPose,
        params,
        max_linear: float,
        pos_tolerance: float,
    ) -> ControlCommand:
        cmd = ControlCommand(linear=0.0, angular=0.0)

        if not robot_pose.valid or target_pose is None or not target_pose.valid:
            return cmd

        dx = target_pose.x - robot_pose.x
        dy = target_pose.y - robot_pose.y

        dist_error = float(np.hypot(dx, dy))

        if dist_error < pos_tolerance:
            return cmd

        target_heading = np.arctan2(dy, dx)
        heading_error = normalize_angle(target_heading - robot_pose.yaw)

        linear_scale = np.clip(dist_error / params["turn_slowdown_distance"], 0.0, 1.0)

        # Weicheres Eindrehen.
        heading_factor = np.clip(np.cos(heading_error), 0.25, 1.0)

        cmd.linear = max_linear * linear_scale * heading_factor
        cmd.angular = params["turn_k_heading"] * heading_error

        # Nahe am Ziel nicht mehr stark drehen.
        if dist_error < 0.25:
            cmd.angular *= dist_error / 0.25

        cmd.angular = float(
            np.clip(
                cmd.angular,
                -params["turn_max_angular"],
                params["turn_max_angular"],
            )
        )

        return cmd

    def compute_yaw_control(
        self,
        robot_pose: RobotPose,
        target_pose: RobotPose,
        params,
    ) -> ControlCommand:
        cmd = ControlCommand(linear=0.0, angular=0.0)

        if not robot_pose.valid or target_pose is None or not target_pose.valid:
            return cmd

        yaw_error = normalize_angle(target_pose.yaw - robot_pose.yaw)

        if abs(yaw_error) < params["turn_yaw_tolerance"]:
            return cmd

        cmd.linear = 0.0
        cmd.angular = params["turn_k_yaw"] * yaw_error

        cmd.angular = float(
            np.clip(
                cmd.angular,
                -params["turn_max_angular"],
                params["turn_max_angular"],
            )
        )

        return cmd


# ============================================================
# >>> VELOCITY RAMPER
# ============================================================

class VelocityRamper:
    def __init__(self, max_accel_linear=0.22, max_accel_angular=1.2, dt=0.1):
        self.max_accel_linear = max_accel_linear
        self.max_accel_angular = max_accel_angular
        self.dt = dt

        self.current_linear = 0.0
        self.current_angular = 0.0

    def update(self, setpoint_linear: float, setpoint_angular: float) -> tuple:
        max_delta_linear = self.max_accel_linear * self.dt
        delta_linear = setpoint_linear - self.current_linear
        delta_linear = np.clip(delta_linear, -max_delta_linear, max_delta_linear)
        self.current_linear += delta_linear

        max_delta_angular = self.max_accel_angular * self.dt
        delta_angular = setpoint_angular - self.current_angular
        delta_angular = np.clip(delta_angular, -max_delta_angular, max_delta_angular)
        self.current_angular += delta_angular

        return self.current_linear, self.current_angular

    def reset(self):
        self.current_linear = 0.0
        self.current_angular = 0.0


# ============================================================
# >>> HAUPTNODE
# ============================================================

class FieldRobotNavigator(Node):
    def __init__(self):
        super().__init__("maize_navigator")

        # ====================================================
        # >>> PARAMETER DEKLARATION
        # ====================================================

        self.declare_parameter("pattern", "1L-1R-2L-3R")

        self.declare_parameter("max_dist_in_row", 0.375)
        self.declare_parameter("row_width", 0.75)
        self.declare_parameter("drive_out_dist", 1.0)

        self.declare_parameter("vel_linear_drive", 0.22)
        self.declare_parameter("vel_linear_count", 0.35)
        self.declare_parameter("vel_linear_turn", 0.14)
        self.declare_parameter("vel_linear_enter_row", 0.12)

        # Reihenregler
        self.declare_parameter("row_center_kp", 1.8)
        self.declare_parameter("row_heading_kp", 0.45)
        self.declare_parameter("row_forward_kp", 0.55)
        self.declare_parameter("row_max_angular", 0.65)
        self.declare_parameter("row_min_linear", 0.0)
        self.declare_parameter("row_curve_slowdown_gain", 0.65)

        # Nahbereichs-Kollisionsvermeidung
        self.declare_parameter("row_clearance_kp", 1.4)
        self.declare_parameter("row_slow_clearance", 0.28)
        self.declare_parameter("row_emergency_clearance", 0.10)
        self.declare_parameter("row_guard_clearance", 0.16)
        self.declare_parameter("row_guard_min_angular", 0.25)

        # Velocity Ramper
        self.declare_parameter("accel_max_linear", 0.22)
        self.declare_parameter("accel_max_angular", 1.2)

        # SLAM / TF FRAMES
        self.declare_parameter("frames.map", "map")
        self.declare_parameter("frames.robot_base", "base_link")

        # Reihenende
        self.declare_parameter("row_end.confirm_cycles", 5)

        # Reihenerkennung beim seitlichen Vorbeifahren
        self.declare_parameter("row_pass.confirm_cycles", 3)
        self.declare_parameter("row_pass.free_cycles", 2)
        self.declare_parameter("row_pass.min_strength", 5)
        self.declare_parameter("row_pass.max_width", 0.45)
        self.declare_parameter("row_pass.fallback_to_slam", True)
        self.declare_parameter("row_pass.max_extra_shift", 0.25)

        # TF
        self.declare_parameter("tf.missing_max_cycles", 10)

        # SLAM-basierte Wendeparameter
        self.declare_parameter("turn.headland_exit_dist", 0.65)
        self.declare_parameter("turn.pre_entry_offset", 0.15)

        self.declare_parameter("turn.exit_pos_tolerance", 0.08)
        self.declare_parameter("turn.shift_pos_tolerance", 0.08)
        self.declare_parameter("turn.pos_tolerance", 0.10)
        self.declare_parameter("turn.yaw_tolerance", 0.10)

        self.declare_parameter("turn.slowdown_distance", 0.65)
        self.declare_parameter("turn.k_heading", 0.9)
        self.declare_parameter("turn.k_yaw", 1.0)
        self.declare_parameter("turn.max_angular", 0.45)

        # ENTER_ROW
        self.declare_parameter("enter_row.confirm_cycles", 5)
        self.declare_parameter("enter_row.max_center_error", 0.20)

        # PERCEPTION PARAMETER
        states = ["drive_in_row", "turn_and_exit", "counting_rows", "turn_to_row"]
        for state_name in states:
            if state_name == "drive_in_row":
                default_x_min = 0.0
                default_x_max = 1.6
                default_y_min = 0.03
                default_y_max = 1.0
            elif state_name == "counting_rows":
                default_x_min = -0.35
                default_x_max = 0.35
                default_y_min = 0.12
                default_y_max = 1.05
            else:
                default_x_min = 0.0
                default_x_max = 2.0
                default_y_min = 0.1
                default_y_max = 1.0

            self.declare_parameter(f"perception.{state_name}.x_min", default_x_min)
            self.declare_parameter(f"perception.{state_name}.x_max", default_x_max)
            self.declare_parameter(f"perception.{state_name}.y_min", default_y_min)
            self.declare_parameter(f"perception.{state_name}.y_max", default_y_max)

        # ROS TOPICS
        self.declare_parameter("topics.laserscan", "/sensors/merged_scan")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")
        self.declare_parameter("topics.field_points", "/field_points")

        # ====================================================
        # >>> PARAMETER EINLESEN
        # ====================================================

        self.params = self.get_all_params()

        # ====================================================
        # >>> TF2 FÜR SLAM-POSE
        # ====================================================

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ====================================================
        # >>> MODULE INITIALISIEREN
        # ====================================================

        self.perception = Perception(self.params["bounding_boxes"])
        self.pattern = Pattern(self.params["pattern"])
        self.state_machine = StateMachine(self.pattern, self)
        self.controller = Controller()

        self.velocity_ramper = VelocityRamper(
            max_accel_linear=self.params["accel_max_linear"],
            max_accel_angular=self.params["accel_max_angular"],
            dt=0.1,
        )

        self.latest_scan = None

        # ====================================================
        # >>> ROS KOMMUNIKATION
        # ====================================================

        self.create_subscription(
            LaserScan,
            self.params["topic_laserscan"],
            self.scan_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.params["topic_cmd_vel"],
            10,
        )

        self.points_pub = self.create_publisher(
            PointCloud2,
            self.params["topic_field_points"],
            10,
        )

        self.start_srv = self.create_service(
            Trigger,
            "start_navigation",
            self.start_nav_callback,
        )

        self.timer = self.create_timer(0.1, self.loop)

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info("FieldRobotNavigator gestartet")
        self.get_logger().info(
            f"SLAM/TF Frames: {self.params['frame_map']} -> {self.params['frame_robot_base']}"
        )

    def parameter_callback(self, params):
        for param in params:
            keys = param.name.split(".")

            if len(keys) == 1:
                self.params[keys[0]] = param.value

                if param.name == "accel_max_linear":
                    self.velocity_ramper.max_accel_linear = param.value

                elif param.name == "accel_max_angular":
                    self.velocity_ramper.max_accel_angular = param.value

            elif len(keys) == 2 and keys[0] == "frames":
                if keys[1] == "map":
                    self.params["frame_map"] = param.value
                elif keys[1] == "robot_base":
                    self.params["frame_robot_base"] = param.value

            elif len(keys) == 2 and keys[0] == "row_end":
                if keys[1] == "confirm_cycles":
                    self.params["row_end_confirm_cycles"] = int(param.value)

            elif len(keys) == 2 and keys[0] == "row_pass":
                mapping = {
                    "confirm_cycles": "row_pass_confirm_cycles",
                    "free_cycles": "row_pass_free_cycles",
                    "min_strength": "row_pass_min_strength",
                    "max_width": "row_pass_max_width",
                    "fallback_to_slam": "row_pass_fallback_to_slam",
                    "max_extra_shift": "row_pass_max_extra_shift",
                }

                if keys[1] in mapping:
                    if keys[1] in ("confirm_cycles", "free_cycles", "min_strength"):
                        self.params[mapping[keys[1]]] = int(param.value)
                    else:
                        self.params[mapping[keys[1]]] = param.value

            elif len(keys) == 2 and keys[0] == "tf":
                if keys[1] == "missing_max_cycles":
                    self.params["tf_missing_max_cycles"] = int(param.value)

            elif len(keys) == 2 and keys[0] == "turn":
                mapping = {
                    "headland_exit_dist": "turn_headland_exit_dist",
                    "pre_entry_offset": "turn_pre_entry_offset",
                    "exit_pos_tolerance": "turn_exit_pos_tolerance",
                    "shift_pos_tolerance": "turn_shift_pos_tolerance",
                    "pos_tolerance": "turn_pos_tolerance",
                    "yaw_tolerance": "turn_yaw_tolerance",
                    "slowdown_distance": "turn_slowdown_distance",
                    "k_heading": "turn_k_heading",
                    "k_yaw": "turn_k_yaw",
                    "max_angular": "turn_max_angular",
                }

                if keys[1] in mapping:
                    self.params[mapping[keys[1]]] = param.value

            elif len(keys) == 2 and keys[0] == "enter_row":
                if keys[1] == "confirm_cycles":
                    self.params["enter_row_confirm_cycles"] = int(param.value)
                elif keys[1] == "max_center_error":
                    self.params["enter_row_max_center_error"] = param.value

            elif len(keys) == 3 and keys[0] == "perception":
                self.params["bounding_boxes"][keys[1]][keys[2]] = param.value

        return SetParametersResult(successful=True)

    def get_all_params(self):
        p = {}

        p["pattern"] = self.get_parameter("pattern").value

        p["max_dist_in_row"] = self.get_parameter("max_dist_in_row").value
        p["row_width"] = self.get_parameter("row_width").value
        p["drive_out_dist"] = self.get_parameter("drive_out_dist").value

        p["vel_linear_drive"] = self.get_parameter("vel_linear_drive").value
        p["vel_linear_count"] = self.get_parameter("vel_linear_count").value
        p["vel_linear_turn"] = self.get_parameter("vel_linear_turn").value
        p["vel_linear_enter_row"] = self.get_parameter("vel_linear_enter_row").value

        p["row_center_kp"] = self.get_parameter("row_center_kp").value
        p["row_heading_kp"] = self.get_parameter("row_heading_kp").value
        p["row_forward_kp"] = self.get_parameter("row_forward_kp").value
        p["row_max_angular"] = self.get_parameter("row_max_angular").value
        p["row_min_linear"] = self.get_parameter("row_min_linear").value
        p["row_curve_slowdown_gain"] = self.get_parameter("row_curve_slowdown_gain").value

        p["row_clearance_kp"] = self.get_parameter("row_clearance_kp").value
        p["row_slow_clearance"] = self.get_parameter("row_slow_clearance").value
        p["row_emergency_clearance"] = self.get_parameter("row_emergency_clearance").value
        p["row_guard_clearance"] = self.get_parameter("row_guard_clearance").value
        p["row_guard_min_angular"] = self.get_parameter("row_guard_min_angular").value

        p["accel_max_linear"] = self.get_parameter("accel_max_linear").value
        p["accel_max_angular"] = self.get_parameter("accel_max_angular").value

        p["frame_map"] = self.get_parameter("frames.map").value
        p["frame_robot_base"] = self.get_parameter("frames.robot_base").value

        p["row_end_confirm_cycles"] = int(self.get_parameter("row_end.confirm_cycles").value)

        p["row_pass_confirm_cycles"] = int(self.get_parameter("row_pass.confirm_cycles").value)
        p["row_pass_free_cycles"] = int(self.get_parameter("row_pass.free_cycles").value)
        p["row_pass_min_strength"] = int(self.get_parameter("row_pass.min_strength").value)
        p["row_pass_max_width"] = self.get_parameter("row_pass.max_width").value
        p["row_pass_fallback_to_slam"] = self.get_parameter("row_pass.fallback_to_slam").value
        p["row_pass_max_extra_shift"] = self.get_parameter("row_pass.max_extra_shift").value

        p["tf_missing_max_cycles"] = int(self.get_parameter("tf.missing_max_cycles").value)

        p["turn_headland_exit_dist"] = self.get_parameter("turn.headland_exit_dist").value
        p["turn_pre_entry_offset"] = self.get_parameter("turn.pre_entry_offset").value

        p["turn_exit_pos_tolerance"] = self.get_parameter("turn.exit_pos_tolerance").value
        p["turn_shift_pos_tolerance"] = self.get_parameter("turn.shift_pos_tolerance").value
        p["turn_pos_tolerance"] = self.get_parameter("turn.pos_tolerance").value
        p["turn_yaw_tolerance"] = self.get_parameter("turn.yaw_tolerance").value

        p["turn_slowdown_distance"] = self.get_parameter("turn.slowdown_distance").value
        p["turn_k_heading"] = self.get_parameter("turn.k_heading").value
        p["turn_k_yaw"] = self.get_parameter("turn.k_yaw").value
        p["turn_max_angular"] = self.get_parameter("turn.max_angular").value

        p["enter_row_confirm_cycles"] = int(self.get_parameter("enter_row.confirm_cycles").value)
        p["enter_row_max_center_error"] = self.get_parameter("enter_row.max_center_error").value

        p["bounding_boxes"] = {}
        states = ["drive_in_row", "turn_and_exit", "counting_rows", "turn_to_row"]

        for state_name in states:
            p["bounding_boxes"][state_name] = {
                "x_min": self.get_parameter(f"perception.{state_name}.x_min").value,
                "x_max": self.get_parameter(f"perception.{state_name}.x_max").value,
                "y_min": self.get_parameter(f"perception.{state_name}.y_min").value,
                "y_max": self.get_parameter(f"perception.{state_name}.y_max").value,
            }

        p["topic_laserscan"] = self.get_parameter("topics.laserscan").value
        p["topic_cmd_vel"] = self.get_parameter("topics.cmd_vel").value
        p["topic_field_points"] = self.get_parameter("topics.field_points").value

        return p

    def get_robot_pose_map(self) -> RobotPose:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.params["frame_map"],
                self.params["frame_robot_base"],
                rclpy.time.Time(),
            )

            t = tf.transform.translation
            q = tf.transform.rotation

            return RobotPose(
                x=float(t.x),
                y=float(t.y),
                yaw=float(yaw_from_quaternion(q)),
                valid=True,
            )

        except Exception as e:
            if self.state_machine.state != State.ROBOT_STOP:
                self.get_logger().warn(
                    f"Keine TF {self.params['frame_map']} -> "
                    f"{self.params['frame_robot_base']} verfügbar: {e}"
                )

            return RobotPose(valid=False)

    def scan_callback(self, msg):
        self.latest_scan = msg

    def start_nav_callback(self, request, response):
        if self.state_machine.state == State.ROBOT_STOP:
            self.state_machine.navigation_triggered = True
            response.success = True
            response.message = "Navigation gestartet!"
            self.get_logger().info("Service 'start_navigation' empfangen. Fahre los...")
        else:
            response.success = False
            response.message = f"Roboter ist bereits im Zustand {self.state_machine.state.name}"

        return response

    def publish_points(self, points, header):
        if not points:
            return

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        point_data = [(p.x, p.y, p.z) for p in points]

        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.latest_scan.header.frame_id

        cloud = point_cloud2.create_cloud(header, fields, point_data)
        self.points_pub.publish(cloud)

    def loop(self):
        if self.latest_scan is None:
            return

        direction = self.state_machine.get_current_direction()

        perception = self.perception.process(
            self.latest_scan,
            self.state_machine.state,
            direction,
        )

        self.publish_points(perception.filtered_points, self.latest_scan.header)

        robot_pose = self.get_robot_pose_map()

        state = self.state_machine.update(
            perception=perception,
            params=self.params,
            robot_pose=robot_pose,
        )

        cmd_setpoint = self.controller.compute(
            state=state,
            perception=perception,
            direction=direction,
            params=self.params,
            node=self,
            robot_pose=robot_pose,
            state_machine=self.state_machine,
        )

        if state == State.ROBOT_STOP:
            self.velocity_ramper.reset()
            cmd_ramped_linear = 0.0
            cmd_ramped_angular = 0.0
        else:
            cmd_ramped_linear, cmd_ramped_angular = self.velocity_ramper.update(
                cmd_setpoint.linear,
                cmd_setpoint.angular,
            )

        twist = Twist()
        twist.linear.x = float(cmd_ramped_linear)
        twist.angular.z = float(cmd_ramped_angular)

        self.cmd_pub.publish(twist)


# ============================================================
# >>> PROGRAMMSTART
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = FieldRobotNavigator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
