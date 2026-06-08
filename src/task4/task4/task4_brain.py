#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from threading import Event
from typing import Dict, List, Sequence, Tuple

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from fre2026_detection_client import DetectorClient, DetectorInitConfig
from fre2026_detection_interfaces.msg import TrackedObjectArray
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

from task4.coverage_planner import (
    CoveragePlannerConfig,
    Point2D,
    Waypoint,
    coords_to_points,
    plan_coverage_path,
    same_point,
    validate_polygon_coords,
)
from task4.shooting_planner import (
    ShootingPlannerConfig,
    ShootingPose,
    ShotTarget,
    distance_2d,
    plan_shooting_poses,
)


class MissionState:
    IDLE = "idle"
    PREPARE_COVERAGE = "prepare_coverage"
    COVERAGE_READY = "coverage_ready"
    DRIVE_COVERAGE = "drive_coverage"
    PLAN_SHOOTING = "plan_shooting"
    DRIVE_TO_SHOOT_POSE = "drive_to_shoot_pose"
    AIM_AND_FIRE = "aim_and_fire"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    ABORTED = "aborted"


class Task4Brain(Node):
    def __init__(self):
        super().__init__("task4_brain")

        self.declare_parameter("polygon_coords", [0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0])
        self.declare_parameter("input_frame", "base_link")
        self.declare_parameter("target_frame", "map")

        self.declare_parameter("operating_width", 1.0)
        self.declare_parameter("robot_width", 0.42)
        self.declare_parameter("turn_radius", 0.37)
        self.declare_parameter("headland_width", 0.5)
        self.declare_parameter("return_start_tolerance_m", 0.15)

        self.declare_parameter("shooting_range_m", 2.0)
        self.declare_parameter("shoot_angle_min_deg", -60.0)
        self.declare_parameter("shoot_angle_max_deg", 60.0)
        self.declare_parameter("candidate_grid_spacing_m", 0.5)
        self.declare_parameter("yaw_sample_step_deg", 10.0)
        self.declare_parameter("path_candidate_stride_m", 0.5)
        self.declare_parameter("object_ring_distance_ratio", 0.75)
        self.declare_parameter("object_ring_angle_step_deg", 20.0)
        self.declare_parameter("min_navigation_segment_m", 0.05)

        self.declare_parameter("tracked_objects_topic", "/tracker/tracked_objects")
        self.declare_parameter("tracker_active_topic", "/tracker/active")
        self.declare_parameter("tracker_reset_service", "/tracker/reset")
        self.declare_parameter("detector_enabled", True)
        self.declare_parameter("detector_required", False)
        self.declare_parameter("detector_namespace", "/detector")
        self.declare_parameter("detector_model_path", "")
        self.declare_parameter("detector_classes", [])
        self.declare_parameter("detector_confidence", 0.5)
        self.declare_parameter("detector_model_type", "yolo")
        self.declare_parameter("detector_use_realsense_ros_wrapper", False)
        self.declare_parameter("detector_use_decimation", False)
        self.declare_parameter("detector_use_spatial", False)
        self.declare_parameter("detector_use_temporal", True)
        self.declare_parameter("detector_use_hole_filling", True)
        self.declare_parameter("detector_use_mask_filter", True)
        self.declare_parameter("detector_color_resolution_width", 640)
        self.declare_parameter("detector_color_resolution_height", 480)
        self.declare_parameter("detector_fps", 30)
        self.declare_parameter("detector_rcnn_class_names", [])
        self.declare_parameter("detector_service_timeout_sec", 10.0)
        self.declare_parameter("detector_release_after_coverage", False)
        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("polygon_marker_topic", "/task4/coverage_polygon_marker")
        self.declare_parameter("shooting_marker_topic", "/task4/shooting_markers")
        self.declare_parameter("pure_pursuit_set_active_service", "/pure_pursuit_node/set_active")
        self.declare_parameter("pure_pursuit_status_topic", "/pure_pursuit_node/status")
        self.declare_parameter("aim_target_topic", "/target_point")
        self.declare_parameter("aim_target_frame", "base_link")
        self.declare_parameter("aim_target_interval_sec", 5.0)
        self.declare_parameter("tf_timeout_sec", 0.5)

        self.callback_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_pub = self.create_publisher(Path, str(self.get_parameter("plan_topic").value), 10)
        self.polygon_marker_pub = self.create_publisher(
            Marker,
            str(self.get_parameter("polygon_marker_topic").value),
            10,
        )
        self.shooting_marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("shooting_marker_topic").value),
            10,
        )
        self.tracker_active_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("tracker_active_topic").value),
            10,
        )
        self.aim_target_pub = self.create_publisher(
            Point,
            str(self.get_parameter("aim_target_topic").value),
            10,
        )

        self.create_subscription(
            TrackedObjectArray,
            str(self.get_parameter("tracked_objects_topic").value),
            self.tracked_objects_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("pure_pursuit_status_topic").value),
            self.pure_pursuit_status_callback,
            10,
            callback_group=self.callback_group,
        )

        self.path_tracking_active_client = self.create_client(
            SetBool,
            str(self.get_parameter("pure_pursuit_set_active_service").value),
            callback_group=self.callback_group,
        )
        self.tracker_reset_client = self.create_client(
            Trigger,
            str(self.get_parameter("tracker_reset_service").value),
            callback_group=self.callback_group,
        )
        self.detector = DetectorClient(
            self,
            namespace=str(self.get_parameter("detector_namespace").value),
            callback_group=self.callback_group,
        )

        self.add_on_set_parameters_callback(self.on_parameter_change)
        self.create_service(
            Trigger,
            "/task4/plan_coverage",
            self.plan_coverage_callback,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/task4/start_navigation",
            self.start_navigation_callback,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/task4/stop_navigation",
            self.stop_navigation_callback,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/task4/reset",
            self.reset_callback,
            callback_group=self.callback_group,
        )

        self.state = MissionState.IDLE
        self.paused_state = MissionState.IDLE
        self.last_error = ""
        self.last_pp_status = ""
        self.object_map: Dict[int, ShotTarget] = {}
        self.coverage_polygon: List[Point2D] = []
        self.coverage_path: List[Waypoint] = []
        self.shooting_poses: List[ShootingPose] = []
        self.uncovered_targets: List[ShotTarget] = []
        self.current_shot_index = 0
        self.current_aim_targets: List[ShotTarget] = []
        self.current_aim_target_index = 0
        self.next_aim_target_time = 0.0
        self.detector_initialized = False
        self.detector_started = False
        self.service_coords_set = False

        self.create_timer(0.2, self.control_loop, callback_group=self.callback_group)

        self.get_logger().info(
            "Task4 brain bereit. Planen: /task4/plan_coverage, Start: /task4/start_navigation"
        )

    def on_parameter_change(self, params):
        for param in params:
            if param.name == "polygon_coords":
                success, reason = validate_polygon_coords(param.value)
                if not success:
                    return SetParametersResult(successful=False, reason=reason)
                self.service_coords_set = True
        return SetParametersResult(successful=True)

    def plan_coverage_callback(self, request, response):
        success, message = self.plan_coverage()
        response.success = success
        response.message = message
        return response

    def start_navigation_callback(self, request, response):
        success, message = self.start_navigation()
        response.success = success
        response.message = message
        return response

    def stop_navigation_callback(self, request, response):
        success, message = self.pause_navigation("Task4 wurde unterbrochen.")
        response.success = success
        response.message = message
        return response

    def reset_callback(self, request, response):
        self.reset_mission()
        response.success = True
        response.message = "Task4 zurueckgesetzt."
        return response

    def plan_coverage(self) -> Tuple[bool, str]:
        if self.state in (
            MissionState.DRIVE_COVERAGE,
            MissionState.PLAN_SHOOTING,
            MissionState.DRIVE_TO_SHOOT_POSE,
            MissionState.AIM_AND_FIRE,
        ):
            return False, f"Task4 kann im Zustand '{self.state}' nicht neu planen."

        self.set_state(MissionState.PREPARE_COVERAGE)
        self.last_error = ""
        self.last_pp_status = ""
        self.clear_planned_mission()

        try:
            polygon_points, start_waypoint = self.load_polygon_and_start_pose()
            self.coverage_polygon = polygon_points
            self.publish_polygon_marker(polygon_points)

            coverage_path = plan_coverage_path(
                polygon_points,
                (start_waypoint.x, start_waypoint.y),
                self.coverage_config(),
            )
            coverage_path = self.remove_return_to_start_if_needed(
                coverage_path,
                (start_waypoint.x, start_waypoint.y),
            )
            if len(coverage_path) < 2:
                raise RuntimeError("Coverage-Pfad enthaelt weniger als zwei Wegpunkte.")

            self.coverage_path = coverage_path
            self.publish_path(coverage_path, self.target_frame())
            self.set_state(MissionState.COVERAGE_READY)
            return True, "Coverage geplant. Navigation noch nicht gestartet."
        except Exception as exc:
            self.fail_mission(f"Coverage-Planung fehlgeschlagen: {exc}")
            return False, self.last_error

    def start_navigation(self) -> Tuple[bool, str]:
        if self.state == MissionState.PAUSED:
            return self.resume_navigation()

        if self.state != MissionState.COVERAGE_READY:
            return False, "Erst /task4/plan_coverage aufrufen."

        if len(self.coverage_path) < 2:
            return False, "Kein gueltiger Coverage-Pfad vorhanden. Erst /task4/plan_coverage aufrufen."

        try:
            self.last_pp_status = ""
            self.start_detector_for_coverage()
            self.reset_tracker()
            self.set_tracker_active(True)
            self.publish_path(self.coverage_path, self.target_frame())

            success, message = self.set_path_tracking_active(True)
            if not success:
                raise RuntimeError(message)

            self.set_state(MissionState.DRIVE_COVERAGE)
            return True, "Coverage-Fahrt gestartet."
        except Exception as exc:
            self.fail_mission(f"Coverage-Start fehlgeschlagen: {exc}")
            return False, self.last_error

    def control_loop(self):
        if self.state == MissionState.DRIVE_COVERAGE:
            if self.last_pp_status == "completed":
                self.finish_coverage_drive()
            return

        if self.state == MissionState.PLAN_SHOOTING:
            self.plan_shooting_phase()
            return

        if self.state == MissionState.DRIVE_TO_SHOOT_POSE:
            if self.last_pp_status == "completed":
                self.begin_aim_and_fire()
            return

        if self.state == MissionState.AIM_AND_FIRE:
            self.publish_next_aim_target_if_due()
            return

    def finish_coverage_drive(self):
        self.set_path_tracking_active(False)
        self.set_tracker_active(False)
        self.stop_detector_for_coverage("Coverage-Fahrt beendet.")
        self.set_state(MissionState.PLAN_SHOOTING)

    def plan_shooting_phase(self):
        targets = list(self.object_map.values())
        if not targets:
            self.set_state(MissionState.DONE)
            self.get_logger().info("Coverage beendet, keine Objekte erkannt.")
            return

        try:
            current = self.current_robot_waypoint()
            shooting_poses, uncovered = plan_shooting_poses(
                targets,
                self.coverage_polygon,
                self.coverage_path,
                (current.x, current.y),
                self.shooting_config(),
            )
        except Exception as exc:
            self.fail_mission(f"Schussplanung fehlgeschlagen: {exc}")
            return

        self.shooting_poses = shooting_poses
        self.uncovered_targets = uncovered
        self.current_shot_index = 0
        self.publish_shooting_markers()

        if uncovered:
            self.get_logger().warn(
                f"{len(uncovered)} Objekte konnten keinem Schusspunkt zugeordnet werden."
            )

        if not shooting_poses:
            self.fail_mission("Keine erreichbaren Schusspunkte gefunden.")
            return

        self.start_next_shot_navigation()

    def start_next_shot_navigation(self):
        if self.current_shot_index >= len(self.shooting_poses):
            if self.uncovered_targets:
                self.fail_mission(
                    f"Schussfahrt beendet, aber {len(self.uncovered_targets)} Ziele blieben unerreichbar."
                )
            else:
                self.set_state(MissionState.DONE)
                self.get_logger().info("Task4 abgeschlossen.")
            return

        pose = self.shooting_poses[self.current_shot_index]
        try:
            current = self.current_robot_waypoint()
            target = Waypoint(x=pose.x, y=pose.y, yaw=pose.yaw)
            if distance_2d((current.x, current.y), (target.x, target.y)) < float(
                self.get_parameter("min_navigation_segment_m").value
            ):
                path = [target]
            else:
                path = [current, target]

            self.last_pp_status = ""
            self.publish_path(path, self.target_frame())
            success, message = self.set_path_tracking_active(True)
            if not success:
                raise RuntimeError(message)
            self.set_state(MissionState.DRIVE_TO_SHOOT_POSE)
            self.get_logger().info(
                f"Fahre Schusspunkt {self.current_shot_index + 1}/{len(self.shooting_poses)} an."
            )
        except Exception as exc:
            self.fail_mission(f"Schusspunkt konnte nicht angefahren werden: {exc}")

    def begin_aim_and_fire(self):
        pose = self.shooting_poses[self.current_shot_index]
        targets = [self.object_map[target_id] for target_id in pose.target_ids if target_id in self.object_map]
        if not targets:
            self.current_shot_index += 1
            self.start_next_shot_navigation()
            return

        self.current_aim_targets = targets
        self.current_aim_target_index = 0
        self.next_aim_target_time = 0.0
        self.set_state(MissionState.AIM_AND_FIRE)
        self.get_logger().info(
            f"Schusspunkt {self.current_shot_index + 1}: "
            f"{len(self.current_aim_targets)} Zielpunkte werden auf "
            f"{self.get_parameter('aim_target_topic').value} publiziert."
        )

    def publish_next_aim_target_if_due(self):
        if self.current_aim_target_index >= len(self.current_aim_targets):
            self.current_aim_targets = []
            self.current_aim_target_index = 0
            self.current_shot_index += 1
            self.start_next_shot_navigation()
            return

        now = time.monotonic()
        if self.next_aim_target_time > 0.0 and now < self.next_aim_target_time:
            return

        target = self.current_aim_targets[self.current_aim_target_index]
        try:
            point = self.target_to_aim_point(target)
        except Exception as exc:
            self.fail_mission(f"Ziel {target.target_id} konnte nicht nach aim_target_frame transformiert werden: {exc}")
            return

        self.aim_target_pub.publish(point)
        self.current_aim_target_index += 1
        self.next_aim_target_time = now + max(
            0.0,
            float(self.get_parameter("aim_target_interval_sec").value),
        )

        self.get_logger().info(
            f"Ziel {target.target_id} auf /target_point publiziert "
            f"({self.current_aim_target_index}/{len(self.current_aim_targets)})."
        )

    def tracked_objects_callback(self, msg: TrackedObjectArray):
        if self.state != MissionState.DRIVE_COVERAGE:
            return

        for tracked in msg.objects:
            try:
                point = self.transform_point_to_target_frame(msg.header.frame_id, tracked.position)
            except Exception as exc:
                self.get_logger().warn(f"Objekt {tracked.id} konnte nicht transformiert werden: {exc}")
                continue

            self.object_map[int(tracked.id)] = ShotTarget(
                target_id=int(tracked.id),
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
                label=str(tracked.label),
            )

    def pure_pursuit_status_callback(self, msg: String):
        self.last_pp_status = str(msg.data)

    def load_polygon_and_start_pose(self) -> Tuple[List[Point2D], Waypoint]:
        coords = list(self.get_parameter("polygon_coords").value)
        success, reason = validate_polygon_coords(coords)
        if not success:
            raise ValueError(reason)

        input_frame = str(self.get_parameter("input_frame").value)
        target_frame = self.target_frame()
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            input_frame,
            Time(),
            timeout=Duration(seconds=self.tf_timeout_sec()),
        )

        polygon_points = []
        for x, y in coords_to_points(coords):
            point_in = PointStamped()
            point_in.header.frame_id = input_frame
            point_in.point.x = x
            point_in.point.y = y
            point_in.point.z = 0.0
            point_out = do_transform_point(point_in, transform)
            polygon_points.append((point_out.point.x, point_out.point.y))

        quat = transform.transform.rotation
        yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
        start = Waypoint(
            x=float(transform.transform.translation.x),
            y=float(transform.transform.translation.y),
            yaw=float(yaw),
        )
        return polygon_points, start

    def current_robot_waypoint(self) -> Waypoint:
        target_frame = self.target_frame()
        base_frame = str(self.get_parameter("input_frame").value)
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            base_frame,
            Time(),
            timeout=Duration(seconds=self.tf_timeout_sec()),
        )
        quat = transform.transform.rotation
        yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
        return Waypoint(
            x=float(transform.transform.translation.x),
            y=float(transform.transform.translation.y),
            yaw=float(yaw),
        )

    def transform_point_to_target_frame(self, source_frame: str, point: Point) -> Point:
        if not source_frame or source_frame == self.target_frame():
            return point

        point_in = PointStamped()
        point_in.header.frame_id = source_frame
        point_in.point = point
        transform = self.tf_buffer.lookup_transform(
            self.target_frame(),
            source_frame,
            Time(),
            timeout=Duration(seconds=self.tf_timeout_sec()),
        )
        return do_transform_point(point_in, transform).point

    def target_to_aim_point(self, target: ShotTarget) -> Point:
        aim_frame = str(self.get_parameter("aim_target_frame").value)
        if not aim_frame:
            aim_frame = str(self.get_parameter("input_frame").value)

        point = self.point_from_target(target)
        if aim_frame == self.target_frame():
            return point

        point_in = PointStamped()
        point_in.header.frame_id = self.target_frame()
        point_in.point = point
        transform = self.tf_buffer.lookup_transform(
            aim_frame,
            self.target_frame(),
            Time(),
            timeout=Duration(seconds=self.tf_timeout_sec()),
        )
        return do_transform_point(point_in, transform).point

    def remove_return_to_start_if_needed(
        self,
        path: Sequence[Waypoint],
        start_xy: Point2D,
    ) -> List[Waypoint]:
        path = list(path)
        if not path:
            return path

        tolerance = float(self.get_parameter("return_start_tolerance_m").value)
        last_xy = (path[-1].x, path[-1].y)
        if same_point(last_xy, start_xy, abs_tol=tolerance):
            self.get_logger().info("Letzten Coverage-Punkt entfernt: Rueckkehr zum Startpunkt erkannt.")
            return path[:-1]
        return path

    def publish_path(self, waypoints: Sequence[Waypoint], frame_id: str):
        ros_path = Path()
        ros_path.header.frame_id = frame_id
        ros_path.header.stamp = self.get_clock().now().to_msg()

        for waypoint in waypoints:
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = float(waypoint.x)
            pose.pose.position.y = float(waypoint.y)
            pose.pose.position.z = 0.0
            q = quaternion_from_euler(0.0, 0.0, float(waypoint.yaw))
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)
        self.get_logger().info(f"Pfad mit {len(ros_path.poses)} Wegpunkten publiziert.")

    def publish_polygon_marker(self, coords: Sequence[Point2D]):
        marker = Marker()
        marker.header.frame_id = self.target_frame()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "coverage_polygon"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0

        for x, y in coords:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            marker.points.append(point)

        if coords:
            point = Point()
            point.x = float(coords[0][0])
            point.y = float(coords[0][1])
            point.z = 0.0
            marker.points.append(point)

        self.polygon_marker_pub.publish(marker)

    def publish_shooting_markers(self):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.target_frame()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for index, pose in enumerate(self.shooting_poses):
            marker = Marker()
            marker.header.frame_id = self.target_frame()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "shooting_poses"
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = float(pose.x)
            marker.pose.position.y = float(pose.y)
            marker.pose.position.z = 0.05
            q = quaternion_from_euler(0.0, 0.0, float(pose.yaw))
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = 0.45
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 1.0
            markers.markers.append(marker)

        for target in self.object_map.values():
            marker = Marker()
            marker.header.frame_id = self.target_frame()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "task4_targets"
            marker.id = int(target.target_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(target.x)
            marker.pose.position.y = float(target.y)
            marker.pose.position.z = float(target.z) + 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18
            marker.color.r = 1.0
            marker.color.g = 0.35
            marker.color.b = 0.1
            marker.color.a = 0.9
            markers.markers.append(marker)

        self.shooting_marker_pub.publish(markers)

    def set_tracker_active(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.tracker_active_pub.publish(msg)

    def reset_tracker(self):
        if not self.tracker_reset_client.service_is_ready():
            self.tracker_reset_client.wait_for_service(timeout_sec=0.2)

        if not self.tracker_reset_client.service_is_ready():
            self.get_logger().warn("Tracker-Reset-Service ist nicht erreichbar.")
            return

        request = Trigger.Request()
        future = self.tracker_reset_client.call_async(request)
        completed = Event()
        future.add_done_callback(lambda _: completed.set())
        completed.wait(timeout=1.0)

    def start_detector_for_coverage(self):
        if not bool(self.get_parameter("detector_enabled").value):
            self.get_logger().info("Detector ist per Parameter deaktiviert.")
            return

        model_path = str(self.get_parameter("detector_model_path").value)
        if not model_path:
            message = "detector_model_path ist leer; Detector wird nicht gestartet."
            if bool(self.get_parameter("detector_required").value):
                raise RuntimeError(message)
            self.get_logger().warn(message)
            return

        timeout_sec = float(self.get_parameter("detector_service_timeout_sec").value)
        if not self.detector.wait_for_services(timeout_sec=timeout_sec):
            message = "Detector-Services sind nicht erreichbar."
            if bool(self.get_parameter("detector_required").value):
                raise RuntimeError(message)
            self.get_logger().warn(message)
            return

        config = self.detector_config()
        self.call_detector_service_sync(
            self.detector.init_from_config(config),
            "init",
            timeout_sec,
        )
        self.detector_initialized = True

        self.call_detector_service_sync(
            self.detector.start(),
            "start",
            timeout_sec,
        )
        self.detector_started = True
        self.detector.clear_results()
        self.get_logger().info(
            f"Detector gestartet: model='{config.model_path}', classes={list(config.classes)}"
        )

    def stop_detector_for_coverage(self, reason: str):
        if not bool(self.get_parameter("detector_enabled").value):
            return

        timeout_sec = float(self.get_parameter("detector_service_timeout_sec").value)

        if self.detector_started:
            try:
                self.call_detector_service_sync(
                    self.detector.stop(),
                    "stop",
                    timeout_sec,
                    raise_on_failure=False,
                )
            finally:
                self.detector_started = False

        if self.detector_initialized and bool(self.get_parameter("detector_release_after_coverage").value):
            try:
                self.call_detector_service_sync(
                    self.detector.release(),
                    "release",
                    timeout_sec,
                    raise_on_failure=False,
                )
            finally:
                self.detector_initialized = False

        self.get_logger().info(f"Detector fuer Coverage gestoppt: {reason}")

    def call_detector_service_sync(
        self,
        future,
        action_name: str,
        timeout_sec: float,
        raise_on_failure: bool = True,
    ):
        completed = Event()
        future.add_done_callback(lambda _: completed.set())

        if not completed.wait(timeout=max(0.1, timeout_sec)):
            message = f"Timeout beim Detector-Service '{action_name}'."
            if raise_on_failure:
                raise RuntimeError(message)
            self.get_logger().warn(message)
            return None

        try:
            response = future.result()
        except Exception as exc:
            message = f"Detector-Service '{action_name}' fehlgeschlagen: {exc}"
            if raise_on_failure:
                raise RuntimeError(message)
            self.get_logger().warn(message)
            return None

        if response is None:
            message = f"Detector-Service '{action_name}' lieferte keine Antwort."
            if raise_on_failure:
                raise RuntimeError(message)
            self.get_logger().warn(message)
            return None

        success = bool(getattr(response, "success", False))
        message = str(getattr(response, "message", ""))
        if not success:
            full_message = f"Detector-Service '{action_name}' meldet Fehler: {message}"
            if raise_on_failure:
                raise RuntimeError(full_message)
            self.get_logger().warn(full_message)

        return response

    def set_path_tracking_active(self, active: bool) -> Tuple[bool, str]:
        if not self.path_tracking_active_client.service_is_ready():
            self.path_tracking_active_client.wait_for_service(timeout_sec=0.5)

        if not self.path_tracking_active_client.service_is_ready():
            return False, "Service '/pure_pursuit_node/set_active' ist nicht erreichbar."

        request = SetBool.Request()
        request.data = bool(active)
        future = self.path_tracking_active_client.call_async(request)
        completed = Event()
        future.add_done_callback(lambda _: completed.set())

        if not completed.wait(timeout=2.0):
            return False, "Timeout beim Aufruf von '/pure_pursuit_node/set_active'."

        try:
            service_response = future.result()
        except Exception as exc:
            return False, f"Serviceaufruf fehlgeschlagen: {exc}"

        return bool(service_response.success), service_response.message

    def abort_mission(self, reason: str):
        self.set_path_tracking_active(False)
        self.set_tracker_active(False)
        self.stop_detector_for_coverage(reason)
        self.current_aim_targets = []
        self.current_aim_target_index = 0
        self.last_error = reason
        self.set_state(MissionState.ABORTED)

    def pause_navigation(self, reason: str) -> Tuple[bool, str]:
        if self.state not in (
            MissionState.DRIVE_COVERAGE,
            MissionState.DRIVE_TO_SHOOT_POSE,
            MissionState.AIM_AND_FIRE,
            MissionState.PLAN_SHOOTING,
        ):
            self.set_path_tracking_active(False)
            return True, "Keine laufende Task4-Navigation aktiv."

        self.paused_state = self.state
        self.set_path_tracking_active(False)
        self.set_tracker_active(False)
        self.stop_detector_for_coverage(reason)
        self.last_error = reason
        self.set_state(MissionState.PAUSED)
        return True, "Task4 Navigation unterbrochen."

    def resume_navigation(self) -> Tuple[bool, str]:
        if self.paused_state == MissionState.DRIVE_COVERAGE:
            if len(self.coverage_path) < 2:
                return False, "Unterbrochene Coverage-Fahrt kann nicht fortgesetzt werden: kein Coverage-Pfad vorhanden."
            self.start_detector_for_coverage()
            self.set_tracker_active(True)
            success, message = self.set_path_tracking_active(True)
            if success:
                self.set_state(MissionState.DRIVE_COVERAGE)
            return success, message

        if self.paused_state == MissionState.DRIVE_TO_SHOOT_POSE:
            success, message = self.set_path_tracking_active(True)
            if success:
                self.set_state(MissionState.DRIVE_TO_SHOOT_POSE)
            return success, message

        if self.paused_state == MissionState.AIM_AND_FIRE:
            self.set_state(MissionState.AIM_AND_FIRE)
            return True, "Aim-and-fire fortgesetzt."

        if self.paused_state == MissionState.PLAN_SHOOTING:
            self.set_state(MissionState.PLAN_SHOOTING)
            return True, "Schussplanung fortgesetzt."

        return False, "Task4 ist unterbrochen, hat aber keinen fortsetzbaren Navigationszustand."

    def reset_mission(self):
        self.set_path_tracking_active(False)
        self.set_tracker_active(False)
        self.stop_detector_for_coverage("Task4 Reset.")
        self.clear_planned_mission()
        self.last_error = ""
        self.last_pp_status = ""
        self.detector_initialized = False
        self.detector_started = False
        self.paused_state = MissionState.IDLE
        self.set_state(MissionState.IDLE)

    def clear_planned_mission(self):
        self.object_map.clear()
        self.coverage_polygon = []
        self.coverage_path = []
        self.shooting_poses = []
        self.uncovered_targets = []
        self.current_shot_index = 0
        self.current_aim_targets = []
        self.current_aim_target_index = 0
        self.next_aim_target_time = 0.0

    def fail_mission(self, reason: str):
        self.set_path_tracking_active(False)
        self.set_tracker_active(False)
        self.stop_detector_for_coverage(reason)
        self.current_aim_targets = []
        self.current_aim_target_index = 0
        self.last_error = reason
        self.set_state(MissionState.FAILED)
        self.get_logger().error(reason)

    def set_state(self, new_state: str):
        if self.state != new_state:
            self.get_logger().info(f"Task4 state: {self.state} -> {new_state}")
            self.state = new_state

    def coverage_config(self) -> CoveragePlannerConfig:
        return CoveragePlannerConfig(
            operating_width=float(self.get_parameter("operating_width").value),
            robot_width=float(self.get_parameter("robot_width").value),
            headland_width=float(self.get_parameter("headland_width").value),
            turn_radius=float(self.get_parameter("turn_radius").value),
        )

    def shooting_config(self) -> ShootingPlannerConfig:
        return ShootingPlannerConfig(
            shooting_range_m=float(self.get_parameter("shooting_range_m").value),
            shoot_angle_min_deg=float(self.get_parameter("shoot_angle_min_deg").value),
            shoot_angle_max_deg=float(self.get_parameter("shoot_angle_max_deg").value),
            candidate_grid_spacing_m=float(self.get_parameter("candidate_grid_spacing_m").value),
            yaw_sample_step_deg=float(self.get_parameter("yaw_sample_step_deg").value),
            path_candidate_stride_m=float(self.get_parameter("path_candidate_stride_m").value),
            object_ring_distance_ratio=float(self.get_parameter("object_ring_distance_ratio").value),
            object_ring_angle_step_deg=float(self.get_parameter("object_ring_angle_step_deg").value),
        )

    def detector_config(self) -> DetectorInitConfig:
        return DetectorInitConfig(
            model_path=str(self.get_parameter("detector_model_path").value),
            classes=list(self.get_parameter("detector_classes").value),
            confidence=float(self.get_parameter("detector_confidence").value),
            use_realsense_ros_wrapper=bool(
                self.get_parameter("detector_use_realsense_ros_wrapper").value
            ),
            model_type=str(self.get_parameter("detector_model_type").value),
            use_decimation=bool(self.get_parameter("detector_use_decimation").value),
            use_spatial=bool(self.get_parameter("detector_use_spatial").value),
            use_temporal=bool(self.get_parameter("detector_use_temporal").value),
            use_hole_filling=bool(self.get_parameter("detector_use_hole_filling").value),
            use_mask_filter=bool(self.get_parameter("detector_use_mask_filter").value),
            color_resolution_width=int(
                self.get_parameter("detector_color_resolution_width").value
            ),
            color_resolution_height=int(
                self.get_parameter("detector_color_resolution_height").value
            ),
            fps=int(self.get_parameter("detector_fps").value),
            rcnn_class_names=list(self.get_parameter("detector_rcnn_class_names").value),
        )

    def point_from_target(self, target: ShotTarget) -> Point:
        point = Point()
        point.x = float(target.x)
        point.y = float(target.y)
        point.z = float(target.z)
        return point

    def target_frame(self) -> str:
        return str(self.get_parameter("target_frame").value)

    def tf_timeout_sec(self) -> float:
        return float(self.get_parameter("tf_timeout_sec").value)


def main(args=None):
    rclpy.init(args=args)
    node = Task4Brain()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
