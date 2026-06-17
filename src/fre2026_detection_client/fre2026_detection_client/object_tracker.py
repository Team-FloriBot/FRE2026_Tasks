#!/usr/bin/env python3

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time

from fre2026_detection_interfaces.msg import TrackedObject, TrackedObjectArray
from geometry_msgs.msg import PointStamped
from ros2_detection_interfaces.msg import DetectionArray
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class TrackedState:
    object_id: int
    label: Optional[str]
    label_scores: Dict[str, float] = field(default_factory=dict)
    last_seen: object = None
    position: object = None
    counter: int = 0
    last_detection_id: int = 0


class ObjectTracker(Node):
    def __init__(self) -> None:
        super().__init__("object_tracker")

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("detector_results_topic", "/detector/results")
        self.declare_parameter("tracked_objects_topic", "/tracker/tracked_objects")
        self.declare_parameter("marker_topic", "/tracker/tracked_object_markers")
        self.declare_parameter("active_topic", "/tracker/active")
        self.declare_parameter("reset_service", "/tracker/reset")
        self.declare_parameter("match_distance", 0.30)
        self.declare_parameter("position_smoothing_alpha", 0.30)
        self.declare_parameter("min_observations_for_publish", 3)
        self.declare_parameter("min_observations_for_label", 3)
        self.declare_parameter("unlabeled_timeout_sec", 10.0)
        self.declare_parameter("tf_timeout_sec", 1.0)
        self.declare_parameter("tf_lookup_offset_sec", 0.0)
        self.declare_parameter("publish_ground_z", True)
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("publish_tentative_markers", True)
        self.declare_parameter("marker_z_offset", 0.15)
        self.declare_parameter("marker_sphere_scale", 0.18)
        self.declare_parameter("marker_text_scale", 0.28)
        self.declare_parameter("simulation_enabled", False)
        self.declare_parameter("simulation_publish_rate", 2.0)
        self.declare_parameter("simulation_object_count", 5)
        self.declare_parameter("simulation_seed", 0)
        self.declare_parameter("simulation_label", "sim_object")
        self.declare_parameter("simulation_x_min", 0.0)
        self.declare_parameter("simulation_x_max", 5.0)
        self.declare_parameter("simulation_y_min", -1.0)
        self.declare_parameter("simulation_y_max", 1.0)
        self.declare_parameter("simulation_z", 0.0)

        self.target_frame = str(self.get_parameter("target_frame").value)
        detector_results_topic = str(self.get_parameter("detector_results_topic").value)
        tracked_objects_topic = str(self.get_parameter("tracked_objects_topic").value)
        marker_topic = str(self.get_parameter("marker_topic").value)
        active_topic = str(self.get_parameter("active_topic").value)
        reset_service = str(self.get_parameter("reset_service").value)

        self.match_distance = float(self.get_parameter("match_distance").value)
        self.position_smoothing_alpha = float(self.get_parameter("position_smoothing_alpha").value)
        self.min_observations_for_publish = int(
            self.get_parameter("min_observations_for_publish").value
        )
        self.min_observations_for_label = int(self.get_parameter("min_observations_for_label").value)
        self.unlabeled_timeout_sec = float(self.get_parameter("unlabeled_timeout_sec").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.tf_lookup_offset_sec = float(self.get_parameter("tf_lookup_offset_sec").value)
        self.publish_ground_z = bool(self.get_parameter("publish_ground_z").value)
        self.ground_z = float(self.get_parameter("ground_z").value)
        self.publish_tentative_markers = bool(
            self.get_parameter("publish_tentative_markers").value
        )
        self.marker_z_offset = float(self.get_parameter("marker_z_offset").value)
        self.marker_sphere_scale = float(self.get_parameter("marker_sphere_scale").value)
        self.marker_text_scale = float(self.get_parameter("marker_text_scale").value)
        self.simulation_enabled = bool(self.get_parameter("simulation_enabled").value)
        self.simulation_publish_rate = float(self.get_parameter("simulation_publish_rate").value)
        self.simulation_object_count = int(self.get_parameter("simulation_object_count").value)
        self.simulation_seed = int(self.get_parameter("simulation_seed").value)
        self.simulation_label = str(self.get_parameter("simulation_label").value)
        self.simulation_x_min = float(self.get_parameter("simulation_x_min").value)
        self.simulation_x_max = float(self.get_parameter("simulation_x_max").value)
        self.simulation_y_min = float(self.get_parameter("simulation_y_min").value)
        self.simulation_y_max = float(self.get_parameter("simulation_y_max").value)
        self.simulation_z = float(self.get_parameter("simulation_z").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        tracker_active_qos = QoSProfile(depth=1)
        tracker_active_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.results_sub = self.create_subscription(
            DetectionArray,
            detector_results_topic,
            self.detection_results_callback,
            2,
        )
        self.active_sub = self.create_subscription(
            Bool,
            active_topic,
            self.active_callback,
            tracker_active_qos,
        )
        self.tracked_objects_pub = self.create_publisher(TrackedObjectArray, tracked_objects_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.reset_srv = self.create_service(Trigger, reset_service, self.reset_callback)

        self.active = False
        self.next_id = 0
        self.objects: Dict[int, TrackedState] = {}
        self.simulated_objects: Optional[TrackedObjectArray] = None
        self.simulation_timer = None
        if self.simulation_enabled:
            self.simulated_objects = self.create_simulated_objects()
            period_sec = 1.0 / max(0.1, self.simulation_publish_rate)
            self.simulation_timer = self.create_timer(period_sec, self.publish_simulated_objects)
            self.get_logger().info(
                f"Object tracker simulation enabled with {self.simulation_object_count} objects"
            )

        self.get_logger().info(
            f"Object tracker ready: {detector_results_topic} -> {tracked_objects_topic} in {self.target_frame}"
        )

    def active_callback(self, msg: Bool) -> None:
        requested_active = bool(msg.data)
        if requested_active != self.active:
            self.get_logger().info(
                f"Object tracker {'activated' if requested_active else 'deactivated'}"
            )
        self.active = requested_active

    def reset_callback(self, request, response):
        self.reset_tracks()
        response.success = True
        response.message = "Object tracker reset"
        self.get_logger().info("Object tracker reset requested via service")
        return response

    def reset_tracks(self) -> None:
        object_count = len(self.objects)
        self.objects.clear()
        self.next_id = 0
        self.simulated_objects = self.create_simulated_objects() if self.simulation_enabled else None
        self.publish_delete_all_markers()
        if self.simulated_objects is not None:
            self.publish_simulated_objects()
        self.get_logger().info(f"Object tracker cleared {object_count} tracked objects")

    def detection_results_callback(self, msg: DetectionArray) -> None:
        if self.simulation_enabled:
            self.get_logger().debug(
                "Ignoring detector results because simulation mode is enabled",
                throttle_duration_sec=5.0,
            )
            return
        if not self.active:
            self.get_logger().debug(
                "Ignoring detector results because tracker is inactive",
                throttle_duration_sec=5.0,
            )
            return

        detection_count = len(msg.detections)
        if detection_count == 0:
            self.get_logger().debug(
                "Received empty detector result batch",
                throttle_duration_sec=5.0,
            )
            return

        transform = self.lookup_detection_transform(msg)
        if transform is None:
            self.get_logger().warn(
                f"Dropping {detection_count} detections because no TF is available "
                f"from {msg.header.frame_id} to {self.target_frame}",
                throttle_duration_sec=2.0,
            )
            self.publish_tracks(msg.header.stamp)
            return

        updated_before = len(self.objects)
        for detection in msg.detections:
            point_in = PointStamped()
            point_in.header.frame_id = msg.header.frame_id
            point_in.header.stamp = msg.header.stamp
            point_in.point = detection.object_center
            point_out = do_transform_point(point_in, transform)
            self.update_track(detection, point_out.point, msg.header.stamp)

        self.get_logger().debug(
            f"Processed {detection_count} detections; "
            f"tracks {updated_before} -> {len(self.objects)}",
            throttle_duration_sec=1.0,
        )
        self.publish_tracks(msg.header.stamp)

    def lookup_detection_transform(self, msg: DetectionArray):
        lookup_time = Time.from_msg(msg.header.stamp) - Duration(
            seconds=max(0.0, self.tf_lookup_offset_sec)
        )
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                lookup_time,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as stamped_exc:
            try:
                
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=self.tf_timeout_sec),
                )
            except Exception as latest_exc:
                self.get_logger().warn(
                    f"TF lookup failed: stamped={stamped_exc}; latest={latest_exc}",
                    throttle_duration_sec=2.0,
                )
                return None

            self.get_logger().debug(
                "Using latest TF for detection timestamp after stamped lookup failed: "
                f"{stamped_exc}",
                throttle_duration_sec=2.0,
            )
            return transform

    def create_simulated_objects(self) -> TrackedObjectArray:
        rng = random.Random(self.simulation_seed)
        x_min, x_max = sorted((self.simulation_x_min, self.simulation_x_max))
        y_min, y_max = sorted((self.simulation_y_min, self.simulation_y_max))

        msg = TrackedObjectArray()
        msg.header.frame_id = self.target_frame
        for idx in range(max(0, self.simulation_object_count)):
            tracked = TrackedObject()
            tracked.id = idx
            tracked.label = self.simulation_label
            tracked.position.x = rng.uniform(x_min, x_max)
            tracked.position.y = rng.uniform(y_min, y_max)
            tracked.position.z = self.simulation_z
            msg.objects.append(tracked)
        return msg

    def publish_simulated_objects(self) -> None:
        if self.simulated_objects is None:
            self.simulated_objects = self.create_simulated_objects()
        self.simulated_objects.header.stamp = self.get_clock().now().to_msg()
        self.tracked_objects_pub.publish(self.simulated_objects)
        self.publish_markers(self.simulated_objects)
        self.get_logger().debug(
            f"Published {len(self.simulated_objects.objects)} simulated tracked objects",
            throttle_duration_sec=5.0,
        )

    def update_track(self, detection, position, stamp) -> None:
        matched_id = self.find_match(position)
        confidence = float(getattr(detection, "confidence", 0.0))
        label = str(getattr(detection, "label", ""))
        detection_id = int(getattr(detection, "id", 0))

        if matched_id is None:
            object_id = self.next_id
            self.next_id += 1
            self.objects[object_id] = TrackedState(
                object_id=object_id,
                label=None,
                label_scores={label: confidence},
                last_seen=stamp,
                position=position,
                counter=1,
                last_detection_id=detection_id,
            )
            self.get_logger().info(
                f"Created track {object_id} from detection {detection_id} "
                f"label={label!r} confidence={confidence:.2f} "
                f"position=({position.x:.2f}, {position.y:.2f}, {position.z:.2f})"
            )
            return

        tracked = self.objects[matched_id]
        previous_label = tracked.label
        alpha = float(max(0.0, min(1.0, self.position_smoothing_alpha)))
        tracked.position.x = (1.0 - alpha) * tracked.position.x + alpha * position.x
        tracked.position.y = (1.0 - alpha) * tracked.position.y + alpha * position.y
        tracked.position.z = (1.0 - alpha) * tracked.position.z + alpha * position.z
        tracked.label_scores[label] = tracked.label_scores.get(label, 0.0) + confidence
        tracked.counter += 1
        if tracked.counter >= self.min_observations_for_label:
            tracked.label = max(tracked.label_scores, key=tracked.label_scores.get)
        tracked.last_seen = stamp
        tracked.last_detection_id = detection_id
        if previous_label is None and tracked.label is not None:
            self.get_logger().info(
                f"Track {tracked.object_id} confirmed as {tracked.label!r} "
                f"after {tracked.counter} observations"
            )

    def find_match(self, position) -> Optional[int]:
        best_id: Optional[int] = None
        best_distance = float("inf")
        for object_id, tracked in self.objects.items():
            if tracked.position is None:
                continue
            distance = self.horizontal_distance(tracked.position, position)
            if distance < self.match_distance and distance < best_distance:
                best_distance = distance
                best_id = object_id
        return best_id

    def publish_tracks(self, stamp) -> None:
        msg = TrackedObjectArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame

        current_time = Time.from_msg(stamp)
        expired_ids = []
        confirmed_count = 0
        for object_id, tracked in self.objects.items():
            if tracked.label is None:
                last_seen = Time.from_msg(tracked.last_seen)
                age_sec = (current_time - last_seen).nanoseconds / 1e9
                if age_sec > self.unlabeled_timeout_sec:
                    expired_ids.append(object_id)
                continue
            confirmed_count += 1

        for object_id in expired_ids:
            del self.objects[object_id]
            self.get_logger().info(
                f"Dropped unlabeled track {object_id} after "
                f"{self.unlabeled_timeout_sec:.1f}s without confirmation"
            )

        for tracked in self.objects.values():
            if tracked.counter < max(1, self.min_observations_for_publish):
                continue
            msg.objects.append(self.tracked_state_to_msg(tracked))

        self.tracked_objects_pub.publish(msg)
        marker_msg = self.build_marker_object_array(msg.header, include_tentative=True)
        self.publish_markers(marker_msg)
        self.get_logger().debug(
            f"Published {len(msg.objects)} tracked objects "
            f"({confirmed_count} confirmed) from {len(self.objects)} stored tracks",
            throttle_duration_sec=1.0,
        )

    def publish_markers(self, tracked_objects: TrackedObjectArray) -> None:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header = tracked_objects.header
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for tracked in tracked_objects.objects:
            markers.markers.append(self.create_object_marker(tracked_objects.header, tracked))
            markers.markers.append(self.create_text_marker(tracked_objects.header, tracked))

        self.marker_pub.publish(markers)

    def build_marker_object_array(self, header, include_tentative: bool) -> TrackedObjectArray:
        if not include_tentative or not self.publish_tentative_markers:
            marker_objects = TrackedObjectArray()
            marker_objects.header = header
            return marker_objects

        marker_objects = TrackedObjectArray()
        marker_objects.header = header
        for tracked in self.objects.values():
            if tracked.position is None:
                continue
            marker_objects.objects.append(self.tracked_state_to_msg(tracked))
        return marker_objects

    def tracked_state_to_msg(self, tracked: TrackedState) -> TrackedObject:
        tracked_msg = TrackedObject()
        tracked_msg.id = tracked.object_id
        tracked_msg.label = tracked.label or self.best_tentative_label(tracked)
        tracked_msg.position.x = tracked.position.x
        tracked_msg.position.y = tracked.position.y
        tracked_msg.position.z = self.output_z(tracked.position)
        return tracked_msg

    def publish_delete_all_markers(self) -> None:
        if not hasattr(self, "marker_pub"):
            return
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def create_object_marker(self, header, tracked: TrackedObject) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "tracked_objects"
        marker.id = int(tracked.id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = tracked.position.x
        marker.pose.position.y = tracked.position.y
        marker.pose.position.z = tracked.position.z + self.marker_z_offset
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.marker_sphere_scale
        marker.scale.y = self.marker_sphere_scale
        marker.scale.z = self.marker_sphere_scale
        marker.color.r = 1.0
        marker.color.g = 0.72
        marker.color.b = 0.08
        marker.color.a = 0.95
        return marker

    def create_text_marker(self, header, tracked: TrackedObject) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = "tracked_object_labels"
        marker.id = int(tracked.id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = tracked.position.x
        marker.pose.position.y = tracked.position.y
        marker.pose.position.z = tracked.position.z + self.marker_z_offset + 0.25
        marker.pose.orientation.w = 1.0
        marker.scale.z = self.marker_text_scale
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = f"id={tracked.id} {tracked.label}"
        return marker

    @staticmethod
    def distance(first, second) -> float:
        return math.sqrt(
            (first.x - second.x) ** 2
            + (first.y - second.y) ** 2
            + (first.z - second.z) ** 2
        )

    @staticmethod
    def horizontal_distance(first, second) -> float:
        return math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2)

    def output_z(self, position) -> float:
        if self.publish_ground_z:
            return self.ground_z
        return float(position.z)

    @staticmethod
    def best_tentative_label(tracked: TrackedState) -> str:
        if not tracked.label_scores:
            return "tentative"
        label = max(tracked.label_scores, key=tracked.label_scores.get)
        return str(label) if label else "tentative"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
