#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
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
        self.declare_parameter("min_observations_for_label", 8)
        self.declare_parameter("unlabeled_timeout_sec", 10.0)
        self.declare_parameter("tf_timeout_sec", 0.10)
        self.declare_parameter("marker_z_offset", 0.15)
        self.declare_parameter("marker_sphere_scale", 0.18)
        self.declare_parameter("marker_text_scale", 0.28)

        self.target_frame = str(self.get_parameter("target_frame").value)
        detector_results_topic = str(self.get_parameter("detector_results_topic").value)
        tracked_objects_topic = str(self.get_parameter("tracked_objects_topic").value)
        marker_topic = str(self.get_parameter("marker_topic").value)
        active_topic = str(self.get_parameter("active_topic").value)
        reset_service = str(self.get_parameter("reset_service").value)

        self.match_distance = float(self.get_parameter("match_distance").value)
        self.position_smoothing_alpha = float(self.get_parameter("position_smoothing_alpha").value)
        self.min_observations_for_label = int(self.get_parameter("min_observations_for_label").value)
        self.unlabeled_timeout_sec = float(self.get_parameter("unlabeled_timeout_sec").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.marker_z_offset = float(self.get_parameter("marker_z_offset").value)
        self.marker_sphere_scale = float(self.get_parameter("marker_sphere_scale").value)
        self.marker_text_scale = float(self.get_parameter("marker_text_scale").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.results_sub = self.create_subscription(
            DetectionArray,
            detector_results_topic,
            self.detection_results_callback,
            2,
        )
        self.active_sub = self.create_subscription(Bool, active_topic, self.active_callback, 10)
        self.tracked_objects_pub = self.create_publisher(TrackedObjectArray, tracked_objects_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.reset_srv = self.create_service(Trigger, reset_service, self.reset_callback)

        self.active = False
        self.next_id = 0
        self.objects: Dict[int, TrackedState] = {}

        self.get_logger().info(
            f"Object tracker ready: {detector_results_topic} -> {tracked_objects_topic} in {self.target_frame}"
        )

    def active_callback(self, msg: Bool) -> None:
        self.active = bool(msg.data)

    def reset_callback(self, request, response):
        self.reset_tracks()
        response.success = True
        response.message = "Object tracker reset"
        return response

    def reset_tracks(self) -> None:
        self.objects.clear()
        self.next_id = 0
        self.publish_delete_all_markers()

    def detection_results_callback(self, msg: DetectionArray) -> None:
        if not self.active:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                msg.header.stamp,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed: {exc}")
            return

        for detection in msg.detections:
            point_in = PointStamped()
            point_in.header.frame_id = msg.header.frame_id
            point_in.header.stamp = msg.header.stamp
            point_in.point = detection.object_center
            point_out = do_transform_point(point_in, transform)
            self.update_track(detection, point_out.point, msg.header.stamp)

        self.publish_tracks(msg.header.stamp)

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
            return

        tracked = self.objects[matched_id]
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

    def find_match(self, position) -> Optional[int]:
        best_id: Optional[int] = None
        best_distance = float("inf")
        for object_id, tracked in self.objects.items():
            if tracked.position is None:
                continue
            distance = self.distance(tracked.position, position)
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
        for object_id, tracked in self.objects.items():
            if tracked.label is None:
                last_seen = Time.from_msg(tracked.last_seen)
                age_sec = (current_time - last_seen).nanoseconds / 1e9
                if age_sec > self.unlabeled_timeout_sec:
                    expired_ids.append(object_id)
                continue

            tracked_msg = TrackedObject()
            tracked_msg.id = tracked.object_id
            tracked_msg.label = tracked.label
            tracked_msg.position.x = tracked.position.x
            tracked_msg.position.y = tracked.position.y
            tracked_msg.position.z = tracked.position.z
            msg.objects.append(tracked_msg)

        for object_id in expired_ids:
            del self.objects[object_id]

        self.tracked_objects_pub.publish(msg)
        self.publish_markers(msg)

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
