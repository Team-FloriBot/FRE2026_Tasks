#!/usr/bin/env python3
"""
Global_Marker_Map_Node

This node maintains a global map of all detected markers.
It transforms markers from camera_frame to map_frame using TF,
filters duplicates based on distance threshold, and publishes
all detected markers as PointCloud2.

The node also provides a list of active (not yet sprayed) markers
for the Pan-Tilt targeting system.

Author: FRE2026 Team
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
from tf2_geometry_msgs import PointStamped as TF2PointStamped
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker

import struct


class GlobalMarkerMap(Node):
    """Node for maintaining a global map of detected markers."""

    def __init__(self) -> None:
        super().__init__("global_marker_map")

        # Declare and get parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("detected_marker_topic", "/detected_marker"),
                ("tf_frame", "map"),
                ("duplicate_distance_threshold", 0.20),
                ("detected_markers_topic", "/detected_markers"),
                ("active_markers_topic", "/active_markers"),
                ("marker_frame", "map"),
                ("publish_rate", 10.0),
                ("max_markers", 100),
                ("marker_diameter_m", 0.05),
                ("marker_color", [0.0, 1.0, 0.0]),
                ("tf_lookup_timeout_ms", 100),
                ("tf_cache_time_sec", 10.0),
            ],
        )

        # Get parameters
        self.detected_marker_topic = self.get_parameter("detected_marker_topic").value
        self.tf_frame = self.get_parameter("tf_frame").value
        self.duplicate_distance_threshold = self.get_parameter(
            "duplicate_distance_threshold"
        ).value
        self.detected_markers_topic = self.get_parameter("detected_markers_topic").value
        self.active_markers_topic = self.get_parameter("active_markers_topic").value
        self.marker_frame = self.get_parameter("marker_frame").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.max_markers = self.get_parameter("max_markers").value
        self.marker_diameter_m = self.get_parameter("marker_diameter_m").value
        self.marker_color = self.get_parameter("marker_color").value
        self.tf_lookup_timeout_ms = self.get_parameter("tf_lookup_timeout_ms").value
        self.tf_cache_time_sec = self.get_parameter("tf_cache_time_sec").value

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=self.tf_cache_time_sec)
        )
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True
        )

        # Marker database (thread-safe)
        self.markers: list[dict] = []
        self.marker_lock = threading.Lock()

        # Create subscribers
        self.marker_sub = self.create_subscription(
            PointStamped, self.detected_marker_topic, self.marker_callback, 10
        )

        # Create publishers
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, self.detected_markers_topic, 10
        )
        self.active_markers_pub = self.create_publisher(
            MarkerArray, self.active_markers_topic, 10
        )

        # Create timer for periodic publishing
        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate, self.publish_markers
        )

        self.get_logger().info("GlobalMarkerMap node initialized")
        self.get_logger().info(f"TF frame: {self.tf_frame}")
        self.get_logger().info(
            f"Duplicate distance threshold: {self.duplicate_distance_threshold} m"
        )
        self.get_logger().info(f"Max markers: {self.max_markers}")

    def marker_callback(self, msg: PointStamped) -> None:
        """Callback for detected marker messages."""
        try:
            # Transform marker to map frame
            marker_in_map = self.transform_to_map(msg)

            if marker_in_map is None:
                return

            # Check for duplicates and add if new
            with self.marker_lock:
                if not self.is_duplicate(marker_in_map):
                    self.add_marker(marker_in_map)
                else:
                    self.get_logger().debug(
                        f"Duplicate marker detected at "
                        f"({marker_in_map['x']:.2f}, {marker_in_map['y']:.2f})"
                    )

        except Exception as e:
            self.get_logger().error(f"Error processing marker: {e}")

    def transform_to_map(self, msg: PointStamped) -> Optional[dict]:
        """Transform marker from its frame to map frame."""
        try:
            # Create TF2 PointStamped
            tf2_point = TF2PointStamped()
            tf2_point.header = msg.header
            tf2_point.point = msg.point

            # Transform to map frame
            transform = self.tf_buffer.transform(
                tf2_point,
                self.tf_frame,
                timeout=Duration(milliseconds=self.tf_lookup_timeout_ms),
            )

            return {
                "x": transform.point.x,
                "y": transform.point.y,
                "z": transform.point.z,
                "frame_id": transform.header.frame_id,
                "stamp": transform.header.stamp,
            }

        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    def is_duplicate(self, marker: dict) -> bool:
        """Check if marker is a duplicate based on distance threshold."""
        for existing_marker in self.markers:
            dx = marker["x"] - existing_marker["x"]
            dy = marker["y"] - existing_marker["y"]
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < self.duplicate_distance_threshold:
                return True

        return False

    def add_marker(self, marker: dict) -> None:
        """Add marker to database, respecting max_markers limit."""
        if len(self.markers) >= self.max_markers:
            # Remove oldest marker (first in list)
            self.markers.pop(0)

        self.markers.append(marker)

    def publish_markers(self) -> None:
        """Publish all markers as PointCloud2 and MarkerArray."""
        with self.marker_lock:
            if len(self.markers) == 0:
                return

            # Publish PointCloud2
            self.publish_pointcloud()

            # Publish MarkerArray for visualization
            self.publish_marker_array()

    def publish_pointcloud(self) -> None:
        """Publish markers as PointCloud2 message."""
        header = Header()
        header.frame_id = self.marker_frame
        header.stamp = self.get_clock().now().to_msg()

        # Create point cloud data
        points = []
        for marker in self.markers:
            # Pack x, y, z as float32
            points.append(
                struct.pack(
                    "fff",
                    marker["x"],
                    marker["y"],
                    marker["z"],
                )
            )

        # Create PointCloud2 message
        point_cloud = PointCloud2()
        point_cloud.header = header
        point_cloud.height = 1
        point_cloud.width = len(self.markers)
        point_cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud.is_bigendian = False
        point_cloud.point_step = 12
        point_cloud.row_step = point_cloud.point_step * point_cloud.width
        point_cloud.is_dense = True
        point_cloud.data = b"".join(points)

        self.pointcloud_pub.publish(point_cloud)

    def publish_marker_array(self) -> None:
        """Publish markers as MarkerArray for visualization."""
        header = Header()
        header.frame_id = self.marker_frame
        header.stamp = self.get_clock().now().to_msg()

        marker_array = MarkerArray()

        for i, marker in enumerate(self.markers):
            marker_msg = Marker()
            marker_msg.header = header
            marker_msg.ns = "detected_markers"
            marker_msg.id = i
            marker_msg.type = Marker.SPHERE
            marker_msg.action = Marker.ADD
            marker_msg.pose.position.x = marker["x"]
            marker_msg.pose.position.y = marker["y"]
            marker_msg.pose.position.z = marker["z"]
            marker_msg.pose.orientation.x = 0.0
            marker_msg.pose.orientation.y = 0.0
            marker_msg.pose.orientation.z = 0.0
            marker_msg.pose.orientation.w = 1.0
            marker_msg.scale.x = self.marker_diameter_m
            marker_msg.scale.y = self.marker_diameter_m
            marker_msg.scale.z = self.marker_diameter_m
            marker_msg.color.r = self.marker_color[0]
            marker_msg.color.g = self.marker_color[1]
            marker_msg.color.b = self.marker_color[2]
            marker_msg.color.a = 1.0
            marker_msg.lifetime = Duration(seconds=1.0 / self.publish_rate).to_msg()

            marker_array.markers.append(marker_msg)

        self.active_markers_pub.publish(marker_array)


def main(args: Optional[list] = None) -> None:
    """Main function to run the node."""
    rclpy.init(args=args)

    node = GlobalMarkerMap()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()