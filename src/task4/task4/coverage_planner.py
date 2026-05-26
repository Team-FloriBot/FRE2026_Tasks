#!/usr/bin/env python3
"""
Coverage Planner Node with Fields2Cover Integration

This node implements coverage path planning for autonomous field operations.
It uses the Fields2Cover library for optimal coverage patterns and provides
a fallback implementation if Fields2Cover is not available.

Key features:
- Systematic coverage patterns (e.g., boustrophedon, spiral)
- Configurable working width (spur width) and safety margin
- Field boundary detection and path generation
- Integration with Nav2 for path following

Author: FRE2026 Team
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header


class CoveragePlanner(Node):
    """Node for coverage path planning with Fields2Cover integration."""

    def __init__(self) -> None:
        super().__init__("coverage_planner")

        # Try to import Fields2Cover
        self.f2c_available = False
        try:
            from f2c import (
                F2CField,
                F2CFieldGenerator,
                F2CPathGenerator,
                F2CPathOptimizer,
            )

            self.f2c = True
            self.f2c_field = None
            self.f2c_path_generator = None
            self.get_logger().info("Fields2Cover imported successfully")
        except ImportError:
            self.get_logger().warn(
                "Fields2Cover not available, using fallback implementation"
            )

        # Declare and get parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("field_boundary_topic", "/field_boundary"),
                ("detected_markers_topic", "/detected_markers"),
                ("working_width_m", 0.75),
                ("safety_margin_m", 0.15),
                ("overlap_m", 0.10),
                ("min_turn_radius", 0.5),
                ("max_path_length", 1000),
                ("global_plan_topic", "/plan"),
                ("local_plan_topic", "/local_plan"),
                ("target_topic", "/target"),
                ("field_boundary_topic_out", "/field_boundary_visual"),
                ("base_frame", "base_link"),
                ("map_frame", "map"),
                ("update_rate", 5.0),
                ("replan_distance", 2.0),
                ("f2c_algorithm", "HillClimbing"),
                ("f2c_max_turns", 10),
            ],
        )

        # Get parameters
        self.field_boundary_topic = self.get_parameter("field_boundary_topic").value
        self.detected_markers_topic = self.get_parameter("detected_markers_topic").value
        self.working_width_m = self.get_parameter("working_width_m").value
        self.safety_margin_m = self.get_parameter("safety_margin_m").value
        self.overlap_m = self.get_parameter("overlap_m").value
        self.min_turn_radius = self.get_parameter("min_turn_radius").value
        self.max_path_length = self.get_parameter("max_path_length").value
        self.global_plan_topic = self.get_parameter("global_plan_topic").value
        self.local_plan_topic = self.get_parameter("local_plan_topic").value
        self.target_topic = self.get_parameter("target_topic").value
        self.field_boundary_topic_out = self.get_parameter(
            "field_boundary_topic_out"
        ).value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.update_rate = self.get_parameter("update_rate").value
        self.replan_distance = self.get_parameter("replan_distance").value
        self.f2c_algorithm = self.get_parameter("f2c_algorithm").value
        self.f2c_max_turns = self.get_parameter("f2c_max_turns").value

        # Initialize state
        self.field_boundary: list[dict] = []
        self.detected_markers: list[dict] = []
        self.current_path: list[dict] = []
        self.last_replan_position: Optional[dict] = None

        # Create subscribers
        self.field_boundary_sub = self.create_subscription(
            MarkerArray, self.field_boundary_topic, self.field_boundary_callback, 10
        )
        self.detected_markers_sub = self.create_subscription(
            PointCloud2, self.detected_markers_topic, self.detected_markers_callback, 10
        )

        # Create publishers
        self.global_plan_pub = self.create_publisher(Path, self.global_plan_topic, 10)
        self.local_plan_pub = self.create_publisher(Path, self.local_plan_topic, 10)
        self.target_pub = self.create_publisher(PoseStamped, self.target_topic, 10)
        self.field_boundary_pub = self.create_publisher(
            Marker, self.field_boundary_topic_out, 10
        )

        # Create timer for periodic planning
        self.plan_timer = self.create_timer(
            1.0 / self.update_rate, self.update_plan
        )

        self.get_logger().info("CoveragePlanner node initialized")
        self.get_logger().info(f"Working width: {self.working_width_m} m")
        self.get_logger().info(f"Safety margin: {self.safety_margin_m} m")
        self.get_logger().info(f"Overlap: {self.overlap_m} m")
        self.get_logger().info(f"Algorithm: {self.f2c_algorithm}")

    def field_boundary_callback(self, msg: MarkerArray) -> None:
        """Callback for field boundary markers."""
        self.field_boundary = []
        for marker in msg.markers:
            self.field_boundary.append({
                "x": marker.pose.position.x,
                "y": marker.pose.position.y,
                "z": marker.pose.position.z,
            })

        self.get_logger().debug(f"Field boundary updated: {len(self.field_boundary)} points")

    def detected_markers_callback(self, msg: PointCloud2) -> None:
        """Callback for detected markers point cloud."""
        self.detected_markers = []

        # Parse PointCloud2 data
        point_step = msg.point_step
        num_points = len(msg.data) // point_step

        for i in range(num_points):
            offset = i * point_step
            x = np.frombuffer(msg.data[offset : offset + 4], dtype=np.float32)[0]
            y = np.frombuffer(msg.data[offset + 4 : offset + 8], dtype=np.float32)[0]
            z = np.frombuffer(msg.data[offset + 8 : offset + 12], dtype=np.float32)[0]

            self.detected_markers.append({
                "x": x,
                "y": y,
                "z": z,
            })

        self.get_logger().debug(f"Detected markers updated: {len(self.detected_markers)} markers")

    def update_plan(self) -> None:
        """Update the coverage plan periodically."""
        if len(self.field_boundary) < 3:
            self.get_logger().debug("Not enough field boundary points")
            return

        if len(self.detected_markers) == 0:
            self.get_logger().debug("No detected markers")
            return

        # Calculate coverage path
        path = self.generate_coverage_path()

        if path is None or len(path) == 0:
            self.get_logger().warn("Failed to generate coverage path")
            return

        self.current_path = path
        self.last_replan_position = path[0].copy()

        # Publish global plan
        self.publish_global_plan(path)

        # Publish target
        if len(path) > 0:
            self.publish_target(path[0])

    def generate_coverage_path(self) -> Optional[list[dict]]:
        """Generate coverage path using Fields2Cover or fallback."""
        if self.f2c:
            return self.generate_coverage_path_f2c()
        else:
            return self.generate_coverage_path_fallback()

    def generate_coverage_path_f2c(self) -> Optional[list[dict]]:
        """Generate coverage path using Fields2Cover."""
        try:
            # Create field from boundary
            field = F2CField()
            boundary = F2CFieldGenerator().generateRectangularField(
                self.field_boundary[0]["x"],
                self.field_boundary[0]["y"],
                10.0,  # width
                10.0,  # height
                0.0,   # angle
            )
            field.setField(boundary)

            # Create path generator
            path_generator = F2CPathGenerator()

            # Set algorithm based on parameter
            if self.f2c_algorithm == "Greedy":
                path = path_generator.generateGreedyPath(
                    field,
                    self.working_width_m,
                    self.safety_margin_m,
                )
            elif self.f2c_algorithm == "HillClimbing":
                path = path_generator.generateHillClimbingPath(
                    field,
                    self.working_width_m,
                    self.safety_margin_m,
                    self.f2c_max_turns,
                )
            else:  # Random
                path = path_generator.generateRandomPath(
                    field,
                    self.working_width_m,
                    self.safety_margin_m,
                )

            # Convert path to list of points
            path_points = []
            for pose in path.getPoses():
                path_points.append({
                    "x": pose.getX(),
                    "y": pose.getY(),
                    "z": pose.getZ(),
                    "yaw": pose.getYaw(),
                })

            return path_points

        except Exception as e:
            self.get_logger().error(f"Fields2Cover path generation failed: {e}")
            return None

    def generate_coverage_path_fallback(self) -> Optional[list[dict]]:
        """Generate coverage path using fallback implementation."""
        if len(self.field_boundary) < 3:
            return None

        # Calculate field centroid
        centroid_x = sum(p["x"] for p in self.field_boundary) / len(self.field_boundary)
        centroid_y = sum(p["y"] for p in self.field_boundary) / len(self.field_boundary)

        # Calculate field extent
        min_x = min(p["x"] for p in self.field_boundary)
        max_x = max(p["x"] for p in self.field_boundary)
        min_y = min(p["y"] for p in self.field_boundary)
        max_y = max(p["y"] for p in self.field_boundary)

        field_width = max_x - min_x
        field_height = max_y - min_y

        # Generate boustrophedon (back-and-forth) pattern
        path_points = []
        num_rows = int(field_height / (self.working_width_m - self.overlap_m))

        for row in range(num_rows):
            # Calculate row position
            row_y = min_y + row * (self.working_width_m - self.overlap_m)

            # Alternate direction for each row
            if row % 2 == 0:
                # Left to right
                x_start = min_x
                x_end = max_x
                x_step = self.working_width_m / 10  # 10 points per row
            else:
                # Right to left
                x_start = max_x
                x_end = min_x
                x_step = -self.working_width_m / 10

            # Generate points for this row
            x = x_start
            while (x_step > 0 and x <= x_end) or (x_step < 0 and x >= x_end):
                path_points.append({
                    "x": x,
                    "y": row_y,
                    "z": 0.0,
                    "yaw": 0.0 if row % 2 == 0 else math.pi,
                })
                x += x_step

            # Add turn point at end of row
            turn_y = row_y + (self.working_width_m - self.overlap_m) / 2
            if turn_y < max_y:
                path_points.append({
                    "x": x_end,
                    "y": turn_y,
                    "z": 0.0,
                    "yaw": math.pi / 2 if row % 2 == 0 else -math.pi / 2,
                })

        return path_points

    def publish_global_plan(self, path: list[dict]) -> None:
        """Publish global plan as Path message."""
        header = Header()
        header.frame_id = self.map_frame
        header.stamp = self.get_clock().now().to_msg()

        path_msg = Path()
        path_msg.header = header

        for point in path[: self.max_path_length]:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = point["x"]
            pose.pose.position.y = point["y"]
            pose.pose.position.z = point.get("z", 0.0)
            pose.pose.orientation.w = 1.0  # Simplified orientation

            path_msg.poses.append(pose)

        self.global_plan_pub.publish(path_msg)

    def publish_target(self, point: dict) -> None:
        """Publish target pose."""
        header = Header()
        header.frame_id = self.map_frame
        header.stamp = self.get_clock().now().to_msg()

        target = PoseStamped()
        target.header = header
        target.pose.position.x = point["x"]
        target.pose.position.y = point["y"]
        target.pose.position.z = point.get("z", 0.0)
        target.pose.orientation.w = 1.0  # Simplified orientation

        self.target_pub.publish(target)


def main(args: Optional[list] = None) -> None:
    """Main function to run the node."""
    rclpy.init(args=args)

    node = CoveragePlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()