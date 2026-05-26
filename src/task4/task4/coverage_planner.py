#!/usr/bin/env python3
"""
Coverage Planner Node with Fields2Cover Integration

This node implements coverage path planning for autonomous field operations.
It uses the Fields2Cover library for optimal coverage patterns.

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
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header


class CoveragePlanner(Node):
    """Node for coverage path planning with Fields2Cover integration."""

    def __init__(self) -> None:
        super().__init__("coverage_planner")

        # Import Fields2Cover - mandatory, node will crash if not available
        try:
            from f2c import (
                F2CField,
                F2CFieldGenerator,
                F2CPathGenerator,
                F2CPathOptimizer,
                F2CPoint,
                F2CLinearRing,
                F2CPolygon,
                F2CCells,
            )

            self.f2c = True
            self.f2c_field = None
            self.f2c_path_generator = None
            self.get_logger().info("Fields2Cover imported successfully")
        except ImportError as e:
            self.get_logger().error(f"Fields2Cover import failed: {e}")
            raise RuntimeError("Fields2Cover is required but not available. Please install the f2c package.") from e

        # Declare and get parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("field_boundary_posts", []),
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
        self.field_boundary_posts_raw = self.get_parameter("field_boundary_posts").value
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

        # Convert raw field boundary posts to list of points
        self.field_boundary: list[dict] = self._convert_field_boundary_posts(
            self.field_boundary_posts_raw
        )

        # Log field boundary information
        if self.field_boundary:
            self.get_logger().info(
                f"Field boundary loaded: {len(self.field_boundary)} posts "
                f"({self.field_boundary_posts_raw})"
            )
        else:
            self.get_logger().warn(
                "Field boundary is empty. Please set the 'field_boundary_posts' parameter."
            )

        # Create subscribers
        self.field_boundary_sub = self.create_subscription(
            MarkerArray, self.field_boundary_topic, self.field_boundary_callback, 10
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

    def _convert_field_boundary_posts(self, raw_posts: list[float]) -> list[dict]:
        """
        Convert flat list of floats to list of point dictionaries.
        
        Args:
            raw_posts: Flat list of floats [x1, y1, x2, y2, x3, y3, ...]
            
        Returns:
            List of point dictionaries with 'x', 'y', 'z' keys
            
        Raises:
            RuntimeError: If the input list has invalid length
        """
        if not raw_posts:
            self.get_logger().warn("Field boundary posts parameter is empty")
            return []

        # Validate that the list has even length (pairs of x, y)
        if len(raw_posts) % 2 != 0:
            raise RuntimeError(
                f"Field boundary posts must have even length (pairs of x,y coordinates), "
                f"got {len(raw_posts)} values"
            )

        # Validate minimum number of points (at least 3 for a polygon)
        num_points = len(raw_posts) // 2
        if num_points < 3:
            raise RuntimeError(
                f"Field boundary requires at least 3 points (6 coordinates), "
                f"got {num_points} points"
            )

        # Convert to list of dictionaries
        points = []
        for i in range(num_points):
            points.append({
                "x": raw_posts[i * 2],
                "y": raw_posts[i * 2 + 1],
                "z": 0.0,
            })

        return points

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

    def update_plan(self) -> None:
        """Update the coverage plan periodically."""
        if len(self.field_boundary) < 3:
            self.get_logger().debug("Not enough field boundary points")
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
        """Generate coverage path using Fields2Cover."""
        return self.generate_coverage_path_f2c()

    def generate_coverage_path_f2c(self) -> Optional[list[dict]]:
        """Generate coverage path using Fields2Cover with actual polygon geometry."""
        if not self.field_boundary or len(self.field_boundary) < 3:
            self.get_logger().error("Cannot generate path: insufficient field boundary points")
            return None

        try:
            # Create F2C points from boundary posts
            f2c_points = []
            for point in self.field_boundary:
                f2c_points.append(F2CPoint(point["x"], point["y"]))

            # Create a closed linear ring (first point must be repeated at the end)
            # F2CLinearRing expects a closed loop
            linear_ring = F2CLinearRing()
            for point in f2c_points:
                linear_ring.add(point)
            # Close the ring by adding the first point again
            if len(f2c_points) > 0:
                linear_ring.add(f2c_points[0])

            # Create polygon from the linear ring
            polygon = F2CPolygon()
            polygon.add(linear_ring)

            # Create F2CCells container
            cells = F2CCells()
            cells.add(polygon)

            # Create field with the polygon
            field = F2CField()
            field.setField(cells)

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

            self.get_logger().info(f"Generated coverage path with {len(path_points)} points")
            return path_points

        except Exception as e:
            self.get_logger().error(f"Fields2Cover path generation failed: {e}")
            return None

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