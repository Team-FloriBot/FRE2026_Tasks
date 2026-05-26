#!/usr/bin/env python3
"""
Marker_Detector_CV Node

This node detects circular markers on the ground using OpenCV.
It uses HSV color filtering and contour analysis (Hough circle transform)
to detect 5cm diameter markers and calculates their 3D position
relative to the camera frame.

Author: FRE2026 Team
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.logging import LoggingSeverity

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header


class MarkerDetector(Node):
    """Node for detecting circular markers using OpenCV."""

    def __init__(self) -> None:
        super().__init__("marker_detector")

        # Declare and get parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("camera_topic", "/sensors/camera/image_raw"),
                ("camera_info_topic", "/sensors/camera/camera_info"),
                ("marker_diameter_m", 0.05),
                ("marker_color_hsv_min", [30, 100, 100]),
                ("marker_color_hsv_max", [90, 255, 255]),
                ("camera_height_m", 0.5),
                ("camera_pitch_rad", 0.52),
                ("min_contour_area", 10.0),
                ("max_contour_area", 500.0),
                ("hough_circle_param1", 50.0),
                ("hough_circle_param2", 30.0),
                ("hough_circle_min_radius", 5),
                ("hough_circle_max_radius", 25),
                ("publish_markers", True),
                ("marker_frame", "camera_frame"),
                ("output_topic", "/detected_marker"),
                ("debug_visualization", False),
                ("debug_topic", "/marker_detector/debug"),
            ],
        )

        # Get parameters
        self.camera_topic = self.get_parameter("camera_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.marker_diameter_m = self.get_parameter("marker_diameter_m").value
        self.marker_color_hsv_min = np.array(
            self.get_parameter("marker_color_hsv_min").value, dtype=np.uint8
        )
        self.marker_color_hsv_max = np.array(
            self.get_parameter("marker_color_hsv_max").value, dtype=np.uint8
        )
        self.camera_height_m = self.get_parameter("camera_height_m").value
        self.camera_pitch_rad = self.get_parameter("camera_pitch_rad").value
        self.min_contour_area = self.get_parameter("min_contour_area").value
        self.max_contour_area = self.get_parameter("max_contour_area").value
        self.hough_circle_param1 = self.get_parameter("hough_circle_param1").value
        self.hough_circle_param2 = self.get_parameter("hough_circle_param2").value
        self.hough_circle_min_radius = self.get_parameter("hough_circle_min_radius").value
        self.hough_circle_max_radius = self.get_parameter("hough_circle_max_radius").value
        self.publish_markers = self.get_parameter("publish_markers").value
        self.marker_frame = self.get_parameter("marker_frame").value
        self.output_topic = self.get_parameter("output_topic").value
        self.debug_visualization = self.get_parameter("debug_visualization").value
        self.debug_topic = self.get_parameter("debug_topic").value

        # Initialize camera info
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.image_width: int = 0
        self.image_height: int = 0

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )

        # Create publishers
        self.marker_pub = self.create_publisher(PointStamped, self.output_topic, 10)

        if self.debug_visualization:
            self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)

        self.get_logger().info("MarkerDetector node initialized")
        self.get_logger().info(f"Camera topic: {self.camera_topic}")
        self.get_logger().info(f"Marker diameter: {self.marker_diameter_m} m")
        self.get_logger().info(f"Camera height: {self.camera_height_m} m")
        self.get_logger().info(f"Camera pitch: {self.camera_pitch_rad} rad")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Callback for camera info messages."""
        self.image_width = msg.width
        self.image_height = msg.height

        # Camera matrix (3x3)
        self.camera_matrix = np.array(msg.k, dtype=np.float32).reshape(3, 3)

        # Distortion coefficients
        self.dist_coeffs = np.array(msg.d, dtype=np.float32)

        self.get_logger().info("Camera info received")
        self.get_logger().info(f"Camera matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"Distortion coefficients: {self.dist_coeffs}")

        # Unsubscribe from camera info topic after receiving first message
        self.destroy_subscription(self.camera_info_sub)

    def image_callback(self, msg: Image) -> None:
        """Callback for image messages."""
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Detect markers
        markers = self.detect_markers(cv_image)

        # Publish markers
        for marker in markers:
            self.publish_marker(marker, msg.header)

        # Publish debug visualization if enabled
        if self.debug_visualization:
            debug_image = self.draw_markers(cv_image.copy(), markers)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

    def detect_markers(self, image: np.ndarray) -> list[dict]:
        """Detect circular markers in the image."""
        markers = []

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for marker color
        mask = cv2.inRange(hsv_image, self.marker_color_hsv_min, self.marker_color_hsv_max)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            # Get minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Filter by radius (based on marker diameter and distance)
            # Expected radius depends on distance to marker
            # For a 5cm marker at distance d: radius = (d * tan(fov/2)) / (image_width/2)
            # This is a rough estimate - adjust based on actual camera calibration

            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # Filter by circularity (circle has circularity = 1)
            if circularity < 0.7:
                continue

            # Calculate marker position in 3D
            marker_3d = self.calculate_marker_3d(center, radius)

            if marker_3d is not None:
                markers.append({
                    "center": center,
                    "radius": radius,
                    "area": area,
                    "circularity": circularity,
                    "position": marker_3d,
                })

        return markers

    def calculate_marker_3d(
        self, center: tuple[int, int], radius: int
    ) -> Optional[dict]:
        """Calculate 3D position of marker relative to camera."""
        if self.camera_matrix is None:
            return None

        # Get camera intrinsic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Calculate pixel size at 1 meter (approximation)
        # This assumes small angles and camera aligned with ground
        pixel_size_at_1m = (fx + fy) / 2.0

        # Calculate distance to marker based on marker size in pixels
        # d = (actual_size * focal_length) / pixel_size
        # For a 5cm marker:
        marker_size_pixels = 2 * radius
        if marker_size_pixels < 1:
            return None

        # Estimate distance from camera to marker
        # d = (marker_diameter_m * fx) / marker_size_pixels
        distance_m = (self.marker_diameter_m * fx) / marker_size_pixels

        # Apply camera pitch correction
        # The camera is pitched down, so the actual distance is:
        # d_actual = d_estimated * cos(pitch)
        distance_m = distance_m * math.cos(self.camera_pitch_rad)

        # Calculate 3D position
        # x = (u - cx) * distance_m / fx
        # y = (v - cy) * distance_m / fy
        # z = distance_m * sin(pitch) + camera_height
        u, v = center
        x = (u - cx) * distance_m / fx
        y = (v - cy) * distance_m / fy
        z = distance_m * math.sin(self.camera_pitch_rad) + self.camera_height_m

        return {"x": x, "y": y, "z": z}

    def publish_marker(self, marker: dict, header: Header) -> None:
        """Publish marker as PointStamped message."""
        point = PointStamped()
        point.header = header
        point.header.frame_id = self.marker_frame
        point.point.x = marker["position"]["x"]
        point.point.y = marker["position"]["y"]
        point.point.z = marker["position"]["z"]

        self.marker_pub.publish(point)

    def draw_markers(
        self, image: np.ndarray, markers: list[dict]
    ) -> np.ndarray:
        """Draw markers on image for debugging."""
        for marker in markers:
            center = marker["center"]
            radius = marker["radius"]

            # Draw circle
            cv2.circle(image, center, radius, (0, 255, 0), 2)

            # Draw center
            cv2.circle(image, center, 2, (0, 0, 255), -1)

            # Draw position text
            pos = marker["position"]
            text = f"({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})"
            cv2.putText(
                image,
                text,
                (center[0] - 50, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return image


def main(args: Optional[list] = None) -> None:
    """Main function to run the node."""
    rclpy.init(args=args)

    node = MarkerDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()