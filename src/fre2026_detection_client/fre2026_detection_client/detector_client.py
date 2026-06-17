#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from std_msgs.msg import String

from ros2_detection_interfaces.msg import DetectionArray
from ros2_detection_interfaces.srv import Init, Release, Start, Stop


@dataclass
class DetectorInitConfig:
    model_path: str
    classes: Sequence[str] = field(default_factory=list)
    confidence: float = 0.1
    use_realsense_ros_wrapper: bool = False
    model_type: str = "yolo"
    use_decimation: bool = False
    use_spatial: bool = False
    use_temporal: bool = True
    use_hole_filling: bool = True
    use_mask_filter: bool = True
    color_resolution_width: int = 640
    color_resolution_height: int = 480
    fps: int = 30
    rcnn_class_names: Sequence[str] = field(default_factory=list)


class DetectorClient:
    """Small adapter around the ros2_detection service/message API."""

    def __init__(
        self,
        node: Node,
        namespace: str = "/detector",
        callback_group: Optional[CallbackGroup] = None,
        on_results: Optional[Callable[[DetectionArray], None]] = None,
    ) -> None:
        self.node = node
        self.namespace = namespace.rstrip("/")
        self.on_results = on_results

        self.latest_status: Optional[str] = None
        self.latest_model_info: Optional[str] = None
        self.latest_results: Optional[DetectionArray] = None

        self.init_client = node.create_client(
            Init,
            self._topic("init"),
            callback_group=callback_group,
        )
        self.start_client = node.create_client(
            Start,
            self._topic("start"),
            callback_group=callback_group,
        )
        self.stop_client = node.create_client(
            Stop,
            self._topic("stop"),
            callback_group=callback_group,
        )
        self.release_client = node.create_client(
            Release,
            self._topic("release"),
            callback_group=callback_group,
        )

        self.status_sub = node.create_subscription(
            String,
            self._topic("status"),
            self._on_status,
            10,
            callback_group=callback_group,
        )
        self.model_info_sub = node.create_subscription(
            String,
            self._topic("model_info"),
            self._on_model_info,
            10,
            callback_group=callback_group,
        )
        self.results_sub = node.create_subscription(
            DetectionArray,
            self._topic("results"),
            self._on_results,
            2,
            callback_group=callback_group,
        )

    def wait_for_services(self, timeout_sec: float = 10.0) -> bool:
        start = time.monotonic()
        clients = [
            self.init_client,
            self.start_client,
            self.stop_client,
            self.release_client,
        ]

        while time.monotonic() - start < timeout_sec:
            if all(client.service_is_ready() for client in clients):
                return True
            for client in clients:
                client.wait_for_service(timeout_sec=0.2)

        return all(client.service_is_ready() for client in clients)

    def init(
        self,
        model_path: str,
        classes: Optional[Sequence[str]] = None,
        confidence: float = 0.1,
        use_realsense_ros_wrapper: bool = False,
    ) -> Future:
        config = DetectorInitConfig(
            model_path=model_path,
            classes=list(classes or []),
            confidence=confidence,
            use_realsense_ros_wrapper=use_realsense_ros_wrapper,
        )
        return self.init_from_config(config)

    def init_from_config(self, config: DetectorInitConfig) -> Future:
        req = Init.Request()
        req.model_type = config.model_type
        req.model_path = config.model_path
        req.classes = list(config.classes)
        req.use_realsense_ros_wrapper = config.use_realsense_ros_wrapper
        req.confidence = float(config.confidence)
        req.use_decimation = config.use_decimation
        req.use_spatial = config.use_spatial
        req.use_temporal = config.use_temporal
        req.use_hole_filling = config.use_hole_filling
        req.use_mask_filter = config.use_mask_filter
        req.color_resolution_width = int(config.color_resolution_width)
        req.color_resolution_height = int(config.color_resolution_height)
        req.fps = int(config.fps)
        req.rcnn_class_names = list(config.rcnn_class_names)

        return self._call(self.init_client, req, "init")

    def start(self) -> Future:
        return self._call(self.start_client, Start.Request(), "start")

    def stop(self) -> Future:
        return self._call(self.stop_client, Stop.Request(), "stop")

    def release(self) -> Future:
        return self._call(self.release_client, Release.Request(), "release")

    def clear_results(self) -> None:
        self.latest_results = None

    def has_detections(self) -> bool:
        return self.latest_results is not None and bool(self.latest_results.detections)

    def detection_count(self) -> int:
        if self.latest_results is None:
            return 0
        return len(self.latest_results.detections)

    def _topic(self, name: str) -> str:
        return f"{self.namespace}/{name}"

    def _call(self, client, request, action_name: str) -> Future:
        future = client.call_async(request)
        future.add_done_callback(lambda done: self._log_response(done, action_name))
        return future

    def _log_response(self, future: Future, action_name: str) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.node.get_logger().error(f"Detector service '{action_name}' failed: {exc}")
            return

        if response is None:
            self.node.get_logger().error(f"Detector service '{action_name}' returned no response")
            return

        success = bool(getattr(response, "success", False))
        message = str(getattr(response, "message", ""))
        if success:
            self.node.get_logger().info(f"Detector service '{action_name}' OK: {message}")
        else:
            self.node.get_logger().error(f"Detector service '{action_name}' FAILED: {message}")

        model_info = getattr(response, "model_info", "")
        if model_info:
            self.node.get_logger().info(f"Detector model info: {model_info}")

    def _on_status(self, msg: String) -> None:
        self.latest_status = msg.data
        self.node.get_logger().debug(f"Detector status: {msg.data}")

    def _on_model_info(self, msg: String) -> None:
        self.latest_model_info = msg.data
        self.node.get_logger().info(f"Detector model info: {msg.data}")

    def _on_results(self, msg: DetectionArray) -> None:
        self.latest_results = msg
        self.node.get_logger().debug(f"Detector results: {len(msg.detections)} detections")
        if self.on_results is not None:
            self.on_results(msg)
