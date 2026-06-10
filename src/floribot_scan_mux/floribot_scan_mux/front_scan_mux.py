from __future__ import annotations

from functools import partial
from typing import Dict, NamedTuple

import rclpy
from fre2026_tasks_interfaces.srv import SetScanProfile
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


PROFILE_NAMES = (
    "rs_crop_scan",
    "rs_nonground_scan",
    "rs_nonground_scan_torsten",
    "sick_front",
)


class FrontScanMux(Node):
    def __init__(self) -> None:
        super().__init__("front_scan_mux")

        self.declare_parameter("active_profile", "sick_front")
        self.declare_parameter("front_output_topic", "/laser_scan_mux/front_scan")
        self.declare_parameter("rear_output_topic", "/laser_scan_mux/rear_scan")
        self.declare_parameter("output_topic", "")
        self.declare_parameter("input_qos_reliability", "best_effort")
        self.declare_parameter("output_qos_reliability", "reliable")

        for profile in PROFILE_NAMES:
            self.declare_parameter(f"profiles.{profile}.topic", f"/{profile}")
            self.declare_parameter(f"profiles.{profile}.front_topic", "")
            self.declare_parameter(f"profiles.{profile}.rear_topic", "")

        self.active_profile = (
            self.get_parameter("active_profile").get_parameter_value().string_value
        )
        self.front_output_topic = self._get_output_topic(
            "front_output_topic",
            fallback_parameter_name="output_topic",
        )
        self.rear_output_topic = self._get_output_topic("rear_output_topic")
        self.profile_topics = self._load_profile_topics()

        if self.active_profile not in self.profile_topics:
            fallback_profile = PROFILE_NAMES[-1]
            self.get_logger().warn(
                f"Unknown active_profile '{self.active_profile}', using '{fallback_profile}'."
            )
            self.active_profile = fallback_profile

        input_qos = self._make_qos("input_qos_reliability")
        output_qos = self._make_qos("output_qos_reliability")

        self.front_publisher = self.create_publisher(
            LaserScan,
            self.front_output_topic,
            output_qos,
        )
        self.rear_publisher = self.create_publisher(
            LaserScan,
            self.rear_output_topic,
            output_qos,
        )
        self.front_subscriptions_by_profile = {
            profile: self.create_subscription(
                LaserScan,
                topics.front,
                partial(self._scan_cb, profile=profile, side="front"),
                input_qos,
            )
            for profile, topics in self.profile_topics.items()
        }
        self.rear_subscriptions_by_profile = {
            profile: self.create_subscription(
                LaserScan,
                topics.rear,
                partial(self._scan_cb, profile=profile, side="rear"),
                input_qos,
            )
            for profile, topics in self.profile_topics.items()
        }

        self.set_profile_service = self.create_service(
            SetScanProfile,
            "set_profile",
            self._set_profile_cb,
        )

        self.get_logger().info(
            f"Scan mux publishing '{self.active_profile}' to "
            f"'{self.front_output_topic}' and '{self.rear_output_topic}'."
        )

    def _get_output_topic(
        self,
        parameter_name: str,
        *,
        fallback_parameter_name: str | None = None,
    ) -> str:
        topic = self.get_parameter(parameter_name).get_parameter_value().string_value
        if topic:
            return topic
        if fallback_parameter_name is None:
            return topic
        return (
            self.get_parameter(fallback_parameter_name)
            .get_parameter_value()
            .string_value
        )

    def _load_profile_topics(self) -> Dict[str, "ProfileTopics"]:
        topics = {}
        for profile in PROFILE_NAMES:
            legacy_topic = (
                self.get_parameter(f"profiles.{profile}.topic")
                .get_parameter_value()
                .string_value
            )
            front_topic = (
                self.get_parameter(f"profiles.{profile}.front_topic")
                .get_parameter_value()
                .string_value
            )
            rear_topic = (
                self.get_parameter(f"profiles.{profile}.rear_topic")
                .get_parameter_value()
                .string_value
            )
            topics[profile] = ProfileTopics(
                front=front_topic or legacy_topic,
                rear=rear_topic or legacy_topic,
            )
        return topics

    def _make_qos(self, parameter_name: str) -> QoSProfile:
        reliability = (
            self.get_parameter(parameter_name).get_parameter_value().string_value
        )
        qos = QoSProfile(depth=10)
        if reliability == "best_effort":
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        elif reliability == "reliable":
            qos.reliability = ReliabilityPolicy.RELIABLE
        else:
            self.get_logger().warn(
                f"Unknown {parameter_name} '{reliability}', using reliable QoS."
            )
            qos.reliability = ReliabilityPolicy.RELIABLE
        return qos

    def _scan_cb(self, msg: LaserScan, *, profile: str, side: str) -> None:
        if profile != self.active_profile:
            return
        if side == "front":
            self.front_publisher.publish(msg)
        elif side == "rear":
            self.rear_publisher.publish(msg)
        else:
            self.get_logger().warn(f"Unknown mux side '{side}', dropping scan.")

    def _set_profile_cb(
        self,
        request: SetScanProfile.Request,
        response: SetScanProfile.Response,
    ) -> SetScanProfile.Response:
        requested_profile = request.profile.strip()
        if requested_profile not in self.profile_topics:
            accepted = ", ".join(self.profile_topics.keys())
            response.success = False
            response.message = (
                f"Unknown profile '{requested_profile}'. Accepted profiles: {accepted}."
            )
            response.active_profile = self.active_profile
            return response

        self.active_profile = requested_profile
        self.set_parameters(
            [Parameter("active_profile", value=self.active_profile)]
        )

        response.success = True
        response.message = f"Active scan profile set to '{self.active_profile}'."
        response.active_profile = self.active_profile
        self.get_logger().info(response.message)
        return response


class ProfileTopics(NamedTuple):
    front: str
    rear: str


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrontScanMux()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
