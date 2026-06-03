from __future__ import annotations

from functools import partial
from typing import Dict

import rclpy
from fre2026_task_interfaces.srv import SetScanProfile
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


PROFILE_NAMES = ("rs_crop_scan", "rs_nonground_scan", "sick_front")


class FrontScanMux(Node):
    def __init__(self) -> None:
        super().__init__("front_scan_mux")

        self.declare_parameter("active_profile", "sick_front")
        self.declare_parameter("output_topic", "/laser_scan_mux/front_scan")
        self.declare_parameter("input_qos_reliability", "best_effort")
        self.declare_parameter("output_qos_reliability", "reliable")

        for profile in PROFILE_NAMES:
            self.declare_parameter(f"profiles.{profile}.topic", f"/{profile}")

        self.active_profile = (
            self.get_parameter("active_profile").get_parameter_value().string_value
        )
        self.output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        self.profile_topics = self._load_profile_topics()

        if self.active_profile not in self.profile_topics:
            fallback_profile = PROFILE_NAMES[-1]
            self.get_logger().warn(
                f"Unknown active_profile '{self.active_profile}', using '{fallback_profile}'."
            )
            self.active_profile = fallback_profile

        input_qos = self._make_qos("input_qos_reliability")
        output_qos = self._make_qos("output_qos_reliability")

        self.publisher = self.create_publisher(LaserScan, self.output_topic, output_qos)
        self.subscriptions_by_profile = {
            profile: self.create_subscription(
                LaserScan,
                topic,
                partial(self._scan_cb, profile=profile),
                input_qos,
            )
            for profile, topic in self.profile_topics.items()
        }

        self.set_profile_service = self.create_service(
            SetScanProfile,
            "set_profile",
            self._set_profile_cb,
        )

        self.get_logger().info(
            f"Front scan mux publishing '{self.active_profile}' to '{self.output_topic}'."
        )

    def _load_profile_topics(self) -> Dict[str, str]:
        topics = {}
        for profile in PROFILE_NAMES:
            topic = (
                self.get_parameter(f"profiles.{profile}.topic")
                .get_parameter_value()
                .string_value
            )
            topics[profile] = topic
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

    def _scan_cb(self, msg: LaserScan, *, profile: str) -> None:
        if profile != self.active_profile:
            return
        self.publisher.publish(msg)

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
        response.message = f"Active front scan profile set to '{self.active_profile}'."
        response.active_profile = self.active_profile
        self.get_logger().info(response.message)
        return response


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
