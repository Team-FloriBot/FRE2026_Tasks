import math
import sys
import types

import numpy as np


def install_ros_import_stubs():
    def module(name):
        stub = types.ModuleType(name)
        sys.modules[name] = stub
        return stub

    rclpy = module("rclpy")
    rclpy_node = module("rclpy.node")
    rclpy_node.Node = object
    rclpy.node = rclpy_node

    for package, names in (
        ("geometry_msgs", ("Point", "Twist")),
        ("nav_msgs", ("OccupancyGrid",)),
        ("sensor_msgs", ("LaserScan",)),
        ("std_msgs", ("Header",)),
        ("std_srvs", ("Trigger",)),
        ("visualization_msgs", ("Marker", "MarkerArray")),
    ):
        package_module = module(package)
        message_module = module(f"{package}.msg" if package != "std_srvs" else f"{package}.srv")
        for name in names:
            setattr(message_module, name, type(name, (), {}))
        if package == "std_srvs":
            package_module.srv = message_module
        else:
            package_module.msg = message_module

    tf2_ros = module("tf2_ros")
    tf2_ros.Buffer = object
    tf2_ros.TransformListener = object
    transformations = module("tf_transformations")
    transformations.euler_from_quaternion = lambda quaternion: (0.0, 0.0, 0.0)


install_ros_import_stubs()

from maize_navigation.maize_navigation import LaserRowFollower, NavigatorParams  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402


def make_scan(lines, outliers=()):
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = 2.0 * math.pi / 1440
    scan.range_min = 0.05
    scan.range_max = 20.0
    ranges = np.full(1440, np.inf, dtype=float)

    points = []
    for slope, intercept in lines:
        for x in np.linspace(0.25, 2.0, 80):
            points.append((x, slope * x + intercept))
    points.extend(outliers)

    for x, y in points:
        distance = math.hypot(x, y)
        angle = math.atan2(y, x)
        index = int(round((angle - scan.angle_min) / scan.angle_increment))
        if 0 <= index < len(ranges):
            ranges[index] = min(ranges[index], distance)
    scan.ranges = ranges.tolist()
    return scan


def process_repeatedly(follower, scan, count=12, map_slope=0.0, map_target=(1.4, 0.0)):
    result = None
    for _ in range(count):
        result = follower.process_scan(scan, map_slope, np.asarray(map_target, dtype=float))
    return result


def test_both_rows_produce_centered_high_weight_target():
    follower = LaserRowFollower(NavigatorParams())
    scan = make_scan([(0.0, 0.375), (0.0, -0.375)])

    result = process_repeatedly(follower, scan)

    assert result.valid
    assert result.left_line.valid
    assert result.right_line.valid
    assert result.weight > 0.70
    assert abs(result.target_base[1]) < 0.02


def test_single_row_uses_expected_width_with_reduced_weight():
    follower = LaserRowFollower(NavigatorParams())
    scan = make_scan([(0.0, 0.375)])

    result = process_repeatedly(follower, scan)

    assert result.valid
    assert result.left_line.valid
    assert not result.right_line.valid
    assert 0.0 < result.weight <= follower.p.laser_max_weight_one_side
    assert abs(result.target_base[1]) < 0.02


def test_ransac_ignores_outliers():
    follower = LaserRowFollower(NavigatorParams())
    outliers = [(0.4, 0.82), (0.8, 0.22), (1.3, -0.75), (1.8, 0.78)]
    scan = make_scan([(0.04, 0.33), (0.04, -0.42)], outliers)

    result = process_repeatedly(follower, scan, map_slope=0.04, map_target=(1.4, 0.01))

    assert result.valid
    assert abs(result.center_slope - 0.04) < 0.03
    assert abs(result.target_base[1] - 0.01) < 0.04


def test_implausible_width_is_rejected():
    follower = LaserRowFollower(NavigatorParams())
    scan = make_scan([(0.0, 0.28), (0.0, -0.20)])

    result = follower.process_scan(scan, 0.0, np.array([1.4, 0.0]))

    assert not result.valid
    assert result.reason == "invalid lane width"


def test_angle_and_center_offset_are_rejected():
    follower = LaserRowFollower(NavigatorParams())
    angled_scan = make_scan([(0.7, 0.375), (0.7, -0.375)])
    offset_scan = make_scan([(0.0, 0.80), (0.0, 0.05)])

    angle_result = follower.process_scan(angled_scan, 0.0, np.array([1.4, 0.0]))
    offset_result = follower.process_scan(offset_scan, 0.0, np.array([1.4, 0.0]))

    assert not angle_result.valid
    assert angle_result.reason == "angle differs from map"
    assert not offset_result.valid
    assert offset_result.reason == "center differs from map"


def test_missing_scan_has_zero_weight():
    follower = LaserRowFollower(NavigatorParams())

    result = follower.process_scan(None, 0.0, np.array([1.4, 0.0]))

    assert not result.valid
    assert result.weight == 0.0
    assert result.reason == "no scan"


def test_missing_scan_decays_previous_confidence():
    follower = LaserRowFollower(NavigatorParams())
    scan = make_scan([(0.0, 0.375), (0.0, -0.375)])
    previous = process_repeatedly(follower, scan)

    result = follower.process_scan(None, 0.0, np.array([1.4, 0.0]))

    assert result.confidence < previous.confidence
    assert result.weight == 0.0
