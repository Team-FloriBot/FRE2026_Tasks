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

from maize_navigation.maize_navigation import (  # noqa: E402
    LaserRowFollower,
    MaizeNavigator,
    NavigatorParams,
    PatternStep,
    Pose2D,
    RowMarchModel,
    RowMarchResult,
)
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


def test_narrow_roi_ignores_implausible_neighbor_structure():
    follower = LaserRowFollower(NavigatorParams())
    scan = make_scan([(0.0, 0.28), (0.0, -0.20)])

    result = follower.process_scan(scan, 0.0, np.array([1.4, 0.0]))

    assert result.valid
    assert result.left_line.valid
    assert not result.right_line.valid
    assert result.weight <= follower.p.laser_max_weight_one_side


def test_angle_and_center_offset_are_rejected():
    follower = LaserRowFollower(NavigatorParams())
    angled_scan = make_scan([(0.7, 0.375), (0.7, -0.375)])
    offset_scan = make_scan([(0.0, 0.80), (0.0, 0.05)])

    angle_result = follower.process_scan(angled_scan, 0.0, np.array([1.4, 0.0]))
    offset_result = follower.process_scan(offset_scan, 0.0, np.array([1.4, 0.0]))

    assert not angle_result.valid
    assert angle_result.reason == "angle differs from map"
    assert not offset_result.valid
    assert offset_result.reason == "no valid side line"


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


def test_rois_rotate_with_map_direction_and_clamp_center_offset():
    follower = LaserRowFollower(NavigatorParams())

    centers, direction = follower.build_rois(0.5, np.array([1.4, 1.2]))

    assert np.allclose(direction, np.array([1.0, 0.5]) / math.sqrt(1.25))
    perp = np.array([-direction[1], direction[0]])
    lane_center = 0.5 * (centers[0] + centers[1])
    assert abs(float((lane_center - (follower.p.laser_roi_x_min + 0.5 * follower.p.laser_roi_length) * direction)[1])) <= 0.250001
    assert abs(float((centers[0] - centers[1]) @ perp) - follower.p.expected_row_width) < 1e-6


def test_rois_fall_back_to_robot_heading_without_map_direction():
    follower = LaserRowFollower(NavigatorParams())

    _, direction = follower.build_rois(None, np.array([1.4, 0.0]))

    assert np.allclose(direction, np.array([1.0, 0.0]))


def bare_navigator():
    navigator = MaizeNavigator.__new__(MaizeNavigator)
    navigator.p = NavigatorParams()
    navigator.robot_pose = Pose2D(3.0, 0.0, 0.0)
    navigator.initial_forward_direction = np.array([1.0, 0.0])
    navigator.row_end_directions_by_side = {"forward": [], "backward": []}
    navigator.entrance_route = np.empty((0, 2))
    navigator.entrance_route_progress_index = 0
    navigator.entrance_route_projection = None
    navigator.entrance_route_target = None
    navigator.entrance_route_remaining_distance = 0.0
    navigator.entrance_route_provisional = False
    navigator.entrance_active_step = None
    navigator.entrance_target = None
    navigator.entrance_hist_peaks = []
    return navigator


def test_freeze_prefix_keeps_passed_points_static():
    navigator = bare_navigator()
    model = RowMarchModel("left", np.array([0.0, 0.4]), np.array([0.3, 0.4]), np.array([1.0, 0.0]), 1)
    model.result = RowMarchResult(
        points=np.array([[0.0, 0.4], [0.3, 0.4], [1.0, 0.4], [2.0, 0.4], [3.0, 0.4], [3.3, 0.4]]),
        point_directions=np.tile(np.array([1.0, 0.0]), (6, 1)),
    )

    navigator.update_frozen_prefix(model)

    assert np.allclose(model.frozen_points, np.array([[0.0, 0.4], [0.3, 0.4], [1.0, 0.4], [2.0, 0.4], [3.0, 0.4]]))
    assert np.allclose(model.frozen_directions, np.tile(np.array([1.0, 0.0]), (5, 1)))


def test_frozen_prefix_continues_with_saved_ransac_direction():
    navigator = bare_navigator()
    model = RowMarchModel("left", np.array([0.0, 0.4]), np.array([0.3, 0.4]), np.array([1.0, 0.0]), 1)
    model.frozen_points = np.array([[0.0, 0.4], [0.3, 0.7]])
    model.frozen_directions = np.array([[1.0, 0.0], [1.0, 0.0]])

    result = navigator.march_row(model, np.empty((0, 2)))

    assert np.allclose(result.current_line_points[2], np.array([0.6, 0.7]))


def test_midline_does_not_modify_independent_row_results():
    navigator = bare_navigator()
    left = RowMarchResult(points=np.array([[0.0, 0.4], [0.3, 0.4], [0.6, 1.5]]), frozen_count=2)
    right = RowMarchResult(points=np.array([[0.0, -0.35], [0.3, -0.35], [0.6, -0.35]]), frozen_count=2)
    left_before = np.array(left.points, copy=True)
    right_before = np.array(right.points, copy=True)

    midline = navigator.build_midline(left.points, right.points)

    assert np.allclose(left.points, left_before)
    assert np.allclose(right.points, right_before)
    assert np.allclose(midline[-1], np.array([0.6, 0.575]))


def test_end_direction_can_still_average_independent_rows():
    navigator = bare_navigator()

    direction = navigator.mean_direction(np.array([1.0, 0.1]), np.array([1.0, -0.1]))

    assert np.allclose(direction, np.array([1.0, 0.0]))


def test_histogram_peak_uses_actual_shifted_row_end():
    navigator = bare_navigator()
    points = np.array([[x, 0.4] for x in np.linspace(0.0, 1.0, 12)] + [[x, -0.4] for x in np.linspace(0.0, 2.0, 12)])

    end = navigator.actual_row_end_from_peak(points, np.array([0.0, 0.0]), np.array([1.0, 0.0]), -0.4)

    assert end[0] > 1.8
    assert abs(end[1] + 0.4) < 1e-6


def test_row_end_direction_average_is_kept_per_field_side():
    navigator = bare_navigator()
    navigator.row_end_directions_by_side["forward"] = [np.array([1.0, 0.0]), np.array([0.98, 0.20])]
    navigator.row_end_directions_by_side["backward"] = [np.array([-1.0, 0.0])]

    averaged = navigator.average_row_end_direction_for_side(np.array([1.0, 0.0]))

    assert averaged[0] > 0.99
    assert 0.08 < averaged[1] < 0.12


def test_route_projection_progress_never_moves_backwards():
    navigator = bare_navigator()
    route = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    _, first_idx, _ = navigator.project_onto_route_forward(route, np.array([2.2, 0.2]), 0)
    _, second_idx, _ = navigator.project_onto_route_forward(route, np.array([0.8, 0.1]), first_idx)

    assert first_idx == 2
    assert second_idx >= first_idx


def test_rounded_route_replaces_sharp_corner_with_curve():
    navigator = bare_navigator()
    support = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    route = navigator.build_rounded_route(support)

    assert len(route) > 3
    assert not any(np.allclose(point, np.array([1.0, 0.0])) for point in route)
    assert np.allclose(route[0], support[0])
    assert np.allclose(route[-1], support[-1])


def test_entry_line_requires_crossing_with_alignment():
    navigator = bare_navigator()
    navigator.entrance_target = np.array([1.0, 0.0])
    direction = np.array([1.0, 0.0])

    navigator.robot_pose = Pose2D(0.95, 0.0, 0.0)
    assert not navigator.entry_line_reached(navigator.entrance_target, direction)
    navigator.robot_pose = Pose2D(1.05, 0.10, 0.0)
    assert navigator.entry_line_reached(navigator.entrance_target, direction)
    navigator.robot_pose = Pose2D(1.05, 0.40, 0.0)
    assert not navigator.entry_line_reached(navigator.entrance_target, direction)


def test_route_target_uses_extension_after_missed_entry_point():
    navigator = bare_navigator()
    navigator.entrance_route = np.array([[0.0, 0.0], [1.0, 0.0], [1.8, 0.0]])
    navigator.robot_pose = Pose2D(1.10, 0.15, 0.0)

    target = navigator.update_entrance_route_target()

    assert target is not None
    assert target[0] > 1.10


def test_route_target_stops_when_deviation_is_too_large():
    navigator = bare_navigator()
    navigator.get_logger = lambda: type("Logger", (), {"warn": lambda *args, **kwargs: None})()
    navigator.entrance_route = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    navigator.robot_pose = Pose2D(0.5, 2.0, 0.0)

    target = navigator.update_entrance_route_target()

    assert target is None


def test_provisional_turn_route_exits_then_moves_to_expected_pattern_offset():
    navigator = bare_navigator()
    navigator.get_logger = lambda: type("Logger", (), {"info": lambda *args, **kwargs: None})()
    navigator.robot_pose = Pose2D(0.0, 0.0, 0.0)
    navigator.plant_row_end_point = np.array([0.0, 0.0])
    navigator.row_end_direction = np.array([1.0, 0.0])
    navigator.row_exit_goal = np.array([0.3, 0.0])
    navigator.pattern_steps = [PatternStep(2, "L")]
    navigator.pattern_index = 0

    assert navigator.ensure_provisional_entrance_route()
    assert navigator.entrance_target is None
    assert navigator.entrance_route_provisional
    assert np.allclose(navigator.entrance_waypoints[0], np.array([0.3, 0.0]))
    assert np.allclose(navigator.entrance_route[-1], np.array([0.3, 1.5]))


def test_turn_waypoints_follow_outermost_peak_on_traversed_route():
    navigator = bare_navigator()
    navigator.robot_pose = Pose2D(0.0, 0.0, 0.0)
    navigator.plant_row_end_point = np.array([0.0, 0.0])
    navigator.row_end_direction = np.array([1.0, 0.0])
    navigator.entrance_active_step = PatternStep(2, "L")
    navigator.entrance_hist_peaks = [
        type("Peak", (), {"lateral": 0.0, "point": np.array([0.4, 0.0])})(),
        type("Peak", (), {"lateral": 0.75, "point": np.array([0.9, 0.75])})(),
        type("Peak", (), {"lateral": 1.5, "point": np.array([0.6, 1.5])})(),
        type("Peak", (), {"lateral": -2.0, "point": np.array([2.0, -2.0])})(),
    ]

    navigator.rebuild_entrance_route(None)

    assert np.allclose(navigator.entrance_waypoints[0], np.array([1.2, 0.0]))
    assert np.allclose(navigator.entrance_waypoints[1], np.array([1.2, 1.5]))
