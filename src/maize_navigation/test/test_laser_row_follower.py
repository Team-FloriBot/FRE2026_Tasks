import math
import sys
import types
from pathlib import Path

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
        ("fre2026_detection_interfaces", ("TrackedObject", "TrackedObjectArray")),
        ("fre2026_tasks_interfaces", ("SetNavigationPattern", "GetNavigationStatus")),
        ("maize_navigation_interfaces", ("StartNavigation",)),
        ("nav_msgs", ("OccupancyGrid",)),
        ("sensor_msgs", ("LaserScan",)),
        ("std_msgs", ("Bool", "Header", "String")),
        ("std_srvs", ("Trigger",)),
        ("visualization_msgs", ("Marker", "MarkerArray")),
    ):
        package_module = module(package)
        module_type = "srv" if package in (
            "fre2026_tasks_interfaces",
            "maize_navigation_interfaces",
            "std_srvs",
        ) else "msg"
        message_module = module(f"{package}.{module_type}")
        for name in names:
            setattr(message_module, name, type(name, (), {}))
        if package == "geometry_msgs":
            class Twist:
                def __init__(self):
                    self.linear = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
                    self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

            message_module.Twist = Twist
        if package == "std_msgs":
            class Bool:
                def __init__(self):
                    self.data = False

            message_module.Bool = Bool

            class String:
                def __init__(self):
                    self.data = ""

            message_module.String = String
        if package == "std_srvs":
            message_module.Trigger.Request = type("Request", (), {})
        if package == "fre2026_detection_interfaces":
            class TrackedObject:
                def __init__(self):
                    self.id = 0
                    self.label = ""
                    self.position = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

            class TrackedObjectArray:
                def __init__(self):
                    self.objects = []

            message_module.TrackedObject = TrackedObject
            message_module.TrackedObjectArray = TrackedObjectArray
        if package in ("fre2026_tasks_interfaces", "maize_navigation_interfaces", "std_srvs"):
            package_module.srv = message_module
        else:
            package_module.msg = message_module

    tf2_ros = module("tf2_ros")
    tf2_ros.Buffer = object
    tf2_ros.TransformListener = object
    transformations = module("tf_transformations")
    transformations.euler_from_quaternion = lambda quaternion: (0.0, 0.0, 0.0)

    detection_client = module("fre2026_detection_client")

    class DetectorClient:
        def __init__(self, *args, **kwargs):
            pass

    detection_client.DetectorClient = DetectorClient


install_ros_import_stubs()

from maize_navigation.maize_navigation import (  # noqa: E402
    DrivingProfile,
    EntrancePeak,
    LaserRowFollower,
    MaizeNavigator,
    MissionState,
    NavigatorParams,
    ObjectStop,
    PatternStep,
    Pose2D,
    RowMarchModel,
    RowMarchResult,
)
from fre2026_detection_interfaces.msg import TrackedObjectArray  # noqa: E402
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


def make_navigator_for_start_callback():
    navigator = MaizeNavigator.__new__(MaizeNavigator)
    navigator.p = NavigatorParams()
    navigator.pattern_steps = [PatternStep(1, "L")]
    navigator.laser_follower = types.SimpleNamespace(reset=lambda: None)
    navigator.driving_profiles = navigator.build_driving_profiles()
    navigator.current_carefulness = "high"
    navigator.paused_state = None
    navigator.published_cmds = []
    navigator.cmd_pub = types.SimpleNamespace(publish=lambda msg: navigator.published_cmds.append(msg))
    navigator.published_audio = []
    navigator.audio_pub = types.SimpleNamespace(publish=lambda msg: navigator.published_audio.append(msg.data))
    navigator.published_tracker_active = []
    navigator.tracker_active_pub = types.SimpleNamespace(
        publish=lambda msg: navigator.published_tracker_active.append(msg.data)
    )
    navigator.slam_reset_client = None
    navigator.tracker_reset_calls = 0
    navigator.tracker_reset_client = types.SimpleNamespace(
        service_is_ready=lambda: True,
        wait_for_service=lambda timeout_sec=0.0: True,
        call_async=lambda request: (
            setattr(navigator, "tracker_reset_calls", navigator.tracker_reset_calls + 1)
            or ImmediateFuture(success=True)
        ),
    )
    navigator.object_detection_enabled = False
    navigator.object_detection_model_path = ""
    navigator.object_detection_initialized = False
    navigator.object_detection_started = False
    navigator.latest_tracked_objects = None
    navigator.handled_tracked_object_ids = set()
    navigator.active_object_stop = None
    navigator.object_stop_hold_until_ns = None
    navigator.now_ns = 0
    navigator.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=navigator.now_ns)
    )
    navigator.get_logger = lambda: types.SimpleNamespace(
        warn=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    return navigator


class ImmediateFuture:
    def __init__(self, success=True, message=""):
        self.response = types.SimpleNamespace(success=success, message=message)

    def result(self):
        return self.response

    def add_done_callback(self, callback):
        callback(self)


def make_tracked_object(object_id, x, y, label="obj"):
    return types.SimpleNamespace(
        id=object_id,
        label=label,
        position=types.SimpleNamespace(x=x, y=y, z=0.0),
    )


def make_tracked_array(*objects):
    msg = TrackedObjectArray()
    msg.objects.extend(objects)
    return msg


def make_object_stop_navigator():
    navigator = make_navigator_for_start_callback()
    navigator.midline = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=float)
    left = RowMarchModel(
        "left",
        np.zeros(2),
        np.zeros(2),
        np.array([1.0, 0.0]),
        2,
    )
    left.result.points = np.array([[0.0, 0.75], [5.0, 0.75]], dtype=float)
    right = RowMarchModel(
        "right",
        np.zeros(2),
        np.zeros(2),
        np.array([1.0, 0.0]),
        1,
    )
    right.result.points = np.array([[0.0, -0.75], [5.0, -0.75]], dtype=float)
    navigator.left_row = left
    navigator.right_row = right
    navigator.state = MissionState.FOLLOW_ROW
    return navigator


def test_start_callback_sets_requested_pattern():
    navigator = make_navigator_for_start_callback()

    request = types.SimpleNamespace(pattern="3L  2r", carefulness="high")
    response = navigator.start_cb(request, types.SimpleNamespace())

    assert response.success
    assert response.message == (
        "Navigation started with pattern: 3L  2r; carefulness: high; object detection disabled"
    )
    assert navigator.p.pattern == "3L  2r"
    assert navigator.pattern_steps == [PatternStep(3, "L"), PatternStep(2, "R")]
    assert navigator.state == MissionState.INITIALIZING
    assert navigator.current_carefulness == "high"
    assert navigator.published_audio[-1] == "navigation started"


def test_start_callback_clears_handled_tracked_object_ids():
    navigator = make_navigator_for_start_callback()
    navigator.handled_tracked_object_ids = {7}

    response = navigator.start_cb(
        types.SimpleNamespace(pattern="3L", carefulness="high", model_path=""),
        types.SimpleNamespace(),
    )

    assert response.success
    assert navigator.handled_tracked_object_ids == set()
    assert navigator.tracker_reset_calls == 1


def test_start_callback_defaults_to_high_carefulness_for_empty_value():
    navigator = make_navigator_for_start_callback()

    response = navigator.start_cb(types.SimpleNamespace(pattern="3L", carefulness=""), types.SimpleNamespace())

    assert response.success
    assert navigator.current_carefulness == "high"


def test_start_callback_applies_requested_carefulness_profile():
    navigator = make_navigator_for_start_callback()
    high = navigator.driving_profiles["high"]

    request = types.SimpleNamespace(pattern="3L", carefulness="medium")
    response = navigator.start_cb(request, types.SimpleNamespace())

    assert response.success
    assert navigator.current_carefulness == "medium"
    assert navigator.p.follow_speed > high.follow_speed
    assert navigator.p.slow_speed > high.slow_speed
    assert navigator.p.lookahead_speed_reduction_gain < high.lookahead_speed_reduction_gain
    assert navigator.p.lookahead_curvature_gain < high.lookahead_curvature_gain
    assert navigator.p.laser_max_weight_both_sides > high.laser_max_weight_both_sides
    assert navigator.p.laser_max_weight_one_side > high.laser_max_weight_one_side
    assert navigator.current_lookahead_distance == navigator.p.lookahead_distance


def test_start_callback_rejects_invalid_pattern_without_changing_mission():
    navigator = make_navigator_for_start_callback()

    request = types.SimpleNamespace(pattern="3L invalid", carefulness="high")
    response = navigator.start_cb(request, types.SimpleNamespace())

    assert not response.success
    assert navigator.p.pattern == "1L 2R"
    assert navigator.pattern_steps == [PatternStep(1, "L")]
    assert navigator.published_audio[-1] == "navigation error"


def test_start_callback_rejects_invalid_carefulness_without_changing_mission():
    navigator = make_navigator_for_start_callback()

    request = types.SimpleNamespace(pattern="3L", carefulness="turbo")
    response = navigator.start_cb(request, types.SimpleNamespace())

    assert not response.success
    assert navigator.p.pattern == "1L 2R"
    assert navigator.pattern_steps == [PatternStep(1, "L")]
    assert navigator.current_carefulness == "high"


def test_pause_and_resume_keep_mission_state_and_profile():
    navigator = make_navigator_for_start_callback()
    navigator.state = MissionState.FOLLOW_ROW
    navigator.current_carefulness = "medium"
    navigator.p.follow_speed = navigator.driving_profiles["medium"].follow_speed
    navigator.midline = np.array([[0.0, 0.0], [1.0, 0.0]])
    navigator.handled_tracked_object_ids = {11}

    pause_response = navigator.pause_cb(types.SimpleNamespace(), types.SimpleNamespace())

    assert pause_response.success
    assert navigator.state == MissionState.PAUSED
    assert navigator.paused_state == MissionState.FOLLOW_ROW
    assert navigator.current_carefulness == "medium"
    assert np.array_equal(navigator.midline, np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert len(navigator.published_cmds) == 1

    resume_response = navigator.resume_cb(types.SimpleNamespace(), types.SimpleNamespace())

    assert resume_response.success
    assert navigator.state == MissionState.FOLLOW_ROW
    assert navigator.paused_state is None
    assert navigator.current_carefulness == "medium"
    assert navigator.handled_tracked_object_ids == {11}
    assert navigator.published_audio == ["navigation stopped", "navigation continuing"]


def test_pause_rejects_idle_navigation():
    navigator = make_navigator_for_start_callback()
    navigator.state = MissionState.IDLE

    response = navigator.pause_cb(types.SimpleNamespace(), types.SimpleNamespace())

    assert not response.success
    assert navigator.state == MissionState.IDLE
    assert navigator.published_audio[-1] == "navigation error"


def test_resume_rejects_when_not_paused():
    navigator = make_navigator_for_start_callback()
    navigator.state = MissionState.FOLLOW_ROW

    response = navigator.resume_cb(types.SimpleNamespace(), types.SimpleNamespace())

    assert not response.success
    assert navigator.state == MissionState.FOLLOW_ROW
    assert navigator.published_audio[-1] == "navigation error"


def test_reset_clears_navigation_and_sensor_state():
    navigator = make_navigator_for_start_callback()
    navigator.state = MissionState.FOLLOW_ROW
    navigator.paused_state = MissionState.FIND_NEXT_ROW_ENTRANCE
    navigator.latest_map = object()
    navigator.latest_scan = object()
    navigator.latest_scan_received_ns = 123
    navigator.robot_pose = Pose2D(1.0, 2.0, 0.5)
    navigator.midline = np.array([[0.0, 0.0], [1.0, 0.0]])
    navigator.stored_rows = {1: np.array([[0.0, 0.0]])}
    navigator.handled_tracked_object_ids = {3}

    response = navigator.reset_cb(types.SimpleNamespace(), types.SimpleNamespace())

    assert response.success
    assert navigator.state == MissionState.IDLE
    assert navigator.paused_state is None
    assert navigator.latest_map is None
    assert navigator.latest_scan is None
    assert navigator.latest_scan_received_ns is None
    assert navigator.robot_pose is None
    assert len(navigator.midline) == 0
    assert navigator.stored_rows == {}
    assert navigator.handled_tracked_object_ids == set()
    assert len(navigator.published_cmds) == 1
    assert navigator.tracker_reset_calls == 1


def test_stop_callback_publishes_navigation_stopped_audio():
    navigator = make_navigator_for_start_callback()
    navigator.state = MissionState.FOLLOW_ROW
    navigator.store_current_rows = lambda: None
    navigator.export_row_map = lambda: None
    navigator.release_object_detection = lambda reason: None
    navigator.reset_entrance_state = lambda: None

    response = navigator.stop_cb(types.SimpleNamespace(), types.SimpleNamespace())

    assert response.success
    assert navigator.state == MissionState.IDLE
    assert navigator.published_audio[-1] == "navigation stopped"


def test_tracked_object_near_left_row_creates_midline_stop():
    navigator = make_object_stop_navigator()
    navigator.latest_tracked_objects = make_tracked_array(make_tracked_object(1, 1.2, 0.75))

    stop = navigator.find_next_object_stop(np.array([0.0, 0.0]), navigator.midline)

    assert stop is not None
    assert stop.object_id == 1
    assert stop.row_side == "left"
    assert stop.plant_row_number == 2
    assert stop.plant_row_offset == 1
    assert np.allclose(stop.stop_point, np.array([1.2, 0.0]))


def test_tracked_object_near_right_row_creates_midline_stop():
    navigator = make_object_stop_navigator()
    navigator.latest_tracked_objects = make_tracked_array(make_tracked_object(2, 1.5, -0.75))

    stop = navigator.find_next_object_stop(np.array([0.0, 0.0]), navigator.midline)

    assert stop is not None
    assert stop.object_id == 2
    assert stop.row_side == "right"
    assert stop.plant_row_number == 1
    assert stop.plant_row_offset == -1
    assert np.allclose(stop.stop_point, np.array([1.5, 0.0]))


def test_tracked_object_outside_current_rows_does_not_create_stop():
    navigator = make_object_stop_navigator()
    navigator.latest_tracked_objects = make_tracked_array(make_tracked_object(3, 1.5, 2.0))

    stop = navigator.find_next_object_stop(np.array([0.0, 0.0]), navigator.midline)

    assert stop is None


def test_handled_tracked_object_id_does_not_create_second_stop():
    navigator = make_object_stop_navigator()
    navigator.handled_tracked_object_ids = {4}
    navigator.latest_tracked_objects = make_tracked_array(make_tracked_object(4, 1.0, 0.75))

    stop = navigator.find_next_object_stop(np.array([0.0, 0.0]), navigator.midline)

    assert stop is None


def test_active_object_stop_holds_for_two_seconds_then_marks_handled():
    navigator = make_object_stop_navigator()
    navigator.p.object_stop_duration_sec = 2.0
    navigator.active_object_stop = ObjectStop(
        object_id=5,
        label="plant",
        row_side="left",
        object_point=np.array([0.0, 0.75]),
        stop_point=np.array([0.0, 0.0]),
        distance_ahead=0.0,
    )
    robot_xy = np.array([0.05, 0.0])

    assert navigator.handle_active_object_stop(robot_xy)
    assert navigator.active_object_stop is not None
    assert navigator.active_object_stop.holding
    assert navigator.handled_tracked_object_ids == set()
    assert len(navigator.published_cmds) == 1
    assert navigator.published_cmds[-1].linear.x == 0.0
    assert navigator.published_audio == ["plant detected on the left"]

    navigator.now_ns = int(2.1e9)
    assert navigator.handle_active_object_stop(robot_xy)

    assert navigator.active_object_stop is None
    assert navigator.object_stop_hold_until_ns is None
    assert navigator.handled_tracked_object_ids == {5}
    assert len(navigator.published_cmds) == 2
    assert navigator.published_cmds[-1].linear.x == 0.0
    assert navigator.published_audio == ["plant detected on the left", "navigation continuing"]


def test_active_object_stop_uses_object_fallback_label():
    navigator = make_object_stop_navigator()
    navigator.active_object_stop = ObjectStop(
        object_id=6,
        label="",
        row_side="right",
        object_point=np.array([0.0, -0.75]),
        stop_point=np.array([0.0, 0.0]),
        distance_ahead=0.0,
    )

    assert navigator.handle_active_object_stop(np.array([0.0, 0.0]))

    assert navigator.published_audio == ["object detected on the right"]


def test_row_aware_object_audio_helpers_format_future_messages():
    navigator = make_object_stop_navigator()
    exact_stop = ObjectStop(
        object_id=7,
        label="weed",
        row_side="left",
        object_point=np.array([0.0, 0.75]),
        stop_point=np.array([0.0, 0.0]),
        distance_ahead=0.0,
        plant_row_number=4,
    )
    offset_stop = ObjectStop(
        object_id=8,
        label="weed",
        row_side="left",
        object_point=np.array([0.0, 2.25]),
        stop_point=np.array([0.0, 0.0]),
        distance_ahead=0.0,
        plant_row_offset=2,
    )
    far_right_stop = ObjectStop(
        object_id=9,
        label="weed",
        row_side="right",
        object_point=np.array([0.0, -3.0]),
        stop_point=np.array([0.0, 0.0]),
        distance_ahead=0.0,
        plant_row_offset=-4,
    )

    assert navigator.build_object_row_number_audio(exact_stop) == "weed detected in plant row 4 on the left"
    assert navigator.build_object_row_offset_audio(offset_stop) == "weed detected two plant rows left"
    assert navigator.build_object_row_offset_audio(far_right_stop) == "weed detected 4 plant rows right"


def test_follow_row_publishes_navigation_finished_audio():
    navigator = make_object_stop_navigator()
    navigator.robot_pose = Pose2D(0.0, 0.0, 0.0)
    navigator.row_exit_goal = np.array([0.0, 0.0])
    navigator.row_end_direction = np.array([1.0, 0.0])
    navigator.finish_after_current_row = True
    navigator.recompute_rows = lambda: None
    navigator.pose_goal_reached = lambda goal, direction: True
    navigator.store_current_rows = lambda: None
    navigator.record_current_row_end_direction = lambda: None
    navigator.export_row_map = lambda: None
    navigator.release_object_detection = lambda reason: None

    navigator.handle_follow_row()

    assert navigator.state == MissionState.FINISHED
    assert navigator.published_audio[-1] == "navigation finished"


def test_service_response_failure_publishes_navigation_error_audio():
    navigator = make_navigator_for_start_callback()

    assert not navigator.service_response_success(ImmediateFuture(success=False, message="nope"), "detector start")

    assert navigator.published_audio[-1] == "navigation error"


def test_tracker_active_is_published_on_detector_start_and_stop():
    navigator = make_navigator_for_start_callback()
    navigator.object_detection_enabled = True
    navigator.state = MissionState.FOLLOW_ROW

    navigator.handle_detector_start_response(ImmediateFuture(success=True), "test")

    assert navigator.object_detection_started
    assert navigator.published_tracker_active[-1] is True

    navigator.detector = types.SimpleNamespace(stop=lambda: ImmediateFuture(success=True))
    navigator.stop_object_detection("test")

    assert not navigator.object_detection_started
    assert navigator.published_tracker_active[-1] is False


def test_tracker_active_is_published_false_on_release():
    navigator = make_navigator_for_start_callback()
    navigator.object_detection_enabled = True
    navigator.object_detection_started = False
    navigator.detector = types.SimpleNamespace(
        clear_results=lambda: None,
        release=lambda: ImmediateFuture(success=True),
    )

    navigator.release_object_detection("test")

    assert navigator.published_tracker_active[-1] is False


def test_driving_profiles_are_derived_from_high_profile():
    navigator = make_navigator_for_start_callback()
    high = navigator.driving_profiles["high"]
    medium = navigator.driving_profiles["medium"]
    low = navigator.driving_profiles["low"]

    assert isinstance(high, DrivingProfile)
    assert math.isclose(high.laser_max_weight_both_sides, 0.30)
    assert math.isclose(high.laser_max_weight_one_side, 0.15)
    assert math.isclose(medium.laser_max_weight_both_sides, 0.55)
    assert math.isclose(medium.laser_max_weight_one_side, 0.275)
    assert math.isclose(low.laser_max_weight_both_sides, 0.80)
    assert math.isclose(low.laser_max_weight_one_side, 0.40)
    assert medium.follow_speed > high.follow_speed
    assert low.follow_speed > medium.follow_speed
    assert medium.laser_max_weight_both_sides > high.laser_max_weight_both_sides
    assert low.laser_max_weight_both_sides > medium.laser_max_weight_both_sides
    assert medium.lookahead_speed_reduction_gain < high.lookahead_speed_reduction_gain
    assert low.lookahead_speed_reduction_gain < medium.lookahead_speed_reduction_gain


def test_both_rows_produce_centered_high_weight_target():
    follower = LaserRowFollower(NavigatorParams())
    scan = make_scan([(0.0, 0.375), (0.0, -0.375)])

    result = process_repeatedly(follower, scan)

    assert result.valid
    assert result.left_line.valid
    assert result.right_line.valid
    assert result.weight > 0.25
    assert result.weight <= follower.p.laser_max_weight_both_sides
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
    navigator.row_exit_goal = None
    navigator.entrance_route = np.empty((0, 2))
    navigator.entrance_route_support = np.empty((0, 2))
    navigator.entrance_route_progress_index = 0
    navigator.entrance_route_projection = None
    navigator.entrance_route_target = None
    navigator.entrance_route_remaining_distance = 0.0
    navigator.entrance_route_provisional = False
    navigator.current_maneuver_lookahead_distance = navigator.p.maneuver_lookahead_distance
    navigator.current_maneuver_lookahead_curvature = 0.0
    navigator.get_logger = lambda: types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None,
    )
    navigator.entrance_active_step = None
    navigator.entrance_traverse_outward = None
    navigator.entrance_target = None
    navigator.entrance_target_direction = None
    navigator.entrance_follow_path = np.empty((0, 2))
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


def test_dynamic_follow_lookahead_stays_maximum_on_straight_midline():
    navigator = bare_navigator()
    navigator.p.lookahead_distance = 1.0
    navigator.p.turn_lookahead_distance = 0.45
    navigator.p.lookahead_curvature_gain = 1.5
    midline = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    lookahead = navigator.dynamic_follow_lookahead(midline, np.array([0.0, 0.0]))

    assert math.isclose(lookahead, navigator.p.lookahead_distance)
    assert math.isclose(navigator.current_lookahead_curvature, 0.0)


def test_dynamic_follow_lookahead_shrinks_when_robot_is_laterally_offset():
    navigator = bare_navigator()
    navigator.p.lookahead_distance = 1.0
    navigator.p.turn_lookahead_distance = 0.45
    navigator.p.lookahead_curvature_gain = 0.0
    navigator.p.lookahead_lateral_error_gain = 2.0
    midline = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    centered = navigator.dynamic_follow_lookahead(midline, np.array([0.0, 0.0]))
    offset = navigator.dynamic_follow_lookahead(midline, np.array([0.0, 0.25]))

    assert math.isclose(centered, navigator.p.lookahead_distance)
    assert offset < centered


def test_dynamic_follow_lookahead_shrinks_on_ninety_degree_midline():
    navigator = bare_navigator()
    navigator.p.lookahead_distance = 1.0
    navigator.p.turn_lookahead_distance = 0.45
    navigator.p.lookahead_curvature_gain = 1.5
    midline = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.5, 1.0]])

    lookahead = navigator.dynamic_follow_lookahead(midline, np.array([0.0, 0.0]))

    assert lookahead < navigator.p.lookahead_distance
    assert navigator.current_lookahead_curvature > 0.0


def test_dynamic_follow_lookahead_detects_multi_point_bend():
    navigator = bare_navigator()
    navigator.p.lookahead_distance = 1.0
    navigator.p.turn_lookahead_distance = 0.45
    navigator.p.lookahead_curvature_gain = 1.5
    navigator.p.lookahead_curvature_sample_count = 7
    midline = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.00],
            [0.50, 0.04],
            [0.75, 0.13],
            [1.00, 0.27],
        ]
    )

    lookahead = navigator.dynamic_follow_lookahead(midline, np.array([0.0, 0.0]))

    assert lookahead < navigator.p.lookahead_distance
    assert navigator.current_lookahead_curvature > 0.0


def test_dynamic_follow_lookahead_can_include_curve_behind_robot():
    navigator = bare_navigator()
    navigator.p.lookahead_distance = 1.0
    navigator.p.turn_lookahead_distance = 0.45
    navigator.p.lookahead_curvature_gain = 1.5
    navigator.p.lookahead_curvature_sample_count = 7
    navigator.p.lookahead_curvature_back_distance = 0.5
    midline = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.5, 1.5]])

    lookahead = navigator.dynamic_follow_lookahead(midline, np.array([0.5, 0.25]))

    assert lookahead < navigator.p.lookahead_distance
    assert navigator.current_lookahead_curvature > 0.0


def test_dynamic_follow_lookahead_is_clamped_to_turn_distance():
    navigator = bare_navigator()
    navigator.p.lookahead_distance = 1.0
    navigator.p.turn_lookahead_distance = 0.45
    navigator.p.lookahead_curvature_gain = 100.0
    navigator.p.lookahead_filter_alpha = 1.0
    midline = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

    lookahead = navigator.dynamic_follow_lookahead(midline, np.array([0.0, 0.0]))

    assert math.isclose(lookahead, navigator.p.turn_lookahead_distance)


def test_dynamic_follow_lookahead_filter_smooths_sudden_reduction():
    navigator = bare_navigator()
    navigator.p.lookahead_distance = 1.0
    navigator.p.turn_lookahead_distance = 0.45
    navigator.p.lookahead_curvature_gain = 100.0
    navigator.p.lookahead_filter_alpha = 0.25
    navigator.current_lookahead_distance = 1.0
    midline = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

    lookahead = navigator.dynamic_follow_lookahead(midline, np.array([0.0, 0.0]))

    assert navigator.p.turn_lookahead_distance < lookahead < navigator.p.lookahead_distance
    assert math.isclose(lookahead, 0.8625)


def test_small_dynamic_lookahead_reduces_follow_speed_before_curvature_grows():
    published = []
    navigator = bare_navigator()
    navigator.cmd_pub = types.SimpleNamespace(publish=published.append)
    navigator.robot_pose = Pose2D(0.0, 0.0, 0.0)
    navigator.last_target_point = None
    navigator.last_cmd_angular_z = 0.0
    navigator.p.follow_speed = 0.40
    navigator.p.slow_speed = 0.10
    navigator.p.lookahead_distance = 1.0
    navigator.p.curve_speed_reduction_gain = 0.0
    navigator.p.lookahead_speed_reduction_gain = 1.5
    navigator.current_lookahead_distance = 1.0

    navigator.drive_to_point(np.array([1.0, 0.0]))
    full_lookahead_speed = published[-1].linear.x

    navigator.last_target_point = None
    navigator.last_cmd_angular_z = 0.0
    navigator.current_lookahead_distance = 0.5
    navigator.drive_to_point(np.array([1.0, 0.0]))
    reduced_lookahead_speed = published[-1].linear.x

    assert math.isclose(full_lookahead_speed, navigator.p.follow_speed)
    assert reduced_lookahead_speed < full_lookahead_speed


def test_small_dynamic_maneuver_lookahead_reduces_maneuver_speed_before_curvature_grows():
    published = []
    navigator = bare_navigator()
    navigator.cmd_pub = types.SimpleNamespace(publish=published.append)
    navigator.robot_pose = Pose2D(0.0, 0.0, 0.0)
    navigator.last_target_point = None
    navigator.last_cmd_angular_z = 0.0
    navigator.p.maneuver_speed = 0.40
    navigator.p.maneuver_slow_speed = 0.10
    navigator.p.maneuver_lookahead_distance = 1.0
    navigator.p.curve_speed_reduction_gain = 0.0
    navigator.p.maneuver_lookahead_speed_reduction_gain = 1.2
    navigator.current_maneuver_lookahead_distance = 1.0

    navigator.drive_to_point(np.array([1.0, 0.0]), navigator.p.maneuver_speed, navigator.p.maneuver_slow_speed)
    full_lookahead_speed = published[-1].linear.x

    navigator.last_target_point = None
    navigator.last_cmd_angular_z = 0.0
    navigator.current_maneuver_lookahead_distance = 0.5
    navigator.drive_to_point(np.array([1.0, 0.0]), navigator.p.maneuver_speed, navigator.p.maneuver_slow_speed)
    reduced_lookahead_speed = published[-1].linear.x

    assert math.isclose(full_lookahead_speed, navigator.p.maneuver_speed)
    assert reduced_lookahead_speed < full_lookahead_speed


def test_rounded_route_replaces_sharp_corner_with_curve():
    navigator = bare_navigator()
    support = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    route = navigator.build_rounded_route(support)

    assert len(route) > 3
    assert not any(np.allclose(point, np.array([1.0, 0.0])) for point in route)
    assert np.allclose(route[0], support[0])
    assert np.allclose(route[-1], support[-1])


def test_rounded_route_is_not_capped_by_minimum_follow_radius():
    navigator = bare_navigator()
    navigator.p.maneuver_corner_radius = 0.75
    navigator.p.min_follow_turn_radius = 0.10
    support = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    route = navigator.build_rounded_route(support)
    first_curve_point = route[np.where(route[:, 1] > 0.0)[0][0]]

    assert first_curve_point[0] < 1.5


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


def test_dynamic_maneuver_lookahead_stays_maximum_on_straight_route():
    navigator = bare_navigator()
    navigator.p.maneuver_lookahead_distance = 0.8
    navigator.p.maneuver_turn_lookahead_distance = 0.35
    navigator.p.maneuver_lookahead_curvature_gain = 2.5
    navigator.p.maneuver_lookahead_lateral_error_gain = 0.5
    navigator.p.maneuver_lookahead_filter_alpha = 1.0
    route = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    lookahead = navigator.dynamic_maneuver_lookahead(route, route[0], 0, 0.0, 2.0)

    assert math.isclose(lookahead, navigator.p.maneuver_lookahead_distance)
    assert math.isclose(navigator.current_maneuver_lookahead_curvature, 0.0)


def test_dynamic_maneuver_lookahead_shrinks_on_curved_route():
    navigator = bare_navigator()
    navigator.p.maneuver_lookahead_distance = 0.8
    navigator.p.maneuver_turn_lookahead_distance = 0.35
    navigator.p.maneuver_lookahead_curvature_gain = 2.5
    navigator.p.maneuver_lookahead_filter_alpha = 1.0
    route = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.5, 1.0]])

    lookahead = navigator.dynamic_maneuver_lookahead(route, route[0], 0, 0.0, 2.0)

    assert lookahead < navigator.p.maneuver_lookahead_distance
    assert navigator.current_maneuver_lookahead_curvature > 0.0


def test_dynamic_maneuver_lookahead_is_clamped_to_turn_distance():
    navigator = bare_navigator()
    navigator.p.maneuver_lookahead_distance = 0.8
    navigator.p.maneuver_turn_lookahead_distance = 0.35
    navigator.p.maneuver_lookahead_curvature_gain = 100.0
    navigator.p.maneuver_lookahead_filter_alpha = 1.0
    route = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

    lookahead = navigator.dynamic_maneuver_lookahead(route, route[0], 0, 0.0, 2.0)

    assert math.isclose(lookahead, navigator.p.maneuver_turn_lookahead_distance)


def test_dynamic_maneuver_lookahead_filter_smooths_sudden_reduction():
    navigator = bare_navigator()
    navigator.p.maneuver_lookahead_distance = 0.8
    navigator.p.maneuver_turn_lookahead_distance = 0.35
    navigator.p.maneuver_lookahead_curvature_gain = 100.0
    navigator.p.maneuver_lookahead_filter_alpha = 0.25
    navigator.current_maneuver_lookahead_distance = 0.8
    route = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])

    lookahead = navigator.dynamic_maneuver_lookahead(route, route[0], 0, 0.0, 2.0)

    assert navigator.p.maneuver_turn_lookahead_distance < lookahead < navigator.p.maneuver_lookahead_distance
    assert math.isclose(lookahead, 0.6875)


def test_dynamic_maneuver_lookahead_respects_remaining_distance_limit():
    navigator = bare_navigator()
    navigator.p.maneuver_lookahead_distance = 0.8
    navigator.p.maneuver_turn_lookahead_distance = 0.35
    navigator.p.maneuver_lookahead_filter_alpha = 1.0
    route = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    lookahead = navigator.dynamic_maneuver_lookahead(route, route[0], 0, 0.0, 0.5)

    assert math.isclose(lookahead, 0.30)


def test_active_navigation_cleanup_removed_unused_parameters():
    source = Path("src/maize_navigation/maize_navigation/maize_navigation.py").read_text()
    params = Path("src/maize_navigation/config/params.yaml").read_text()
    param_names = {line.strip().split(":", 1)[0] for line in params.splitlines() if ":" in line}

    assert "yaw_kp" not in source
    assert "p.max_angular_speed" not in source
    assert "    max_angular_speed: float" not in source
    for unused_param in (
        "odom_frame",
        "use_slam_map",
        "require_map_for_turns",
        "roi_x_min",
        "roi_x_max",
        "roi_y_abs_min",
        "roi_y_abs_max",
        "acquire_roi_x_min",
        "acquire_roi_x_max",
        "acquire_roi_y_abs_min",
        "acquire_roi_y_abs_max",
        "ransac_iterations",
        "ransac_distance",
        "min_inliers",
        "min_visible_length",
        "max_abs_line_slope",
        "centerline_max_abs_slope",
        "tracker_alpha",
        "confidence_decay",
        "front_density_x_min",
        "front_density_x_max",
        "front_density_y_abs",
        "front_density_threshold",
        "end_probability_threshold",
        "end_stable_frames_required",
        "min_follow_confidence",
        "min_enter_confidence",
        "enter_stable_frames_required",
        "acquire_timeout_sec",
        "enter_speed",
        "turn_speed",
        "heading_kp",
        "lateral_kp",
        "stanley_min_speed",
        "max_linear_speed",
        "max_angular_speed",
        "turn_max_angular_speed",
        "turn_min_angular_speed",
        "path_goal_yaw_tolerance",
        "exit_distance",
        "turn_forward_distance",
        "min_turn_radius",
        "enter_distance",
        "row_shift_count",
        "row_shift_direction",
        "turn_180",
        "headland_maneuver_enabled",
        "headland_exit_straight_distance",
        "headland_exit_straight_speed",
        "exit_curve_speed",
        "exit_curve_angular_speed",
        "exit_curve_yaw_change",
        "headland_shift_speed",
        "headland_shift_tolerance",
        "headland_shift_overshoot_tolerance",
        "headland_yaw_tolerance",
        "headland_use_map_row_heading",
        "headland_heading_kp",
        "headland_heading_max_yaw_error",
        "entry_curve_speed",
        "entry_curve_angular_speed",
        "headland_total_yaw_change",
        "entry_curve_yaw_change",
        "entry_yaw_accept_tolerance",
        "entry_shift_accept_tolerance",
        "entry_row_min_confidence",
        "entry_row_stable_frames",
        "entry_require_full_lane",
        "entry_center_b_tolerance",
        "entry_lane_width_tolerance",
        "entry_row_yaw_tolerance",
        "neighbor_reference_turn_enabled",
        "neighbor_reference_entry_requires_shift",
        "neighbor_reference_requires_same_side_row",
        "map_row_detection_enabled",
        "map_row_search_x_forward",
        "map_row_search_x_backward",
        "map_row_search_y_side",
        "map_row_use_pca_orientation",
        "map_row_pca_radius",
        "map_row_pca_min_points",
        "map_row_lateral_bin",
        "map_row_min_band_points",
        "map_row_min_band_length",
        "map_row_max_extrapolated_lanes",
        "map_row_line_ransac_iterations",
        "map_row_line_distance",
        "map_row_min_line_inliers",
        "map_row_min_line_length",
        "map_row_max_abs_line_slope",
        "map_row_max_lines",
        "map_row_line_merge_distance",
        "map_lane_accept_tolerance",
        "turn_replan_enabled",
        "turn_replan_period_frames",
        "turn_replan_max_attempts",
        "turn_exit_on_local_row",
        "turn_exit_min_confidence",
        "turn_exit_stable_frames",
        "enable_safety",
        "obstacle_stop_distance",
        "obstacle_slow_distance",
    ):
        assert unused_param not in param_names


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
    assert np.allclose(navigator.entrance_waypoints[0], np.array([0.8, 0.5]))
    assert np.allclose(navigator.entrance_route[-1], np.array([0.8, 1.5]))


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

    assert np.allclose(navigator.entrance_waypoints[0], np.array([1.2, 0.5]))
    assert np.allclose(navigator.entrance_waypoints[1], np.array([1.2, 1.5]))


def test_support_route_keeps_first_waypoint_visible_when_active_route_skips_it():
    navigator = bare_navigator()
    navigator.robot_pose = Pose2D(0.26, 0.0, 0.0)
    navigator.plant_row_end_point = np.array([0.0, 0.0])
    navigator.row_end_direction = np.array([1.0, 0.0])
    navigator.entrance_active_step = PatternStep(1, "L")

    navigator.rebuild_entrance_route(None)

    assert len(navigator.entrance_waypoints) == 2
    assert np.allclose(navigator.entrance_route_support[0], np.array([0.3, 0.0]))
    assert np.allclose(navigator.entrance_route_support[2], np.array([0.675, 0.375]))
    assert np.any(np.all(np.isclose(navigator.entrance_route, np.array([0.675, 0.375])), axis=1))


def test_headland_route_stays_anchored_at_row_exit_goal():
    navigator = bare_navigator()
    navigator.robot_pose = Pose2D(0.9, 0.2, 0.0)
    navigator.plant_row_end_point = np.array([0.0, 0.0])
    navigator.row_exit_goal = np.array([0.3, 0.0])
    navigator.row_end_direction = np.array([1.0, 0.0])
    navigator.entrance_active_step = PatternStep(2, "L")

    navigator.rebuild_entrance_route(None)

    assert np.allclose(navigator.entrance_route[0], navigator.row_exit_goal)
    assert np.allclose(navigator.entrance_route_support[0], navigator.row_exit_goal)


def test_entrance_peaks_are_regularized_to_expected_spacing():
    navigator = bare_navigator()
    peaks = [
        EntrancePeak(0.00, np.array([0.0, 0.0])),
        EntrancePeak(0.74, np.array([0.0, 0.74])),
        EntrancePeak(1.10, np.array([0.0, 1.10])),
        EntrancePeak(1.50, np.array([0.0, 1.50])),
        EntrancePeak(2.25, np.array([0.0, 2.25])),
    ]

    regularized = navigator.regularize_entrance_peak_spacing(peaks)

    assert [round(peak.lateral, 2) for peak in regularized] == [0.0, 0.74, 1.5, 2.25]


def test_locked_turn_route_is_not_resampled_for_small_peak_jitter():
    navigator = bare_navigator()
    navigator.get_logger = lambda: type("Logger", (), {"info": lambda *args, **kwargs: None})()
    navigator.robot_pose = Pose2D(0.0, 0.0, 0.0)
    navigator.plant_row_end_point = np.array([0.0, 0.0])
    navigator.row_end_direction = np.array([1.0, 0.0])
    navigator.entrance_active_step = PatternStep(2, "L")
    navigator.pattern_index = 0
    navigator.pattern_steps = [PatternStep(2, "L")]
    navigator.entrance_route_provisional = True
    navigator.left_row = RowMarchModel("left", np.array([0.0, 0.75]), np.array([0.3, 0.75]), np.array([1.0, 0.0]), 2)
    navigator.right_row = RowMarchModel("right", np.array([0.0, 0.0]), np.array([0.3, 0.0]), np.array([1.0, 0.0]), 1)
    navigator.row_number_increase_direction = np.array([0.0, 1.0])
    navigator.get_all_map_points = lambda: np.empty((0, 2))
    navigator.associate_known_rows_with_entrance_peaks = lambda center, outgoing: (0, 1)
    navigator.build_entrance_follow_path = lambda: np.array([[0.8, 1.5], [1.6, 1.5]])
    navigator.find_entrance_histogram_peaks = lambda points, center, outgoing: [
        EntrancePeak(0.0, np.array([0.0, 0.0]), 1),
        EntrancePeak(0.75, np.array([0.0, 0.75]), 2),
        EntrancePeak(1.5, np.array([0.0, 1.5]), 3),
        EntrancePeak(2.25, np.array([0.0, 2.25]), 4),
    ]

    assert navigator.lock_next_row_entrance()
    locked_route = np.array(navigator.entrance_route, copy=True)
    navigator.robot_pose = Pose2D(0.4, 0.1, 0.0)
    navigator.find_entrance_histogram_peaks = lambda points, center, outgoing: [
        EntrancePeak(0.0, np.array([0.0, 0.0]), 1),
        EntrancePeak(0.75, np.array([0.0, 0.75]), 2),
        EntrancePeak(1.5, np.array([0.02, 1.5]), 3),
        EntrancePeak(2.25, np.array([0.0, 2.25]), 4),
    ]

    assert navigator.lock_next_row_entrance()

    assert np.allclose(navigator.entrance_route, locked_route)


def test_missed_entry_line_replans_forward_onto_follow_path():
    navigator = bare_navigator()
    navigator.robot_pose = Pose2D(0.2, 0.35, 0.0)
    navigator.plant_row_end_point = np.array([0.0, 0.0])
    navigator.row_end_direction = np.array([-1.0, 0.0])
    navigator.entrance_active_step = PatternStep(1, "L")
    navigator.entrance_target_direction = np.array([1.0, 0.0])
    navigator.entrance_follow_path = np.array([[0.8, 0.0], [1.6, 0.0], [2.4, 0.0]])

    navigator.rebuild_entrance_route(np.array([0.0, 0.0]))

    assert np.allclose(navigator.entrance_route[0], np.array([0.2, 0.35]))
    assert np.all(np.diff(navigator.entrance_route[:, 0]) >= -1e-9)
    assert np.allclose(navigator.entrance_route[-1], np.array([2.4, 0.0]))
