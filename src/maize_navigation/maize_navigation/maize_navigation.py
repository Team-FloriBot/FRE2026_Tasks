import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import numpy as np
from enum import Enum
from dataclasses import dataclass
import re

@dataclass
class PerceptionData:
    left_dist: float
    right_dist: float
    center_error: float
    row_detected: bool
    min_dist: float


@dataclass
class ControlCommand:
    linear: float
    angular: float


# class Perception handles laser data preprocessing
class Perception:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def process(self, cloud_msg) -> PerceptionData:
        left = []
        right = []
        min_dist = float('inf')

        for p in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            dist = np.sqrt(x**2 + y**2)
            min_dist = min(min_dist, dist)

            if self.x_min < x < self.x_max:
                if -self.y_max < y < -self.y_min:
                    left.append(abs(y))
                elif self.y_min < y < self.y_max:
                    right.append(abs(y))

        if len(left) < 3 or len(right) < 3:
            return PerceptionData(0, 0, 0, False, min_dist)

        left_dist = np.median(left)
        right_dist = np.median(right)

        center_error = (right_dist - left_dist) / 2.0

        return PerceptionData(
            left_dist=left_dist,
            right_dist=right_dist,
            center_error=center_error,
            row_detected=True,
            min_dist=min_dist
        )
    

# State Machine
class State(Enum):
    DRIVE = 1
    EXIT_ROW = 2
    TURN = 3
    ENTER_ROW = 4

class Pattern:
    def __init__(self, pattern_str):
        self.steps = self.parse(pattern_str)
        self.index = 0

    def parse(self, pattern_str):
        tokens = pattern_str.split("-")
        result = []
        for t in tokens:
            match = re.match(r"(\d+)([LR])", t.strip())
            if match:
                count = int(match.group(1))
                direction = match.group(2)
                result.append((count, direction))
        return result

    def current(self):
        if self.index < len(self.steps):
            return self.steps[self.index]
        return None

    def next(self):
        self.index += 1

class StateMachine:
    def __init__(self, pattern: Pattern):
        self.state = State.DRIVE
        self.pattern = pattern

        self.rows_to_skip = 0
        self.current_direction = None
        self.row_counter = 0

    def update(self, perception: PerceptionData):
        if self.state == State.DRIVE:
            if not perception.row_detected:
                step = self.pattern.current()
                if step is None:
                    return self.state

                self.rows_to_skip, self.current_direction = step
                self.row_counter = 0
                self.state = State.EXIT_ROW

        elif self.state == State.EXIT_ROW:
            # To-Do: Aus Reihe herausfahren
            self.state = State.TURN

        elif self.state == State.TURN:
            # To-Do: Wende gemäß Muster (rechts oder links)
            self.state = State.ENTER_ROW

        elif self.state == State.ENTER_ROW:
            if perception.row_detected:
                self.row_counter += 1

                if self.row_counter >= self.rows_to_skip:
                    self.pattern.next()
                    self.state = State.DRIVE

        return self.state

# Controller
class Controller:
    def __init__(self):
        self.kp = 2.0
        self.kd = 0.0
        self.ki = 0.0

        self.prev_error = 0
        self.integral = 0

    def compute(self, state: State, perception: PerceptionData, direction=None):
        if state == State.DRIVE:
            return self.drive(perception)

        elif state == State.EXIT_ROW:
            return ControlCommand(0.3, 0.0)

        elif state == State.TURN:
            return self.turn(direction)

        elif state == State.ENTER_ROW:
            return ControlCommand(0.2, 0.0)

        return ControlCommand(0.0, 0.0)

    def drive(self, perception):
        error = perception.center_error

        self.integral += error
        derivative = error - self.prev_error

        angular = self.kp * error + self.kd * derivative
        linear = 0.5

        self.prev_error = error

        return ControlCommand(linear, angular)

    def turn(self, direction):
        if direction == "L":
            return ControlCommand(0.2, 0.8)
        elif direction == "R":
            return ControlCommand(0.2, -0.8)
        return ControlCommand(0.0, 0.0)


# ROS Node
class FieldRobotNavigator(Node):
    def __init__(self):
        super().__init__("maize_navigator")

        # -------------------------
        # Parameter deklarieren
        # -------------------------
        self.declare_parameter("pattern", "1L-1R-2L-3R")

        self.declare_parameter("perception.x_min", 0.0)
        self.declare_parameter("perception.x_max", 2.0)
        self.declare_parameter("perception.y_min", 0.1)
        self.declare_parameter("perception.y_max", 1.0)

        self.declare_parameter("controller.kp", 2.0)
        self.declare_parameter("controller.kd", 0.5)
        self.declare_parameter("controller.ki", 0.0)

        self.declare_parameter("topics.pointcloud", "/point_cloud")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")

        # -------------------------
        # Parameter holen
        # -------------------------
        pattern_str = self.get_parameter("pattern").get_parameter_value().string_value

        x_min = self.get_parameter("perception.x_min").value
        x_max = self.get_parameter("perception.x_max").value
        y_min = self.get_parameter("perception.y_min").value
        y_max = self.get_parameter("perception.y_max").value

        kp = self.get_parameter("controller.kp").value
        kd = self.get_parameter("controller.kd").value
        ki = self.get_parameter("controller.ki").value

        pointcloud_topic = self.get_parameter("topics.pointcloud").value
        cmd_vel_topic = self.get_parameter("topics.cmd_vel").value

        # -------------------------
        # Module initialisieren
        # -------------------------
        self.perception = Perception(x_min, x_max, y_min, y_max)

        self.pattern = Pattern(pattern_str)
        self.state_machine = StateMachine(self.pattern)

        self.controller = Controller()
        self.controller.kp = kp
        self.controller.kd = kd
        self.controller.ki = ki

        self.latest_cloud = None

        # -------------------------
        # ROS Schnittstellen
        # -------------------------
        self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.cloud_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.timer = self.create_timer(0.1, self.loop)

        # -------------------------
        # Parameter-Update Callback (optional, aber stark!)
        # -------------------------
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info("FieldRobotNavigator gestartet")

    # -------------------------
    # Dynamische Parameteränderung
    # -------------------------
    def parameter_callback(self, params):
        for param in params:
            if param.name == "controller.kp":
                self.controller.kp = param.value
            elif param.name == "controller.kd":
                self.controller.kd = param.value
            elif param.name == "controller.ki":
                self.controller.ki = param.value

        return rclpy.parameter.SetParametersResult(successful=True)

    # -------------------------
    def cloud_callback(self, msg):
        self.latest_cloud = msg

    # -------------------------
    def loop(self):
        if self.latest_cloud is None:
            return

        perception = self.perception.process(self.latest_cloud)
        state = self.state_machine.update(perception)

        direction = self.state_machine.current_direction
        cmd = self.controller.compute(state, perception, direction)

        twist = Twist()
        twist.linear.x = cmd.linear
        twist.angular.z = cmd.angular

        self.cmd_pub.publish(twist)

        self.get_logger().debug(f"State: {state}, cmd: {cmd}")


# main
def main(args=None):
    rclpy.init(args=args)
    node = FieldRobotNavigator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

