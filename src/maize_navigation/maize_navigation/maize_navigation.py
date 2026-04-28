import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point32
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger

import numpy as np
from enum import Enum
from dataclasses import dataclass
import re

# ============================================================
# >>> DATENCONTAINER: PERCEPTION OUTPUT
# Wird von Perception berechnet und von StateMachine + Controller genutzt
# ============================================================
@dataclass
class PerceptionData:
    # --- Lidar Daten ---
    left_dist: float = np.inf
    right_dist: float = np.inf
    center_error: float = 0.0
    row_end_detected: bool = False
    
    x_mean: float = np.inf
    y_mean: float = np.inf
    
    min_dist: float = np.inf
    num_points_in_box: int = 0
    filtered_points: list = None


# ============================================================
# >>> DATENCONTAINER: CONTROLLER OUTPUT
# Wird später in Twist übersetzt
# ============================================================
@dataclass
class ControlCommand:
    linear: float
    angular: float


# ============================================================
# >>> STATE MACHINE ZUSTÄNDE
# ============================================================
class State(Enum):
    ROBOT_STOP = 0
    DRIVE_IN_ROW = 1
    EXIT_ROW = 2
    TURN = 3
    COUNTING_ROWS = 4
    ENTER_ROW = 5


# ============================================================
# >>> PATTERN (z.B. "1L-2R")
# Steuert wie viele Reihen gefahren werden und in welche Richtung
# ============================================================
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


# ============================================================
# >>> PERCEPTION (Sensorverarbeitung)
# ============================================================
class Perception:
    def __init__(self, bounding_boxes):
        self.bounding_boxes = bounding_boxes

    def process(self, scan_msg, current_state, pattern_direction) -> PerceptionData:
        data = PerceptionData()
        points = []
        
        # ========================================================
        # >>> STATE-ABHÄNGIGE WAHRNEHMUNG (ANPASSBAR!)
        # ========================================================
        if current_state == State.DRIVE_IN_ROW:
            box = self.bounding_boxes['drive_in_row']
            both_sides = 'both'
        elif current_state == State.EXIT_ROW:
            box = self.bounding_boxes['turn_and_exit']
            both_sides = pattern_direction
        elif current_state == State.TURN:
            box = self.bounding_boxes['turn_and_exit']
            both_sides = pattern_direction
        elif current_state == State.COUNTING_ROWS:
            box = self.bounding_boxes['counting_rows']
            both_sides = pattern_direction
        elif current_state == State.ENTER_ROW:
            box = self.bounding_boxes['turn_to_row']
            both_sides = 'both'
        else:
            box = self.bounding_boxes['drive_in_row']
            both_sides = 'both'

        x_min, x_max = box['x_min'], box['x_max']
        y_min, y_max = box['y_min'], box['y_max']

        min_distance = np.inf

        # ========================================================
        # >>> LASERDATEN → PUNKTE
        # ========================================================
        for i, dist in enumerate(scan_msg.ranges):
            if dist < scan_msg.range_min or dist > scan_msg.range_max or np.isinf(dist):
                continue
            
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            # >>> POLAR → KARTESISCH
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)

            if dist < min_distance:
                min_distance = dist
                
            # ====================================================
            # >>> FILTERLOGIK (HIER PASSIERT DIE MAGIE!)
            # ====================================================
            if both_sides == 'both':
                if y_min < abs(y) < y_max and x_min < x < x_max:
                    points.append(Point32(x=float(x), y=float(y), z=0.0))

            elif both_sides == 'R':
                if -y_max < y < -y_min and x_min < x < x_max:
                    points.append(Point32(x=float(x), y=float(y), z=0.0))
            elif both_sides == 'L':
                if y_min < y < y_max and x_min < x < x_max:
                    points.append(Point32(x=float(x), y=float(y), z=0.0))

        # ========================================================
        # >>> FEATURE BERECHNUNG (ENTSCHEIDUNGSBASIS!)
        # ========================================================
        data.min_dist = min_distance
        data.num_points_in_box = len(points)
        
        left_y = [p.y for p in points if p.y > 0]
        right_y = [p.y for p in points if p.y <= 0]
        
        data.left_dist = np.mean(np.abs(left_y)) if len(left_y) > 0 else np.inf
        data.right_dist = np.mean(np.abs(right_y)) if len(right_y) > 0 else np.inf
        
        # >>> REIHENENDE-ERKENNUNG
        if np.isinf(data.left_dist) or np.isinf(data.right_dist):
            data.row_end_detected = True
        else:
            data.center_error = (data.right_dist - data.left_dist) / 2.0
            data.row_end_detected = False
            
        points_x = [p.x for p in points]
        data.x_mean = np.mean(points_x) if len(points_x) > 0 else np.inf
        
        points_y = [p.y for p in points]
        data.y_mean = np.mean(points_y) if len(points_y) > 0 else np.inf

        data.filtered_points = points

        return data


# ============================================================
# >>> STATE MACHINE
# ============================================================
class StateMachine:
    def __init__(self, pattern, node):
        self.state = State.ROBOT_STOP # Initialer State
        self.pattern = pattern
        self.node = node
        self.navigation_triggered = False # Flag für den Service-Start
        
        # >>> INTERNE VARIABLEN (KRITISCH!)
        self.exit_start_time = 0.0
        
        self.row_counter = 1
        self.previous_row = 1
        self.actual_row = 1
        self.actual_dist = np.inf

    def get_current_direction(self):
        step = self.pattern.current()
        if step:
            return step[1]
        return 'L'

    def update(self, perception: PerceptionData, params):
        old_state = self.state

        # ====================================================
        # >>> STATE: ROBOT_STOP
        # ====================================================
        if self.state == State.ROBOT_STOP:
            if self.navigation_triggered:
                self.state = State.DRIVE_IN_ROW
                self.navigation_triggered = False # Reset Flag

        # ====================================================
        # >>> STATE: DRIVE_IN_ROW
        # ====================================================
        if self.state == State.DRIVE_IN_ROW:
            if perception.row_end_detected:
                self.exit_start_time = self.node.get_clock().now().nanoseconds / 1e9
                self.state = State.EXIT_ROW  # <<< TRANSITION

        # ====================================================
        # >>> STATE: EXIT_ROW
        # ====================================================
        elif self.state == State.EXIT_ROW:
            time_to_drive = params['drive_out_dist'] / params['vel_linear_drive']
            current_time = self.node.get_clock().now().nanoseconds / 1e9
            if (current_time - self.exit_start_time) >= time_to_drive:
                self.state = State.TURN  # <<< TRANSITION

        # ====================================================
        # >>> STATE: TURN
        # ====================================================
        elif self.state == State.TURN:
            if -0.25 < perception.x_mean < 0.25:
                step = self.pattern.current()
                if step and step[0] == 1:
                    self.state = State.ENTER_ROW
                else:
                    self.row_counter = 1
                    self.previous_row = 1
                    self.actual_row = 1
                    self.actual_dist = perception.min_dist
                    self.state = State.COUNTING_ROWS

        # ====================================================
        # >>> STATE: COUNTING_ROWS
        # ====================================================
        elif self.state == State.COUNTING_ROWS:
            step = self.pattern.current()
            if step and step[0] == self.row_counter:
                self.state = State.ENTER_ROW
            else:
                self.actual_row = 1 if perception.num_points_in_box > 0 else 0
                
                # >>> FLANKENERKENNUNG
                if self.actual_row > self.previous_row:
                    self.row_counter += 1

                self.previous_row = self.actual_row

        # ====================================================
        # >>> STATE: ENTER_ROW (Angepasst für Ende-Erkennung)
        # ====================================================
        elif self.state == State.ENTER_ROW:
            if -0.25 < perception.y_mean < 0.25:
                self.pattern.next() # Zum nächsten Schritt im Muster
                
                # Prüfen, ob das Muster beendet ist
                if self.pattern.current() is None:
                    self.state = State.ROBOT_STOP
                    self.node.get_logger().info("PATTERN COMPLETED. Stopping Robot.")
                else:
                    self.state = State.DRIVE_IN_ROW

        if self.state != old_state:
            self.node.get_logger().info(f"State transition: {old_state.name} -> {self.state.name}")

        return self.state


# ============================================================
# >>> CONTROLLER
# ============================================================
class Controller:
    def compute(self, state, perception, direction, params, node):
        cmd = ControlCommand(linear=0.0, angular=0.0)

        if state == State.ROBOT_STOP:
            return cmd
        
        if state == State.DRIVE_IN_ROW:
            if not perception.row_end_detected:
                cmd.angular = -perception.center_error * 5 * params['vel_linear_drive']

                # >>> GESCHWINDIGKEIT + STABILITÄT
                if np.abs(perception.center_error) > 0.15:
                    cmd.linear = 0.1
                else:
                    cmd.linear = params['vel_linear_drive'] * (params['max_dist_in_row'] - np.abs(perception.center_error)) / params['max_dist_in_row']

        elif state == State.EXIT_ROW:
            cmd.linear = params['vel_linear_drive']
            
        elif state == State.TURN:
            if not (-0.25 < perception.x_mean < 0.25):
                cmd.linear = params['vel_linear_turn']
                radius = params['row_width'] / 2.0
                if direction == 'R':
                    radius = -radius
                cmd.angular = params['vel_linear_turn'] / radius

        elif state == State.COUNTING_ROWS:
            gain = 2.5 if direction == 'L' else -2.5
            cmd.linear = params['vel_linear_count']
            if perception.num_points_in_box > 0:
                cmd.angular = gain * (perception.min_dist - params['actual_dist_target'])
            else:
                cmd.angular = 0.0

        elif state == State.ENTER_ROW:
            if not (-0.25 < perception.y_mean < 0.25):
                cmd.linear = params['vel_linear_turn']
                gain = 1 if direction == 'L' else -1
                radius = gain * params['row_width'] / 2.0
                cmd.angular = params['vel_linear_turn'] / radius

        return cmd


# ============================================================
# >>> HAUPTNODE (ALLES KOMMT HIER ZUSAMMEN)
# ============================================================
class FieldRobotNavigator(Node):
    def __init__(self):
        super().__init__("maize_navigator")

        # ====================================================
        # >>> PARAMETER DEKLARATION (ROS2)
        # ====================================================
        self.declare_parameter("pattern", "1L-1R-2L-3R")
        self.declare_parameter("max_dist_in_row", 0.375)
        self.declare_parameter("row_width", 0.75)
        self.declare_parameter("drive_out_dist", 1.0)
        
        self.declare_parameter("vel_linear_drive", 0.5)
        self.declare_parameter("vel_linear_count", 0.5)
        self.declare_parameter("vel_linear_turn", 0.3)

        # >>> PERCEPTION PARAMETER (BOUNDING BOXEN)
        states = ['drive_in_row', 'turn_and_exit', 'counting_rows', 'turn_to_row']
        for s in states:
            self.declare_parameter(f"perception.{s}.x_min", 0.0)
            self.declare_parameter(f"perception.{s}.x_max", 2.0)
            self.declare_parameter(f"perception.{s}.y_min", 0.1)
            self.declare_parameter(f"perception.{s}.y_max", 1.0)

        # >>> ROS TOPICS
        self.declare_parameter("topics.pointcloud", "/merged_point_cloud")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")
        self.declare_parameter("topics.field_points", "/field_points")

        # ====================================================
        # >>> PARAMETER EINLESEN
        # ====================================================
        self.params = self.get_all_params()

        # ====================================================
        # >>> MODULE INITIALISIEREN
        # ====================================================
        self.perception = Perception(self.params['bounding_boxes'])
        self.pattern = Pattern(self.params['pattern'])
        self.state_machine = StateMachine(self.pattern, self)
        self.controller = Controller()

        self.latest_cloud = None

        # ====================================================
        # >>> ROS KOMMUNIKATION
        # ====================================================
        self.create_subscription(LaserScan, self.params['topic_pointcloud'], self.cloud_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.params['topic_cmd_vel'], 10)
        self.points_pub = self.create_publisher(PointCloud2, self.params['topic_field_points'], 10)
        self.start_srv = self.create_service(Trigger, 'start_navigation', self.start_nav_callback)

        # >>> HAUPTLOOP (10 Hz)
        self.timer = self.create_timer(0.1, self.loop)
        
        # >>> DYNAMISCHE PARAMETERÄNDERUNG
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.get_logger().info("FieldRobotNavigator gestartet")

    def parameter_callback(self, params):
        # ====================================================
        # >>> LIVE PARAMETER UPDATE (SEHR PRAKTISCH!)
        # ====================================================
        for param in params:
            keys = param.name.split('.')
            if len(keys) == 1:
                self.params[keys[0]] = param.value
            elif len(keys) == 3 and keys[0] == 'perception':
                self.params['bounding_boxes'][keys[1]][keys[2]] = param.value
                
        return rclpy.parameter.SetParametersResult(successful=True)

    def get_all_params(self):
        # ====================================================
        # >>> SAMMELT ALLE ROS PARAMETER IN EIN DICT
        # ====================================================
        p = {}
        p['pattern'] = self.get_parameter("pattern").value
        p['max_dist_in_row'] = self.get_parameter("max_dist_in_row").value
        p['row_width'] = self.get_parameter("row_width").value
        p['drive_out_dist'] = self.get_parameter("drive_out_dist").value
        p['vel_linear_drive'] = self.get_parameter("vel_linear_drive").value
        p['vel_linear_count'] = self.get_parameter("vel_linear_count").value
        p['vel_linear_turn'] = self.get_parameter("vel_linear_turn").value
        
        # >>> BOUNDING BOXEN
        p['bounding_boxes'] = {}
        states = ['drive_in_row', 'turn_and_exit', 'counting_rows', 'turn_to_row']
        for s in states:
            p['bounding_boxes'][s] = {
                'x_min': self.get_parameter(f"perception.{s}.x_min").value,
                'x_max': self.get_parameter(f"perception.{s}.x_max").value,
                'y_min': self.get_parameter(f"perception.{s}.y_min").value,
                'y_max': self.get_parameter(f"perception.{s}.y_max").value,
            }
            
        p['topic_pointcloud'] = self.get_parameter("topics.pointcloud").value
        p['topic_cmd_vel'] = self.get_parameter("topics.cmd_vel").value
        p['topic_field_points'] = self.get_parameter("topics.field_points").value
        return p

    def cloud_callback(self, msg):
        # >>> SPEICHERT AKTUELLSTE SENSOR DATEN
        self.latest_cloud = msg
    def start_nav_callback(self, request, response):
        """Wird aufgerufen, wenn der ROS Service gerufen wird."""
        if self.state_machine.state == State.ROBOT_STOP:
            self.state_machine.navigation_triggered = True
            response.success = True
            response.message = "Navigation gestartet!"
            self.get_logger().info("Service 'start_navigation' empfangen. Fahre los...")
        else:
            response.success = False
            response.message = f"Roboter ist bereits im Zustand {self.state_machine.state.name}"
        return response

    def publish_points(self, points, header):
        # ====================================================
        # >>> DEBUG VISUALISIERUNG (RViz)
        # ====================================================
        if not points:
            return
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        point_data = [(p.x, p.y, p.z) for p in points]
        
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.latest_cloud.header.frame_id
        cloud = point_cloud2.create_cloud(header, fields, point_data)
        self.points_pub.publish(cloud)

    def loop(self):
        # ====================================================
        # >>> HAUPTABLAUF (PIPELINE)
        # ====================================================
        if self.latest_cloud is None:
            return

        direction = self.state_machine.get_current_direction()
        perception = self.perception.process(self.latest_cloud, self.state_machine.state, direction)
        
        self.publish_points(perception.filtered_points, self.latest_cloud.header)

        state = self.state_machine.update(perception, self.params)
        
        # >>> WICHTIG FÜR COUNTING_ROWS
        self.params['actual_dist_target'] = getattr(self.state_machine, 'actual_dist', np.inf)

        cmd = self.controller.compute(state, perception, direction, self.params, self)

        # >>> BEWEGUNG PUBISHEN
        twist = Twist()
        twist.linear.x = cmd.linear
        twist.angular.z = cmd.angular
        self.cmd_pub.publish(twist)


# ============================================================
# >>> PROGRAMMSTART
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = FieldRobotNavigator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
