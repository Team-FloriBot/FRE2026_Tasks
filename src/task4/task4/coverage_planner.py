#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

# ROS 2 Messages
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# TF Transformations (wurde in deinem Dockerfile bereits installiert)
from tf_transformations import quaternion_from_euler

# Fields2Cover
import fields2cover as f2c

class CoveragePlanner(Node):
    def __init__(self):
        super().__init__('coverage_planner')

        # 1. Parameter deklarieren
        self.declare_parameter('polygon_coords', [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0])
        self.declare_parameter('operating_width', 0.5)
        self.declare_parameter('robot_width', 0.6)
        self.declare_parameter('headland_width', 1.5)
        self.declare_parameter('turn_radius', 1.0)
        self.declare_parameter('swath_angle_deg', 0.0)
        self.declare_parameter('routing_pattern', 'boustrophedon')
        self.declare_parameter('frame_id', 'map')

        # 2. Publisher für den finalen Pfad
        self.path_pub = self.create_publisher(Path, 'plan', 10)

        # 3. Planung direkt beim Start ausführen (kann später auch in einen Service / Action ausgelagert werden)
        self.generate_coverage_path()

    def generate_coverage_path(self):
        # Parameter auslesen
        coords = self.get_parameter('polygon_coords').get_parameter_value().double_array_value
        op_width = self.get_parameter('operating_width').get_parameter_value().double_value
        rob_width = self.get_parameter('robot_width').get_parameter_value().double_value
        hl_width = self.get_parameter('headland_width').get_parameter_value().double_value
        turn_rad = self.get_parameter('turn_radius').get_parameter_value().double_value
        swath_angle = math.radians(self.get_parameter('swath_angle_deg').get_parameter_value().double_value)
        pattern = self.get_parameter('routing_pattern').get_parameter_value().string_value

        self.get_logger().info(f"Starte Path Planning für Task 4 (Pattern: {pattern}, Arbeitsbreite: {op_width}m)")

        if len(coords) % 2 != 0 or len(coords) < 6:
            self.get_logger().error("polygon_coords muss eine gerade Anzahl an Elementen (x, y) besitzen und mindestens ein Dreieck bilden!")
            return

        # --- Fields2Cover Pipeline ---
        try:
            # A) Feldgrenzen definieren (LinearRing)
            ring = f2c.LinearRing()
            for i in range(0, len(coords), 2):
                ring.addPoint(coords[i], coords[i+1])
            
            # WICHTIG: In Fields2Cover muss das Polygon explizit geschlossen werden
            ring.addPoint(coords[0], coords[1])
            
            cell = f2c.Cell(ring)
            cells = f2c.Cells(cell)

            # B) Roboter-Eigenschaften festlegen
            robot = f2c.Robot(rob_width, op_width)
            robot.setMinRadius(turn_rad)

            # C) Vorgewende (Headland) generieren
            hl_gen = f2c.HG_Const_hl()
            no_hl = hl_gen.generateHeadlands(cells, hl_width)

            # D) Fahrgassen (Swaths) im inneren Feld (ohne Vorgewende) generieren
            sg = f2c.SG_BruteForce()
            # no_hl.getGeometry(0) liefert das nutzbare innere Feld
            swaths = sg.generateSwaths(swath_angle, op_width, no_hl.getGeometry(0))

            # E) Routenplanung (Reihenfolge der Gassen)
            if pattern == 'snake':
                rp = f2c.RP_Snake()
            else:
                rp = f2c.RP_Boustrophedon()
            
            route = rp.genRoute(swaths)

            # F) Pfadplanung (Verbindung der Gassen mit realistischen Wendemanövern)
            pp = f2c.PP_PathPlanning()
            dubins = f2c.PT_Dubins()
            # PlanPath fügt Dubins-Kurven für realistische Roboterklimatiken hinzu
            f2c_path = pp.planPath(robot, route, dubins)

            self.get_logger().info("Fields2Cover Pfad erfolgreich berechnet! Konvertiere zu ROS Path...")
            self.publish_ros_path(f2c_path)

        except Exception as e:
            self.get_logger().error(f"Fehler bei der Fields2Cover Berechnung: {e}")

    def publish_ros_path(self, f2c_path):
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        
        ros_path = Path()
        ros_path.header.frame_id = frame_id
        ros_path.header.stamp = self.get_clock().now().to_msg()

        # f2c_path.states enthält alle Wegpunkte (X, Y, Theta)
        for state in f2c_path.states:
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = state.point.getX()
            pose.pose.position.y = state.point.getY()
            pose.pose.position.z = 0.0
            
            # Yaw (Theta) in Quaternion umwandeln, damit Nav2/ROS damit arbeiten kann
            q = quaternion_from_euler(0, 0, state.angle)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            
            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)
        self.get_logger().info(f"Pfad mit {len(ros_path.poses)} Wegpunkten publiziert auf Topic '/plan'.")

def main(args=None):
    rclpy.init(args=args)
    node = CoveragePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()