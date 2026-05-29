#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

# ROS 2 Services & Messages
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path

# TF2 für die Transformationen
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from tf_transformations import quaternion_from_euler

# Fields2Cover
import fields2cover as f2c

class CoveragePlanner(Node):
    def __init__(self):
        super().__init__('coverage_planner')

        # 1. Parameter deklarieren
        self.declare_parameter('polygon_coords', [0.0, -5.0, 10.0, -5.0, 10.0, 5.0, 0.0, 5.0])
        self.declare_parameter('operating_width', 0.5)
        self.declare_parameter('robot_width', 0.6)
        self.declare_parameter('headland_width', 1.5)
        self.declare_parameter('turn_radius', 1.0)
        self.declare_parameter('swath_angle_deg', 0.0)
        self.declare_parameter('routing_pattern', 'boustrophedon')
        
        # input_frame: Wo eure Koordinaten herkommen (Roboter)
        # target_frame: Das weltfeste System (Odom oder Map)
        self.declare_parameter('input_frame', 'base_link')
        self.declare_parameter('target_frame', 'odom')

        # 2. TF2 Setup initialisieren
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 3. Publisher für den finalen Pfad
        self.path_pub = self.create_publisher(Path, 'plan', 10)

        # 4. Service erstellen (Wartet passiv auf den Startknopf/Aufruf)
        self.srv = self.create_service(
            Trigger, 
            'trigger_coverage_planning', 
            self.planning_service_callback
        )
        self.get_logger().info("Coverage Planner Service '/trigger_coverage_planning' ist bereit!")

    def planning_service_callback(self, request, response):
        input_frame = self.get_parameter('input_frame').get_parameter_value().string_value
        target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        try:
            # Aktuellste Transformation zwischen base_link und odom/map holen
            transform = self.tf_buffer.lookup_transform(
                target_frame, input_frame, rclpy.time.Time()
            )
            
            self.get_logger().info(f"TF von {input_frame} nach {target_frame} erfolgreich geholt. Starte Pfadplanung...")
            self.generate_coverage_path(transform)
            
            response.success = True
            response.message = f"Pfadplanung erfolgreich in '{target_frame}' berechnet und publiziert."
        except Exception as e:
            response.success = False
            response.message = f"Fehler bei der Transformation oder Fields2Cover-Planung: {str(e)}"
            self.get_logger().error(response.message)
        
        return response

    def generate_coverage_path(self, transform):
        # Parameter auslesen
        coords = self.get_parameter('polygon_coords').get_parameter_value().double_array_value
        op_width = self.get_parameter('operating_width').get_parameter_value().double_value
        rob_width = self.get_parameter('robot_width').get_parameter_value().double_value
        hl_width = self.get_parameter('headland_width').get_parameter_value().double_value
        turn_rad = self.get_parameter('turn_radius').get_parameter_value().double_value
        swath_angle = math.radians(self.get_parameter('swath_angle_deg').get_parameter_value().double_value)
        pattern = self.get_parameter('routing_pattern').get_parameter_value().string_value
        target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        if len(coords) % 2 != 0 or len(coords) < 6:
            self.get_logger().error("polygon_coords muss eine gerade Anzahl an Elementen besitzen!")
            return

        # --- 1. SCHRITT: Transformation der Koordinaten von base_link nach odom ---
        transformed_coords = []
        for i in range(0, len(coords), 2):
            point_in = PointStamped()
            point_in.header.frame_id = self.get_parameter('input_frame').get_parameter_value().string_value
            point_in.point.x = coords[i]
            point_in.point.y = coords[i+1]
            point_in.point.z = 0.0

            # Punkt ins globale System transformieren
            point_out = do_transform_point(point_in, transform)
            transformed_coords.append((point_out.point.x, point_out.point.y))

# --- 2. SCHRITT: Fields2Cover Pipeline (Exakte API-Anpassung) ---
        try:
            ring = f2c.LinearRing()
            for pt in transformed_coords:
                ring.addPoint(pt[0], pt[1])
            
            # Polygon explizit mit dem ersten transformierten Punkt schließen
            ring.addPoint(transformed_coords[0][0], transformed_coords[0][1])
            
            cell = f2c.Cell(ring)
            cells = f2c.Cells(cell)

            robot = f2c.Robot(rob_width, op_width)
            robot.max_curv = 1.0 / turn_rad

            # Vorgewende abgrenzen 
            hl_gen = f2c.HG_Const_gen()
            no_hl = hl_gen.generateHeadlands(cells, hl_width)

            # Gassen generieren
            sg = f2c.SG_BruteForce()
            swaths = sg.generateSwaths(swath_angle, op_width, no_hl.getCell(0))

            # Routing Muster bestimmen & Gassen sortieren
            if pattern == 'snake':
                rp = f2c.RP_Snake()
            else:
                rp = f2c.RP_Boustrophedon()
            
            sorted_swaths = rp.genSortedSwaths(swaths)

            # Kinematische Pfadplanung mit Dubins-Kurven
            pp = f2c.PP_PathPlanning()
            
            # KORREKTUR: PP_DubinsCurves statt PT_Dubins
            dubins = f2c.PP_DubinsCurves()
            
            # Pfad generieren
            f2c_path = pp.planPath(robot, sorted_swaths, dubins)

            # Pfad publizieren
            self.publish_ros_path(f2c_path, target_frame)

        except Exception as e:
            raise RuntimeError(f"Fields2Cover Kern-Fehler: {e}")

    def publish_ros_path(self, f2c_path, frame_id):
        ros_path = Path()
        ros_path.header.frame_id = frame_id
        ros_path.header.stamp = self.get_clock().now().to_msg()

        for state in f2c_path.states:
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = state.point.getX()
            pose.pose.position.y = state.point.getY()
            pose.pose.position.z = 0.0
            
            # Orientierung (Yaw) berechnen
            q = quaternion_from_euler(0, 0, state.angle)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            
            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)
        self.get_logger().info(f"Pfad mit {len(ros_path.poses)} Wegpunkten erfolgreich auf /plan publiziert!")

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