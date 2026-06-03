#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import math
from numbers import Real
from threading import Event

# ROS 2 Services & Messages
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from rcl_interfaces.msg import SetParametersResult

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
        
        self.declare_parameter('input_frame', 'base_link')
        self.declare_parameter('target_frame', 'odom')

        # Flag, ob polygon_coords per Service gesetzt wurden
        self.service_coords_set = False

        self.task4_service_group = ReentrantCallbackGroup()

        # 2. TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 3. Publisher
        self.path_pub = self.create_publisher(Path, 'plan', 10)
        self.marker_pub = self.create_publisher(Marker, 'coverage_polygon_marker', 10)

        # 4. Services
        self.set_params_handler = self.add_on_set_parameters_callback(self.on_parameter_change)

        self.srv = self.create_service(
            Trigger, 
            'trigger_coverage_planning', 
            self.planning_service_callback
        )
        self.get_logger().info("Coverage Planner Service '/trigger_coverage_planning' ist bereit!")

        self.path_tracking_active_client = self.create_client(
            SetBool,
            '/pure_pursuit_node/set_active',
            callback_group=self.task4_service_group
        )
        self.start_navigation_srv = self.create_service(
            Trigger,
            '/task4/start_navigation',
            self.start_navigation_callback,
            callback_group=self.task4_service_group
        )
        self.stop_navigation_srv = self.create_service(
            Trigger,
            '/task4/stop_navigation',
            self.stop_navigation_callback,
            callback_group=self.task4_service_group
        )
        self.get_logger().info("Task-4 Services '/task4/start_navigation' und '/task4/stop_navigation' sind bereit!")

    def on_parameter_change(self, params):
        polygon_coords_were_set = False

        for p in params:
            if p.name == 'polygon_coords':
                success, reason = self.validate_polygon_coords(p.value)
                if not success:
                    return SetParametersResult(successful=False, reason=reason)
                polygon_coords_were_set = True

        if polygon_coords_were_set:
            self.service_coords_set = True
            self.get_logger().info("Polygon-Koordinaten erfolgreich per Service gesetzt.")

        return SetParametersResult(successful=True)

    def set_path_tracking_active(self, active):
        if not self.path_tracking_active_client.service_is_ready():
            self.path_tracking_active_client.wait_for_service(timeout_sec=0.5)

        if not self.path_tracking_active_client.service_is_ready():
            return False, "Service '/pure_pursuit_node/set_active' ist nicht erreichbar."

        request = SetBool.Request()
        request.data = bool(active)
        future = self.path_tracking_active_client.call_async(request)
        completed = Event()
        future.add_done_callback(lambda _: completed.set())

        if not completed.wait(timeout=2.0):
            return False, "Timeout beim Aufruf von '/pure_pursuit_node/set_active'."

        try:
            service_response = future.result()
        except Exception as exc:
            return False, f"Serviceaufruf fehlgeschlagen: {exc}"

        return bool(service_response.success), service_response.message

    def start_navigation_callback(self, request, response):
        success, message = self.set_path_tracking_active(True)
        response.success = success
        response.message = message or ("Task-4-Navigation gestartet." if success else "Task-4-Navigation konnte nicht gestartet werden.")
        return response

    def stop_navigation_callback(self, request, response):
        success, message = self.set_path_tracking_active(False)
        response.success = success
        response.message = message or ("Task-4-Navigation gestoppt." if success else "Task-4-Navigation konnte nicht gestoppt werden.")
        return response

    def planning_service_callback(self, request, response):
        input_frame = self.get_parameter('input_frame').get_parameter_value().string_value
        target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        if not self.service_coords_set:
            self.get_logger().warn(
                "Keine Polygon-Koordinaten per Service gesetzt. Verwende Default aus der Parameterdatei."
            )

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, input_frame, rclpy.time.Time()
            )
            self.get_logger().info(f"TF von {input_frame} nach {target_frame} geholt. Starte Pfadplanung...")
            
            # Aufruf der Planungs-Pipeline
            self.generate_coverage_path(transform)
            
            response.success = True
            response.message = f"Pfadplanung erfolgreich in '{target_frame}' berechnet und publiziert."
        except Exception as e:
            response.success = False
            response.message = f"Fehler bei der Transformation oder Fields2Cover-Planung: {str(e)}"
            self.get_logger().error(response.message)
        
        return response

    def validate_polygon_coords(self, coords):
        if (
            isinstance(coords, (str, bytes))
            or not hasattr(coords, '__iter__')
            or not hasattr(coords, '__len__')
        ):
            return False, "polygon_coords muss eine Liste aus Zahlen sein."

        coords = list(coords)

        if len(coords) % 2 != 0:
            return False, "polygon_coords muss eine gerade Anzahl an Elementen besitzen."

        if len(coords) < 6:
            return False, "polygon_coords muss mindestens drei Punkte enthalten."

        for value in coords:
            if isinstance(value, bool) or not isinstance(value, Real):
                return False, "polygon_coords darf nur numerische Werte enthalten."
            if not math.isfinite(float(value)):
                return False, "polygon_coords darf keine NaN- oder Infinity-Werte enthalten."

        points = self.coords_to_points(coords)
        if len(points) > 1 and self.same_point(points[0], points[-1]):
            points = points[:-1]

        if len(points) < 3:
            return False, "polygon_coords muss mindestens drei unterschiedliche Polygonpunkte enthalten."

        if len(set(points)) < 3:
            return False, "polygon_coords muss mindestens drei unterschiedliche Polygonpunkte enthalten."

        if math.isclose(self.signed_area_twice(points), 0.0, abs_tol=1e-9):
            return False, "polygon_coords darf kein degeneriertes Polygon bilden."

        return True, ""

    def get_polygon_coords(self):
        coords = list(self.get_parameter('polygon_coords').value)
        success, reason = self.validate_polygon_coords(coords)
        if not success:
            raise ValueError(reason)

        points = self.coords_to_points(coords)
        if len(points) > 1 and self.same_point(points[0], points[-1]):
            points = points[:-1]

        return [coord for point in points for coord in point]

    def coords_to_points(self, coords):
        return [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]

    def same_point(self, first, second):
        return (
            math.isclose(first[0], second[0], abs_tol=1e-9)
            and math.isclose(first[1], second[1], abs_tol=1e-9)
        )

    def signed_area_twice(self, points):
        area = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return area

    def generate_coverage_path(self, transform):
        coords = self.get_polygon_coords()
        op_width = self.get_parameter('operating_width').get_parameter_value().double_value
        rob_width = self.get_parameter('robot_width').get_parameter_value().double_value
        hl_width = self.get_parameter('headland_width').get_parameter_value().double_value
        turn_rad = self.get_parameter('turn_radius').get_parameter_value().double_value
        target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        # 1. Transformation nach odom/map
        transformed_coords = []
        for i in range(0, len(coords), 2):
            point_in = PointStamped()
            point_in.header.frame_id = self.get_parameter('input_frame').get_parameter_value().string_value
            point_in.point.x = coords[i]
            point_in.point.y = coords[i+1]
            point_in.point.z = 0.0

            point_out = do_transform_point(point_in, transform)
            transformed_coords.append((point_out.point.x, point_out.point.y))

        self.publish_polygon_marker(transformed_coords, target_frame)

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y
        
        # Fields2Cover Point für den echten Start des Graphen erstellen
        robot_start_point = f2c.Point(robot_x, robot_y)

        # 2. Fields2Cover Pipeline
        try:
            ring = f2c.LinearRing()
            for pt in transformed_coords:
                ring.addPoint(pt[0], pt[1])
            ring.addPoint(transformed_coords[0][0], transformed_coords[0][1])
            
            cell = f2c.Cell(ring)
            cells = f2c.Cells(cell)

            robot = f2c.Robot(rob_width, op_width)
            robot.setMinTurningRadius(turn_rad)

            hl_gen = f2c.HG_Const_gen()
            route_hl_width = hl_width / 2.0
            mid_hl = hl_gen.generateHeadlands(cells, route_hl_width)
            no_hl = hl_gen.generateHeadlands(cells, hl_width)

            sg = f2c.SG_BruteForce()
            swath_objective = f2c.OBJ_NSwath()
            swaths = sg.generateBestSwaths(swath_objective, robot.getCovWidth(), no_hl.getGeometry(0))

            # --- NEU: Globale Graphen-Routenplanung (TSP) ---
            route_planner = f2c.RP_RoutePlannerBase()
            
            # Startpunkt nativ im Optimierer verankern
            route_planner.setStartAndEndPoint(robot_start_point)

            # SWIG-Sicherheit: Swaths in den geforderten Zellen-Container packen
            swaths_by_cells = f2c.SwathsByCells()
            try:
                swaths_by_cells.append(swaths)
            except AttributeError:
                swaths_by_cells.push_back(swaths)

            # Optimierte Route berechnen (Reihenfolge & Richtungen im Graphen optimiert)
            route = route_planner.genRoute(mid_hl, swaths_by_cells)

            # --- Kinematische Pfadplanung ---
            pp = f2c.PP_PathPlanning()
            dubins = f2c.PP_DubinsCurves()
            
            # Wichtig: Überladung von planPath nutzt jetzt das 'route'-Objekt
            f2c_path = pp.planPath(robot, route, dubins)

            self.publish_ros_path(f2c_path, target_frame)

        except Exception as e:
            raise RuntimeError(f"{e}")

    def publish_ros_path(self, f2c_path, frame_id):
        ros_path = Path()
        ros_path.header.frame_id = frame_id
        ros_path.header.stamp = self.get_clock().now().to_msg()

        states = []
        if hasattr(f2c_path, 'states'):
            states = f2c_path.states
        elif hasattr(f2c_path, 'getStates'):
            states = f2c_path.getStates()
        else:
            states = f2c_path

        try:
            for state in states:
                pose = PoseStamped()
                pose.header = ros_path.header
                pose.pose.position.x = state.point.getX()
                pose.pose.position.y = state.point.getY()
                pose.pose.position.z = 0.0
                
                q = quaternion_from_euler(0, 0, state.angle)
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                
                ros_path.poses.append(pose)
        except Exception as e:
            self.get_logger().error(f"Fehler beim Parsen der Pfad-Zustände: {e}")
            if len(states) > 0:
                self.get_logger().info(f"Verfügbare Attribute im Zustand: {dir(states[0])}")
            return

        self.path_pub.publish(ros_path)
        self.get_logger().info(f"Pfad mit {len(ros_path.poses)} Wegpunkten erfolgreich auf /plan publiziert!")

    def publish_polygon_marker(self, coords, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "coverage_polygon"
        marker.id = 0
        
        # LINE_STRIP verbindet die Punkte mit Linien
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Linienbreite
        marker.scale.x = 0.05

        # Farbe (z.B. ein gut sichtbares Blau)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0  # Wichtig: Alpha muss > 0 sein, sonst ist der Marker unsichtbar

        # Punkte zum Marker hinzufügen
        for pt in coords:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.0
            marker.points.append(p)

        # Das Polygon schließen, indem der erste Punkt am Ende nochmal angehängt wird
        if len(coords) > 0:
            p = Point()
            p.x = float(coords[0][0])
            p.y = float(coords[0][1])
            p.z = 0.0
            marker.points.append(p)

        self.marker_pub.publish(marker)
        self.get_logger().info("Coverage Polygon Marker publiziert!")

def main(args=None):
    rclpy.init(args=args)
    node = CoveragePlanner()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
