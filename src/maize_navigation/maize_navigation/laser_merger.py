import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
import laser_geometry.laser_geometry as lg
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import message_filters # Wichtig für die Synchronisation

class LaserMerger(Node):
    def __init__(self):
        super().__init__('laser_merger')
        self.lp = lg.LaserProjection()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pc_pub = self.create_publisher(PointCloud2, '/sensors/merged_cloud', 10)
        
        # Erstelle Subscriber über message_filters
        self.front_sub = message_filters.Subscriber(self, LaserScan, '/sensors/scan_front')
        self.rear_sub = message_filters.Subscriber(self, LaserScan, '/sensors/scan_rear')
        
        # Synchronisiere die beiden Topics (Zeitfenster ca. 0.1 Sekunde)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.front_sub, self.rear_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_cb)

    def synchronized_cb(self, scan_front, scan_rear):
        try:
            # 1. Beide Scans in PointClouds umwandeln
            cloud_front = self.lp.projectLaser(scan_front)
            cloud_rear = self.lp.projectLaser(scan_rear)
            
            # 2. Transformationen holen
            trans_front = self.tf_buffer.lookup_transform('FloriBot/base_link', scan_front.header.frame_id, rclpy.time.Time())
            trans_rear = self.tf_buffer.lookup_transform('FloriBot/base_link', scan_rear.header.frame_id, rclpy.time.Time())
            
            # 3. Beide in den base_link transformieren
            tf_front = do_transform_cloud(cloud_front, trans_front)
            tf_rear = do_transform_cloud(cloud_rear, trans_rear)
            
            # 4. Daten zusammenfügen (Merge)
            # In Python ist das Zusammenfügen von PointCloud2 etwas mühsam, 
            # am einfachsten ist es, die Datenfelder (data) zu kombinieren, 
            # sofern die Felder (fields) identisch sind:
            merged_cloud = tf_front
            merged_cloud.data += tf_rear.data
            merged_cloud.width += tf_rear.width # Nur bei unorganized clouds korrekt
            merged_cloud.row_step += tf_rear.row_step
            
            self.pc_pub.publish(merged_cloud)
            
        except Exception as e:
            self.get_logger().error(f"Fehler beim Mergen: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LaserMerger()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
