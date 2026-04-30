import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import laser_geometry.laser_geometry as lg
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import message_filters
import numpy as np

class LaserMerger(Node):
    def __init__(self):
        super().__init__('laser_merger')
        self.lp = lg.LaserProjection()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pc_pub = self.create_publisher(PointCloud2, '/sensors/merged_cloud', 10)
        
        self.front_sub = message_filters.Subscriber(self, LaserScan, '/sensors/scan_front')
        self.rear_sub = message_filters.Subscriber(self, LaserScan, '/sensors/scan_rear')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.front_sub, self.rear_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_cb)

    def synchronized_cb(self, scan_front, scan_rear):
        try:
            # 1. Projektion in PointCloud2 (im jeweiligen Sensor-Frame)
            cloud_front = self.lp.projectLaser(scan_front)
            cloud_rear = self.lp.projectLaser(scan_rear)
            
            # 2. Transformation mit korrektem Zeitstempel (Temporal Alignment)
            # Wir nutzen die Zeit der Nachricht, um Bewegungsfehler zu vermeiden
            stamp = scan_front.header.stamp 
            
            trans_front = self.tf_buffer.lookup_transform(
                'base_link', cloud_front.header.frame_id, stamp, 
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            trans_rear = self.tf_buffer.lookup_transform(
                'base_link', cloud_rear.header.frame_id, stamp,
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            tf_front = do_transform_cloud(cloud_front, trans_front)
            tf_rear = do_transform_cloud(cloud_rear, trans_rear)
            
            # 3. Schnelles Mergen ohne manuelle Loops
            # Wir extrahieren die Rohdaten (Typ: uint8 array)
            merged_data = tf_front.data + tf_rear.data
            
            # 4. Neue Nachricht zusammenbauen
            merged_cloud = PointCloud2()
            merged_cloud.header = tf_front.header
            merged_cloud.header.frame_id = 'base_link'
            merged_cloud.height = 1
            merged_cloud.width = tf_front.width + tf_rear.width
            merged_cloud.fields = tf_front.fields
            merged_cloud.is_bigendian = tf_front.is_bigendian
            merged_cloud.point_step = tf_front.point_step
            merged_cloud.row_step = merged_cloud.point_step * merged_cloud.width
            merged_cloud.is_dense = tf_front.is_dense and tf_rear.is_dense
            merged_cloud.data = merged_data
            
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