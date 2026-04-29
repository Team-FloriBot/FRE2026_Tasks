import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
import laser_geometry.laser_geometry as lg
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

class LaserMerger(Node):
    def __init__(self):
        super().__init__('laser_merger')
        self.lp = lg.LaserProjection()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pc_pub = self.create_publisher(PointCloud2, '/sensors/merged_cloud', 10)
        
        self.create_subscription(LaserScan, '/sensors/scan_front', self.scan_cb, 10)
        self.create_subscription(LaserScan, '/sensors/scan_rear', self.scan_cb, 10)

    def scan_cb(self, msg):
        try:
            # Transformiere Scan direkt in den base_link
            # Die "Auf dem Kopf" Info kommt automatisch aus der URDF/TF
            trans = self.tf_buffer.lookup_transform('FloriBot/base_link', msg.header.frame_id, rclpy.time.Time())
            cloud_out = self.lp.projectLaser(msg)
            res_cloud = do_transform_cloud(cloud_out, trans)
            self.pc_pub.publish(res_cloud)
        except Exception as e:
            self.get_logger().debug(f"TF wait: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LaserMerger()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
