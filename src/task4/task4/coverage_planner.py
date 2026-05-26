import rclpy
from rclpy.node import Node
import fields2cover as f2c

class CoverageNode(Node):
    def __init__(self):
        super().__init__('coverage_node')
        
        # Ein einfaches Fields2Cover-Objekt erstellen, um den Import zu testen
        pt = f2c.Point(1.5, 2.5)
        self.get_logger().info(f'Fields2Cover Punkt erfolgreich erstellt bei: X={pt.getX():.2f}, Y={pt.getY():.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = CoverageNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()