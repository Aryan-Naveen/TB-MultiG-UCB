import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

class LQR(Node):
    def __init__(self):
        super().__init__('lqr')
        self.pose = None
        self.tf = self.create_subscription(
            PoseStamped,
            '/vrpn_client_node/MobileSensor5/pose',
            self.update,
            10)
    def update(self, msg):
        self.pose = msg
        print(msg)

def main(args=None):
    rclpy.init(args=args)

    lqr = LQR()
    rclpy.spin(lqr)


if __name__ == '__main__':
    main()