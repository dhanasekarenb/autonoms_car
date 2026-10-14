"""
Mock LiDAR Publisher Node

Generates simulated sensor_msgs/LaserScan data for testing both the rule-based
teacher and the AI controller node completely offline without running Isaac Sim.
"""

import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object
    LaserScan = object
    Twist = object


class MockLidarNode(Node):
    """
    Publishes synthetic LaserScan messages with dynamic obstacle scenarios.
    Also logs received /cmd_vel commands to observe vehicle control responses.
    """

    def __init__(self):
        super().__init__('mock_lidar_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('scenario_cycle_sec', 6.0)

        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        publish_rate = self.get_parameter('publish_rate').value
        self.scenario_cycle_sec = self.get_parameter('scenario_cycle_sec').value

        # Publisher and Subscriber
        self.publisher = self.create_publisher(LaserScan, scan_topic, 10)
        self.cmd_sub = self.create_subscription(Twist, cmd_vel_topic, self.cmd_callback, 10)

        # Timer
        period = 1.0 / max(publish_rate, 1.0)
        self.timer = self.create_timer(period, self.publish_scan)

        self.start_time = self.get_clock().now()
        self.step_count = 0

        self.get_logger().info(
            f'Mock LiDAR Node started at {publish_rate} Hz.\n'
            f'  Publishing /scan on: {scan_topic}\n'
            f'  Listening /cmd_vel on: {cmd_vel_topic}\n'
            f'  Cycling scenarios every {self.scenario_cycle_sec} seconds.'
        )

    def cmd_callback(self, msg: Twist):
        # Log received control commands at low frequency
        if self.step_count % 10 == 0:
            self.get_logger().info(
                f'[Feedback cmd_vel] Linear v = {msg.linear.x:+.2f} m/s | '
                f'Angular w = {msg.angular.z:+.2f} rad/s'
            )

    def publish_scan(self):
        self.step_count += 1
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        # Determine active scenario based on elapsed time
        phase = int((elapsed / self.scenario_cycle_sec) % 4)

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'lidar_link'

        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        num_rays = 360
        msg.angle_increment = (msg.angle_max - msg.angle_min) / float(num_rays)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.10
        msg.range_max = 10.00

        # Base scan: open space at 5.0m
        ranges = np.full(num_rays, 5.0, dtype=np.float32)
        center = num_rays // 2  # 180 is 0.0 radians (front)

        scenario_name = ""
        if phase == 0:
            scenario_name = "1. Clear Corridor (Cruise forward)"
            # Open space ahead

        elif phase == 1:
            scenario_name = "2. Approaching Front Wall (Should brake & steer away)"
            # Obstacle directly ahead at 0.8m
            ranges[center - 30 : center + 30] = 0.8
            # Right side is tighter (0.9m), Left side is open (4.0m) -> Should steer left
            ranges[center - 60 : center - 30] = 0.9

        elif phase == 2:
            scenario_name = "3. Obstacle on Left (Should steer right)"
            ranges[center : center + 60] = 0.6

        elif phase == 3:
            scenario_name = "4. Emergency Critical Obstacle (Should trigger safety shield)"
            ranges[center - 25 : center + 25] = 0.22  # Below emergency stop threshold (0.30m)

        # Add minor gaussian sensor noise
        noise = np.random.normal(0.0, 0.015, size=num_rays).astype(np.float32)
        ranges = np.clip(ranges + noise, msg.range_min, msg.range_max)

        msg.ranges = ranges.tolist()
        self.publisher.publish(msg)

        if self.step_count % 20 == 0:
            self.get_logger().info(f"[Mock Scenario] {scenario_name}")


def main(args=None):
    rclpy.init(args=args)
    node = MockLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
