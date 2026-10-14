"""
ROS 2 AI Controller Node

Loads the trained PyTorch ObstacleAvoidanceMLP model, processes live
LaserScan messages, runs real-time inference, applies a safety watchdog
and an emergency collision shield, and publishes cmd_vel to control the car.
"""

import os
import time
from typing import Optional
import numpy as np
import torch

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

from autonoms_car.model import ObstacleAvoidanceMLP
from autonoms_car.preprocessor import LidarPreprocessor


class AIControllerNode(Node):
    """
    Real-time AI Inference Node with integrated Safety Watchdog.
    """

    def __init__(self):
        super().__init__('ai_controller_node')

        # Declare ROS 2 parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('model_path', 'models/obstacle_avoidance_mlp.pth')
        self.declare_parameter('max_linear_vel', 0.35)
        self.declare_parameter('max_angular_vel', 1.20)
        self.declare_parameter('emergency_stop_dist', 0.30)
        self.declare_parameter('watchdog_timeout', 0.35)
        self.declare_parameter('num_sectors', 36)
        self.declare_parameter('range_max', 5.0)
        self.declare_parameter('use_cuda', True)

        # Retrieve parameters
        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        model_path = self.get_parameter('model_path').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.emergency_stop_dist = self.get_parameter('emergency_stop_dist').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout').value
        num_sectors = self.get_parameter('num_sectors').value
        range_max = self.get_parameter('range_max').value
        use_cuda = self.get_parameter('use_cuda').value

        # Select Compute Device
        self.device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'

        # Load Neural Network Model
        self.get_logger().info(f'Loading AI model weights from: {model_path} on {self.device}...')
        if not os.path.exists(model_path):
            self.get_logger().error(
                f"Model file '{model_path}' not found! "
                f"Please run the training script first: 'python3 -m autonoms_car.train'"
            )
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        self.model = ObstacleAvoidanceMLP.load_from_checkpoint(model_path, device=self.device)
        self.get_logger().info('Neural network model successfully loaded!')

        # Sensor Preprocessor
        self.preprocessor = LidarPreprocessor(
            num_sectors=num_sectors,
            range_max=range_max
        )

        # ROS 2 Publisher and Subscriber
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10
        )

        # Safety Watchdog State
        self.last_scan_time = time.time()
        self.watchdog_timer = self.create_timer(0.05, self.watchdog_callback)  # 20 Hz health check

        self.get_logger().info(
            f'AI Controller Node active.\n'
            f'  Subscribing to:    {scan_topic}\n'
            f'  Publishing to:     {cmd_vel_topic}\n'
            f'  Emergency Shield:  {self.emergency_stop_dist:.2f} m\n'
            f'  Watchdog Timeout:  {self.watchdog_timeout:.2f} s'
        )

    def scan_callback(self, msg: LaserScan):
        """Processes incoming LiDAR scan and publishes AI velocity command."""
        self.last_scan_time = time.time()

        cleaned_ranges = self.preprocessor.clean_ranges(msg.ranges)
        sector_distances = self.preprocessor.sector_min_pool(cleaned_ranges)
        normalized_features = self.preprocessor.normalize(sector_distances)

        # 1. Neural Network Inference
        pred_v, pred_w = self.model.predict_single(normalized_features, device=self.device)

        # 2. Safety Shield (Emergency Reflex Override)
        left_d, front_d, right_d = self.preprocessor.extract_zones(sector_distances)
        safety_override = False

        if front_d <= self.emergency_stop_dist:
            # Emergency brake override
            pred_v = 0.0
            # Steer away from closest obstacle
            pred_w = self.max_angular_vel if left_d >= right_d else -self.max_angular_vel
            safety_override = True

        # 3. Velocity Clamping
        safe_v = float(np.clip(pred_v, 0.0, self.max_linear_vel))
        safe_w = float(np.clip(pred_w, -self.max_angular_vel, self.max_angular_vel))

        # 4. Publish Command
        cmd = Twist()
        cmd.linear.x = safe_v
        cmd.angular.z = safe_w
        self.cmd_pub.publish(cmd)

        if safety_override:
            self.get_logger().warn(
                f'[Emergency Shield] Obstacle at {front_d:.2f}m! '
                f'Braking and turning ({pred_w:.2f} rad/s)',
                throttle_duration_sec=1.0
            )

    def watchdog_callback(self):
        """Monitors sensor liveliness and commands zero velocity if data stalls."""
        elapsed = time.time() - self.last_scan_time
        if elapsed > self.watchdog_timeout:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.cmd_pub.publish(stop_cmd)
            self.get_logger().warn(
                f'[Watchdog] Sensor data stale ({elapsed:.2f}s > {self.watchdog_timeout:.2f}s). '
                f'Vehicle stopped.',
                throttle_duration_sec=2.0
            )

    def stop_robot(self):
        """Gracefully halts the vehicle upon shutdown."""
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_pub.publish(stop_cmd)
        self.get_logger().info('AI Controller stopped. Zero velocities published.')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AIControllerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, FileNotFoundError):
        pass
    finally:
        if 'node' in locals():
            node.stop_robot()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
