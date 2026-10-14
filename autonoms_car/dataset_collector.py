"""
Dataset Collector Node

Logs synchronized pairs of normalized LiDAR feature vectors and vehicle
velocity commands (from the rule-based teacher or manual teleoperation)
into a structured .npz file for supervised model training.
"""

import os
from typing import Optional
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

from autonoms_car.preprocessor import LidarPreprocessor


class DatasetCollectorNode(Node):
    """
    Synchronizes incoming LaserScan with the active velocity command,
    preprocesses the scan into fixed-length features, and saves the dataset.
    """

    def __init__(self):
        super().__init__('dataset_collector_node')

        # Declare parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('output_path', 'dataset/training_data.npz')
        self.declare_parameter('num_sectors', 36)
        self.declare_parameter('range_max', 5.0)
        self.declare_parameter('target_samples', 4000)
        self.declare_parameter('ignore_idle', True)

        # Retrieve parameters
        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.output_path = self.get_parameter('output_path').value
        num_sectors = self.get_parameter('num_sectors').value
        range_max = self.get_parameter('range_max').value
        self.target_samples = self.get_parameter('target_samples').value
        self.ignore_idle = self.get_parameter('ignore_idle').value

        # Preprocessor
        self.preprocessor = LidarPreprocessor(
            num_sectors=num_sectors,
            range_max=range_max
        )

        # In-memory storage buffers
        self.features_buffer = []  # List of 1D arrays of size num_sectors
        self.labels_buffer = []    # List of [linear.x, angular.z]

        # Latest received command
        self.latest_cmd = (0.0, 0.0)
        self.has_received_cmd = False

        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.cmd_callback,
            10
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10
        )

        self.get_logger().info(
            f'Dataset Collector started.\n'
            f'  Scan Topic:      {scan_topic}\n'
            f'  Command Topic:   {cmd_vel_topic}\n'
            f'  Output Target:   {self.output_path}\n'
            f'  Target Samples:  {self.target_samples}\n'
            f'Drive the car or run the rule-based teacher to log data.'
        )

    def cmd_callback(self, msg: Twist):
        self.latest_cmd = (float(msg.linear.x), float(msg.angular.z))
        self.has_received_cmd = True

    def scan_callback(self, msg: LaserScan):
        if not self.has_received_cmd:
            return

        linear_v, angular_w = self.latest_cmd

        # Optional: Skip recording when the robot is completely stationary
        if self.ignore_idle and abs(linear_v) < 1e-3 and abs(angular_w) < 1e-3:
            return

        # Preprocess scan into normalized feature vector
        features = self.preprocessor.process(msg.ranges)

        self.features_buffer.append(features)
        self.labels_buffer.append([linear_v, angular_w])

        count = len(self.features_buffer)
        if count % 200 == 0 or count == self.target_samples:
            self.get_logger().info(
                f'Collected {count}/{self.target_samples} samples '
                f'({(count / self.target_samples) * 100:.1f}%)'
            )

        if count >= self.target_samples:
            self.get_logger().info('Target sample count reached. Saving dataset...')
            self.save_dataset()
            rclpy.shutdown()

    def save_dataset(self):
        """Saves recorded buffers to compressed .npz archive."""
        if len(self.features_buffer) == 0:
            self.get_logger().warn('No samples collected; nothing saved.')
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        X = np.array(self.features_buffer, dtype=np.float32)
        Y = np.array(self.labels_buffer, dtype=np.float32)

        np.savez_compressed(
            self.output_path,
            features=X,
            labels=Y
        )

        self.get_logger().info(
            f'Successfully saved {len(X)} samples to {self.output_path}!\n'
            f'  Features shape: {X.shape} (dtype={X.dtype})\n'
            f'  Labels shape:   {Y.shape} (dtype={Y.dtype})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user. Saving collected data...')
    finally:
        node.save_dataset()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
