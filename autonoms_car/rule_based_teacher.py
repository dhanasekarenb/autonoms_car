"""
Rule-Based Teacher Controller Node

Serves two critical roles:
1. Operational baseline: Proves that Isaac Sim physics, wheel joints, and LiDAR communication work.
2. Expert supervisor: Generates collision-free driving behavior used to train the neural network.
"""

from typing import Tuple
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


class RuleBasedTeacher:
    """
    Pure algorithmic decision maker (no ROS dependencies inside logic).
    Enables unit testing and offline training data synthesis.
    """

    def __init__(
        self,
        max_linear_vel: float = 0.35,
        max_angular_vel: float = 1.20,
        safe_distance: float = 1.20,
        stop_distance: float = 0.35
    ):
        self.max_linear_vel = float(max_linear_vel)
        self.max_angular_vel = float(max_angular_vel)
        self.safe_distance = float(safe_distance)
        self.stop_distance = float(stop_distance)

    def compute_action(
        self,
        left_dist: float,
        front_dist: float,
        right_dist: float
    ) -> Tuple[float, float]:
        """
        Calculates linear and angular velocities based on zone clearances.

        Args:
            left_dist: Minimum distance in left sector (meters).
            front_dist: Minimum distance in front sector (meters).
            right_dist: Minimum distance in right sector (meters).

        Returns:
            (linear_vel, angular_vel) in (m/s, rad/s).
        """
        # Critical proximity: Stop forward progress and rotate in place
        if front_dist <= self.stop_distance:
            linear_v = 0.0
            # Rotate toward whichever side has more room
            if left_dist >= right_dist:
                angular_w = self.max_angular_vel  # Turn left (counter-clockwise)
            else:
                angular_w = -self.max_angular_vel  # Turn right (clockwise)
            return linear_v, angular_w

        # Approaching obstacle: Scale speed down and steer away
        if front_dist < self.safe_distance:
            # Linear speed scales down smoothly between safe_distance and stop_distance
            ratio = (front_dist - self.stop_distance) / max(self.safe_distance - self.stop_distance, 1e-4)
            linear_v = self.max_linear_vel * np.clip(ratio, 0.1, 1.0)

            # Steering severity inversely proportional to distance
            urgency = 1.0 - ratio  # 0.0 (far) to 1.0 (close)
            turn_direction = 1.0 if left_dist >= right_dist else -1.0
            angular_w = turn_direction * self.max_angular_vel * (0.5 + 0.5 * urgency)
            return float(linear_v), float(angular_w)

        # Clear path ahead: Cruise forward with gentle wall-centering bias
        linear_v = self.max_linear_vel

        # Gentle bias if one side is significantly closer than the other
        side_diff = left_dist - right_dist
        centering_w = np.clip(side_diff * 0.25, -0.3, 0.3)
        angular_w = float(centering_w)

        return float(linear_v), float(angular_w)


class RuleBasedTeacherNode(Node):
    """
    ROS 2 Node subscribing to LaserScan and publishing cmd_vel commands.
    """

    def __init__(self):
        super().__init__('rule_based_teacher_node')

        # Declare ROS 2 parameters with sensible defaults
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('max_linear_vel', 0.35)
        self.declare_parameter('max_angular_vel', 1.20)
        self.declare_parameter('safe_distance', 1.20)
        self.declare_parameter('stop_distance', 0.35)
        self.declare_parameter('num_sectors', 36)
        self.declare_parameter('range_max', 5.0)

        # Retrieve parameters
        scan_topic = self.get_parameter('scan_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        max_linear_vel = self.get_parameter('max_linear_vel').value
        max_angular_vel = self.get_parameter('max_angular_vel').value
        safe_distance = self.get_parameter('safe_distance').value
        stop_distance = self.get_parameter('stop_distance').value
        num_sectors = self.get_parameter('num_sectors').value
        range_max = self.get_parameter('range_max').value

        # Preprocessor and Decision Logic
        self.preprocessor = LidarPreprocessor(
            num_sectors=num_sectors,
            range_max=range_max
        )
        self.teacher = RuleBasedTeacher(
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            safe_distance=safe_distance,
            stop_distance=stop_distance
        )

        # ROS 2 Publisher and Subscriber
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10
        )

        self.get_logger().info(
            f'Rule-Based Teacher active.\n'
            f'  Subscribing to: {scan_topic}\n'
            f'  Publishing to:  {cmd_vel_topic}\n'
            f'  Speed Limits:   v_max={max_linear_vel:.2f} m/s, w_max={max_angular_vel:.2f} rad/s\n'
            f'  Thresholds:     safe={safe_distance:.2f} m, stop={stop_distance:.2f} m'
        )

    def scan_callback(self, msg: LaserScan):
        cleaned_ranges = self.preprocessor.clean_ranges(msg.ranges)
        sector_distances = self.preprocessor.sector_min_pool(cleaned_ranges)
        left_d, front_d, right_d = self.preprocessor.extract_zones(sector_distances)

        v, w = self.teacher.compute_action(left_d, front_d, right_d)

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """Sends immediate zero velocities upon shutdown."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.get_logger().info('Vehicle stopped safely.')


def main(args=None):
    rclpy.init(args=args)
    node = RuleBasedTeacherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
