"""
Lidar Preprocessor Module

Transforms raw sensor_msgs/LaserScan range measurements into a clean,
fixed-size, normalized feature vector suitable for both rule-based
decision making and neural network inference.
"""

from typing import List, Tuple, Union, Optional
import numpy as np


class LidarPreprocessor:
    """
    Processes variable-length LaserScan range arrays into fixed-size,
    normalized obstacle distance vectors using sector min-pooling.
    """

    def __init__(
        self,
        num_sectors: int = 36,
        range_min: float = 0.10,
        range_max: float = 5.00,
        fov_deg: float = 360.0,
        invert_normalization: bool = False
    ):
        """
        Args:
            num_sectors: Number of discrete angular bins (default 36, 10 deg each for 360 FOV).
            range_min: Minimum reliable distance threshold in meters.
            range_max: Maximum distance to consider in meters (clips farther readings).
            fov_deg: Field of view covered by the LiDAR in degrees (typically 360 or 240).
            invert_normalization:
                If False: 0.0 = obstacle at range_min (danger), 1.0 = clear at range_max.
                If True:  1.0 = obstacle close (danger), 0.0 = completely clear.
        """
        self.num_sectors = int(num_sectors)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.fov_deg = float(fov_deg)
        self.invert_normalization = invert_normalization

    def clean_ranges(
        self,
        raw_ranges: Union[List[float], np.ndarray],
        msg_range_min: Optional[float] = None,
        msg_range_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Cleans raw LaserScan measurements by replacing invalid, infinite,
        or NaN readings with valid bounds.

        - inf (no obstacle hit within range) -> range_max (clear space)
        - nan (sensor error/reflection failure) -> range_min (assumed conservative obstacle)
        - values < range_min -> range_min
        - values > range_max -> range_max
        """
        ranges = np.array(raw_ranges, dtype=np.float32)

        # Replace NaNs with safe conservative minimum
        nan_mask = np.isnan(ranges)
        ranges[nan_mask] = self.range_min

        # Replace positive infinity (open space) with configured max distance
        inf_mask = np.isinf(ranges) & (ranges > 0)
        ranges[inf_mask] = self.range_max

        # Replace negative infinity with minimum
        neg_inf_mask = np.isinf(ranges) & (ranges < 0)
        ranges[neg_inf_mask] = self.range_min

        # Clip all values to [range_min, range_max]
        clipped = np.clip(ranges, self.range_min, self.range_max)
        return clipped

    def sector_min_pool(self, cleaned_ranges: np.ndarray) -> np.ndarray:
        """
        Downsamples an arbitrary number of LiDAR beams into a fixed number
        of sectors (self.num_sectors) using minimum-pooling.

        Why Min-Pooling?
        In robotics obstacle avoidance, preserving the closest obstacle
        within an angular corridor is critical for safety. Averaging would
        dangerously dilute a close obstacle if neighboring rays hit a distant wall.
        """
        total_rays = len(cleaned_ranges)
        if total_rays == 0:
            return np.full(self.num_sectors, self.range_max, dtype=np.float32)

        if total_rays == self.num_sectors:
            return cleaned_ranges

        # Split rays into num_sectors approximately equal chunks
        sector_indices = np.array_split(np.arange(total_rays), self.num_sectors)
        sector_distances = np.empty(self.num_sectors, dtype=np.float32)

        for i, indices in enumerate(sector_indices):
            if len(indices) > 0:
                sector_distances[i] = np.min(cleaned_ranges[indices])
            else:
                sector_distances[i] = self.range_max

        return sector_distances

    def normalize(self, sector_distances: np.ndarray) -> np.ndarray:
        """
        Linearly normalizes distances into the [0.0, 1.0] interval.

        Default (invert_normalization = False):
            0.0 -> Obstacle touching vehicle (range_min)
            1.0 -> Fully clear corridor (range_max)
        """
        denom = max(self.range_max - self.range_min, 1e-5)
        normalized = (sector_distances - self.range_min) / denom
        normalized = np.clip(normalized, 0.0, 1.0)

        if self.invert_normalization:
            normalized = 1.0 - normalized

        return normalized.astype(np.float32)

    def process(self, raw_ranges: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        End-to-end processing pipeline:
        Raw Scan -> Clean Bounds -> Sector Min-Pool -> Normalize -> Fixed Vector [num_sectors]
        """
        cleaned = self.clean_ranges(raw_ranges)
        pooled = self.sector_min_pool(cleaned)
        normalized = self.normalize(pooled)
        return normalized

    def extract_zones(
        self,
        sector_distances: np.ndarray,
        forward_sector_ratio: float = 0.25
    ) -> Tuple[float, float, float]:
        """
        Extracts representative minimum obstacle distances for three
        primary operational zones: Left, Center (Front), and Right.

        Assuming a 360-degree scan where index 0 is typically the rear (-pi)
        or front (0), or centered at 180 deg. This helper inspects sector
        distances to provide zone clearances for the rule-based teacher.

        Returns:
            (left_dist, front_dist, right_dist) in meters.
        """
        k = len(sector_distances)
        quarter = k // 4
        eighth = k // 8

        # In standard ROS 2 LaserScan:
        # angle_min = -pi (-180 deg, rear)
        # angle = 0 (0 deg, straight ahead / center)
        # angle_max = +pi (+180 deg, rear)
        # Index k/2 is roughly the forward center (0 radians)
        center_idx = k // 2

        front_start = max(0, center_idx - eighth)
        front_end = min(k, center_idx + eighth)
        front_dist = float(np.min(sector_distances[front_start:front_end]))

        # Right side: from rear-right to front-right (indices below front)
        right_start = max(0, center_idx - 2 * eighth)
        right_dist = float(np.min(sector_distances[right_start:front_start]))

        # Left side: from front-left to rear-left (indices above front)
        left_end = min(k, center_idx + 2 * eighth)
        left_dist = float(np.min(sector_distances[front_end:left_end]))

        return left_dist, front_dist, right_dist
