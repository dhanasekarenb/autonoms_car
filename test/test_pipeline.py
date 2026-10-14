"""
Unit Tests for autonoms_car Pipeline Components
Verifies preprocessor logic, rule-based teacher decisions, and PyTorch model behavior.
"""

import os
import tempfile
import unittest
import numpy as np
import torch

from autonoms_car.preprocessor import LidarPreprocessor
from autonoms_car.rule_based_teacher import RuleBasedTeacher
from autonoms_car.model import ObstacleAvoidanceMLP


class TestLidarPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = LidarPreprocessor(
            num_sectors=36,
            range_min=0.10,
            range_max=5.00
        )

    def test_clean_ranges(self):
        raw = [float('inf'), float('-inf'), float('nan'), 0.05, 10.0, 2.5]
        cleaned = self.preprocessor.clean_ranges(raw)

        self.assertEqual(cleaned[0], 5.0)   # inf -> range_max
        self.assertEqual(cleaned[1], 0.10)  # -inf -> range_min
        self.assertEqual(cleaned[2], 0.10)  # nan -> range_min
        self.assertEqual(cleaned[3], 0.10)  # < range_min -> range_min
        self.assertEqual(cleaned[4], 5.0)   # > range_max -> range_max
        self.assertAlmostEqual(cleaned[5], 2.5, places=4)

    def test_sector_min_pool(self):
        # 360 rays downsampled to 36 sectors (10 rays per sector)
        raw_360 = np.full(360, 4.0, dtype=np.float32)
        # Place an obstacle in sector 18 (indices 180 to 189)
        raw_360[183] = 0.8
        raw_360[184] = 0.5  # closest

        pooled = self.preprocessor.sector_min_pool(raw_360)
        self.assertEqual(len(pooled), 36)
        self.assertAlmostEqual(pooled[18], 0.5, places=4)
        self.assertAlmostEqual(pooled[0], 4.0, places=4)

    def test_normalization(self):
        distances = np.array([0.10, 2.55, 5.00], dtype=np.float32)
        norm = self.preprocessor.normalize(distances)
        self.assertAlmostEqual(norm[0], 0.0, places=3)
        self.assertAlmostEqual(norm[1], 0.5, places=2)
        self.assertAlmostEqual(norm[2], 1.0, places=3)


class TestRuleBasedTeacher(unittest.TestCase):

    def setUp(self):
        self.teacher = RuleBasedTeacher(
            max_linear_vel=0.35,
            max_angular_vel=1.20,
            safe_distance=1.20,
            stop_distance=0.35
        )

    def test_open_corridor_cruising(self):
        # Open field: all zones clear at 4.0m
        v, w = self.teacher.compute_action(left_dist=4.0, front_dist=4.0, right_dist=4.0)
        self.assertAlmostEqual(v, 0.35, places=2)
        self.assertAlmostEqual(w, 0.0, places=2)

    def test_front_obstacle_slowdown_and_steer(self):
        # Obstacle ahead at 0.8m, left has more room than right (left=3.0, right=0.9)
        v, w = self.teacher.compute_action(left_dist=3.0, front_dist=0.8, right_dist=0.9)
        # Should slow down
        self.assertLess(v, 0.35)
        self.assertGreater(v, 0.0)
        # Should turn left (positive angular velocity)
        self.assertGreater(w, 0.0)

    def test_critical_emergency_stop_and_spin(self):
        # Obstacle at 0.20m (below stop_distance 0.35m)
        v, w = self.teacher.compute_action(left_dist=0.5, front_dist=0.20, right_dist=2.0)
        # Must stop completely
        self.assertEqual(v, 0.0)
        # Right has more room -> turn right (negative angular velocity)
        self.assertAlmostEqual(w, -1.20, places=2)


class TestObstacleAvoidanceMLP(unittest.TestCase):

    def setUp(self):
        self.model = ObstacleAvoidanceMLP(
            input_size=36,
            max_linear_vel=0.35,
            max_angular_vel=1.20
        )

    def test_forward_output_bounds(self):
        batch_size = 8
        dummy_input = torch.rand((batch_size, 36))
        output = self.model(dummy_input)

        self.assertEqual(output.shape, (batch_size, 2))

        # Check linear velocities are non-negative and <= max_linear_vel
        linear_v = output[:, 0]
        self.assertTrue(torch.all(linear_v >= 0.0))
        self.assertTrue(torch.all(linear_v <= 0.35 + 1e-4))

        # Check angular velocities are within [-max_angular_vel, max_angular_vel]
        angular_w = output[:, 1]
        self.assertTrue(torch.all(angular_w >= -1.20 - 1e-4))
        self.assertTrue(torch.all(angular_w <= 1.20 + 1e-4))

    def test_save_and_load_checkpoint(self):
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.model.save_checkpoint(tmp_path)
            loaded_model = ObstacleAvoidanceMLP.load_from_checkpoint(tmp_path)
            self.assertEqual(loaded_model.input_size, 36)
            self.assertAlmostEqual(loaded_model.max_linear_vel, 0.35, places=2)

            test_x = np.random.uniform(0.0, 1.0, size=36).astype(np.float32)
            v1, w1 = self.model.predict_single(test_x)
            v2, w2 = loaded_model.predict_single(test_x)
            self.assertAlmostEqual(v1, v2, places=4)
            self.assertAlmostEqual(w1, w2, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
