"""
PyTorch Training Script for Supervised Obstacle Avoidance

Trains the ObstacleAvoidanceMLP using recorded teacher data or synthetic
scenarios, evaluates validation loss, and saves the trained model weights.
"""

import os
import argparse
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from autonoms_car.model import ObstacleAvoidanceMLP
from autonoms_car.rule_based_teacher import RuleBasedTeacher


class LidarDataset(Dataset):
    """PyTorch Dataset wrapping normalized LiDAR features and target velocities."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(features).float()
        self.Y = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def generate_synthetic_dataset(
    num_samples: int = 4000,
    num_sectors: int = 36,
    range_max: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates realistic synthetic obstacle scenarios and labels them
    with the RuleBasedTeacher. Useful for offline verification and sanity testing.
    """
    print(f"[Dataset Generator] Creating {num_samples} synthetic training scenarios...")
    teacher = RuleBasedTeacher(
        max_linear_vel=0.35,
        max_angular_vel=1.20,
        safe_distance=1.20,
        stop_distance=0.35
    )

    X_list = []
    Y_list = []

    center_idx = num_sectors // 2
    quarter = num_sectors // 4
    eighth = num_sectors // 8

    np.random.seed(42)

    for i in range(num_samples):
        # Scenario types:
        # 0: Open field
        # 1: Wall directly ahead
        # 2: Obstacle on left
        # 3: Obstacle on right
        # 4: Narrow corridor / corner
        scenario = np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.25, 0.20, 0.20, 0.10])

        # Start with random open distances between 2.0m and range_max
        distances = np.random.uniform(2.5, range_max, size=num_sectors).astype(np.float32)

        if scenario == 1:
            # Frontal obstacle: distance between 0.2m and 1.5m
            dist = np.random.uniform(0.2, 1.5)
            distances[center_idx - eighth : center_idx + eighth] = dist
            # One side slightly more open than the other
            side = np.random.choice(['left', 'right'])
            if side == 'left':
                distances[center_idx + eighth : center_idx + quarter] = np.random.uniform(2.0, 4.0)
            else:
                distances[center_idx - quarter : center_idx - eighth] = np.random.uniform(2.0, 4.0)

        elif scenario == 2:
            # Obstacle on left
            dist = np.random.uniform(0.3, 1.2)
            distances[center_idx : center_idx + quarter] = dist

        elif scenario == 3:
            # Obstacle on right
            dist = np.random.uniform(0.3, 1.2)
            distances[center_idx - quarter : center_idx] = dist

        elif scenario == 4:
            # Narrow passage / trapped corner
            dist_front = np.random.uniform(0.2, 0.6)
            distances[center_idx - eighth : center_idx + eighth] = dist_front
            distances[center_idx : center_idx + quarter] = np.random.uniform(0.3, 0.8)
            distances[center_idx - quarter : center_idx] = np.random.uniform(0.3, 0.8)

        # Extract zone metrics
        front_d = float(np.min(distances[center_idx - eighth : center_idx + eighth]))
        right_d = float(np.min(distances[center_idx - 2 * eighth : center_idx - eighth]))
        left_d = float(np.min(distances[center_idx + eighth : center_idx + 2 * eighth]))

        # Teacher generates ground truth action
        v, w = teacher.compute_action(left_d, front_d, right_d)

        # Normalize distances to [0.0, 1.0]
        norm_distances = np.clip(distances / range_max, 0.0, 1.0)

        X_list.append(norm_distances)
        Y_list.append([v, w])

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def train_model(
    data_path: str = 'dataset/training_data.npz',
    output_model_path: str = 'models/obstacle_avoidance_mlp.pth',
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    synthetic_fallback: bool = True
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] Using compute device: {device}")

    # 1. Load or synthesize dataset
    if os.path.exists(data_path):
        print(f"[Training] Loading dataset from: {data_path}")
        data = np.load(data_path)
        X = data['features']
        Y = data['labels']
    elif synthetic_fallback:
        print(f"[Training] Dataset file '{data_path}' not found.")
        X, Y = generate_synthetic_dataset(num_samples=4000)
        os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
        np.savez_compressed(data_path, features=X, labels=Y)
        print(f"[Training] Saved synthetic baseline dataset to '{data_path}'.")
    else:
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"[Training] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

    # 2. Train / Validation Split (80% / 20%)
    dataset = LidarDataset(X, Y)
    val_size = int(len(dataset) * 0.20)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer
    input_dim = X.shape[1]
    model = ObstacleAvoidanceMLP(
        input_size=input_dim,
        hidden_dim1=64,
        hidden_dim2=32,
        max_linear_vel=0.35,
        max_angular_vel=1.20
    ).to(device)

    # Weighted MSE loss: prioritize accurate steering (w) to prevent collisions
    def custom_loss(preds, targets):
        mse_v = nn.functional.mse_loss(preds[:, 0], targets[:, 0])
        mse_w = nn.functional.mse_loss(preds[:, 1], targets[:, 1])
        return mse_v + 2.0 * mse_w

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    print("-" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'LR':<12}{'Status'}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = custom_loss(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_set)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                loss = custom_loss(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_set)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(output_model_path)
            status = "★ Saved Best"

        if epoch % 5 == 0 or epoch == 1 or status:
            print(f"{epoch:<8}{train_loss:<15.5f}{val_loss:<15.5f}{current_lr:<12.6f}{status}")

    print("-" * 65)
    print(f"[Training Complete] Best Validation Loss: {best_val_loss:.5f}")
    print(f"[Model Saved] Target weights stored at: {output_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Obstacle Avoidance MLP")
    parser.add_argument('--data', type=str, default='dataset/training_data.npz')
    parser.add_argument('--output', type=str, default='models/obstacle_avoidance_mlp.pth')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()
