"""
Utility functions for NeRF.

This module contains various utility functions used in the NeRF pipeline,
including data visualization, transformation helpers, and config management.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
import cv2
from pathlib import Path


def visualize_depth(depth: torch.Tensor,
                   cmap: str = 'turbo') -> np.ndarray:
    """
    Visualize a depth map using a colormap.
    
    Args:
        depth: Depth map tensor
        cmap: Matplotlib colormap name
        
    Returns:
        Colored depth map as a numpy array
    """
    depth = depth.cpu().numpy()
    
    # Normalize to [0, 1]
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    
    # Apply colormap
    cmap_func = plt.get_cmap(cmap)
    depth_colored = cmap_func(depth_norm)
    
    # Convert to RGB, removing alpha channel
    depth_colored = depth_colored[..., :3]
    
    return depth_colored


def create_visualization_grid(images: List[np.ndarray],
                             n_cols: int,
                             title: str = None,
                             save_path: Optional[str] = None) -> np.ndarray:
    """
    Create a grid of images for visualization.
    
    Args:
        images: List of images to visualize
        n_cols: Number of columns in the grid
        title: Optional title for the figure
        save_path: Optional path to save the figure
        
    Returns:
        Grid of images as a numpy array
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Determine image size
    h, w = images[0].shape[:2]
    
    # Create grid
    grid = np.zeros((h * n_rows, w * n_cols, 3), dtype=np.float32)
    
    # Fill grid
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    if title or save_path:
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        plt.imshow(grid)
        
        if title:
            plt.title(title)
        
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        
    return grid


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save experiment configuration to a file.
    
    Args:
        config: Dictionary with configuration
        path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Determine file type
    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    elif path.endswith(('.yaml', '.yml')):
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    print(f"Config saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from a file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Dictionary with configuration
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Determine file type
    if path.endswith('.json'):
        with open(path, 'r') as f:
            config = json.load(f)
    elif path.endswith(('.yaml', '.yml')):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    return config


def create_folder_structure(base_dir: str) -> Dict[str, str]:
    """
    Create a standard folder structure for experiments.
    
    Args:
        base_dir: Base directory for the experiment
        
    Returns:
        Dictionary with paths to each folder
    """
    folders = {
        'data': os.path.join(base_dir, 'data'),
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'logs': os.path.join(base_dir, 'logs'),
        'results': os.path.join(base_dir, 'results'),
        'configs': os.path.join(base_dir, 'configs')
    }
    
    # Create folders
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders


def get_all_files(directory: str, extension: str) -> List[str]:
    """
    Get all files with a specific extension in a directory.
    
    Args:
        directory: Directory to search
        extension: File extension to filter
        
    Returns:
        List of file paths
    """
    if not os.path.exists(directory):
        return []
    
    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension
    
    # Find all files
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    
    return sorted(files)


def make_rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """
    Create a rotation matrix from Euler angles (in radians).
    
    Args:
        angle_x: Rotation around X-axis
        angle_y: Rotation around Y-axis
        angle_z: Rotation around Z-axis
        
    Returns:
        3x3 rotation matrix
    """
    # X-axis rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    # Y-axis rotation
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    # Z-axis rotation
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: Rx * Ry * Rz
    return Rx @ Ry @ Rz


def poses_avg(poses: np.ndarray) -> np.ndarray:
    """
    Calculate the average pose.
    
    Args:
        poses: Array of pose matrices, shape (N, 4, 4)
        
    Returns:
        Average pose matrix, shape (4, 4)
    """
    # Compute the center of all camera positions
    center = poses[:, :3, 3].mean(0)
    
    # Compute the average z-axis (forward direction)
    z = poses[:, :3, 2].mean(0)
    z = z / np.linalg.norm(z)
    
    # Compute the average y-axis (up direction)
    y_ = poses[:, :3, 1].mean(0)
    x = np.cross(y_, z)
    x = x / np.linalg.norm(x)
    
    # Recompute y to ensure orthogonality
    y = np.cross(z, x)
    
    # Construct average pose matrix
    pose_avg = np.stack([x, y, z, center], axis=1)
    
    # Add homogeneous row
    pose_avg = np.concatenate([pose_avg, np.array([[0, 0, 0, 1]])], axis=0)
    
    return pose_avg


def normalize_poses(poses: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize camera poses to be centered at the origin and have a specific scale.
    
    Args:
        poses: Array of pose matrices, shape (N, 4, 4)
        
    Returns:
        Tuple of:
            - Normalized poses, shape (N, 4, 4)
            - Scene scale factor
    """
    # Compute average pose
    pose_avg = poses_avg(poses)
    
    # Compute average distance to the camera center
    distances = np.linalg.norm(poses[:, :3, 3] - pose_avg[:3, 3], axis=1)
    scale = 1.0 / distances.mean()
    
    # Normalize poses
    poses_normalized = np.copy(poses)
    
    # Center and scale the poses
    for i in range(len(poses)):
        poses_normalized[i][:3, 3] = scale * (poses[i][:3, 3] - pose_avg[:3, 3])
    
    return poses_normalized, scale


def load_and_resize_image(path: str, 
                         target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image and optionally resize it.
    
    Args:
        path: Path to the image
        target_size: Optional target size (width, height)
        
    Returns:
        Loaded and resized image
    """
    # Load image
    img = cv2.imread(path)
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def prepare_output_image(rgb: torch.Tensor, depth: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
    """
    Prepare output images for visualization.
    
    Args:
        rgb: RGB image tensor
        depth: Optional depth image tensor
        
    Returns:
        Dictionary with prepared images
    """
    # Convert to numpy and ensure correct range
    rgb_np = rgb.cpu().numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    
    outputs = {'rgb': rgb_np}
    
    # Process depth if provided
    if depth is not None:
        depth_colored = visualize_depth(depth)
        outputs['depth'] = depth_colored
    
    return outputs
