"""
Data loading and preprocessing module for NeRF.

This module contains functions and classes for loading, processing,
and preparing camera data and images for NeRF training.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, Dict, List, Optional, Union
import glob


class CameraIntrinsics:
    """Camera intrinsic parameters."""
    
    def __init__(self, width: int, height: int, focal_length: float):
        """
        Initialize camera intrinsics.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            focal_length: Focal length of the camera
        """
        self.width = width
        self.height = height
        self.focal_length = focal_length
        
    @classmethod
    def from_json(cls, json_data: Dict) -> 'CameraIntrinsics':
        """Create camera intrinsics from JSON data."""
        return cls(
            width=int(json_data['width']),
            height=int(json_data['height']),
            focal_length=float(json_data['focal_length'])
        )


class CameraPose:
    """Camera extrinsic parameters (pose)."""
    
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        """
        Initialize camera pose.
        
        Args:
            rotation: 3x3 rotation matrix
            translation: 3x1 translation vector
        """
        self.rotation = rotation  # 3x3 matrix
        self.translation = translation  # 3x1 vector
        
    @property
    def matrix(self) -> np.ndarray:
        """Get the 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix
    
    @classmethod
    def from_json(cls, json_data: Dict) -> 'CameraPose':
        """Create camera pose from JSON data."""
        rotation = np.array(json_data['rotation']).reshape(3, 3)
        translation = np.array(json_data['translation'])
        return cls(rotation=rotation, translation=translation)


class NeRFDataset(Dataset):
    """Dataset for NeRF training."""
    
    def __init__(self, 
                 data_dir: str, 
                 split: str = 'train',
                 img_size: Optional[Tuple[int, int]] = None,
                 preload: bool = False):
        """
        Initialize NeRF dataset.
        
        Args:
            data_dir: Path to the directory containing the dataset
            split: Dataset split ('train', 'val', or 'test')
            img_size: Optional tuple to resize images (width, height)
            preload: Whether to preload all images into memory
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.preload = preload
        
        # Load dataset information
        self._load_metadata()
        
        # Preload images if necessary
        self.images = None
        if self.preload:
            self._preload_images()
    
    def _load_metadata(self):
        """Load dataset metadata."""
        json_path = os.path.join(self.data_dir, f'transforms_{self.split}.json')
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Metadata file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Parse camera parameters
        self.camera_intrinsics = CameraIntrinsics.from_json(metadata['camera'])
        
        # Parse frames
        self.frames = []
        for frame in metadata['frames']:
            image_path = os.path.join(self.data_dir, frame['file_path'])
            camera_pose = CameraPose.from_json(frame['transform_matrix'])
            
            self.frames.append({
                'image_path': image_path,
                'camera_pose': camera_pose
            })
    
    def _preload_images(self):
        """Preload all images into memory."""
        self.images = []
        for frame in self.frames:
            img = self._load_image(frame['image_path'])
            self.images.append(img)
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        img = Image.open(path)
        
        # Resize if necessary
        if self.img_size is not None:
            img = img.resize(self.img_size, Image.LANCZOS)
        
        # Convert to tensor and normalize
        img = np.array(img) / 255.0
        img = torch.from_numpy(img).float()
        
        # Handle grayscale images
        if len(img.shape) == 2:
            img = img.unsqueeze(-1).repeat(1, 1, 3)
        
        return img
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
                - 'image': The RGB image tensor (H, W, 3)
                - 'pose': The camera pose matrix (4, 4)
                - 'intrinsics': Camera intrinsics tensor [fx, fy, cx, cy]
        """
        frame = self.frames[idx]
        
        # Load image if not preloaded
        if self.images is not None:
            image = self.images[idx]
        else:
            image = self._load_image(frame['image_path'])
        
        # Get camera pose
        pose = torch.from_numpy(frame['camera_pose'].matrix).float()
        
        # Camera intrinsics
        intr = self.camera_intrinsics
        fx = fy = intr.focal_length
        cx = intr.width / 2
        cy = intr.height / 2
        intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        
        return {
            'image': image,  # (H, W, 3)
            'pose': pose,    # (4, 4)
            'intrinsics': intrinsics  # (4,)
        }


def create_data_loader(dataset: NeRFDataset, 
                      batch_size: int,
                      shuffle: bool = True,
                      num_workers: int = 4) -> DataLoader:
    """
    Create a data loader for a NeRF dataset.
    
    Args:
        dataset: The NeRF dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def prepare_data_for_nerf(data_dir: str, 
                         img_size: Optional[Tuple[int, int]] = None,
                         batch_size: int = 1,
                         preload: bool = False) -> Dict[str, DataLoader]:
    """
    Prepare data loaders for NeRF training.
    
    Args:
        data_dir: Path to the data directory
        img_size: Optional tuple to resize images (width, height)
        batch_size: Batch size for training
        preload: Whether to preload images into memory
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test' splits
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = NeRFDataset(
                data_dir=data_dir,
                split=split,
                img_size=img_size,
                preload=preload
            )
            
            is_train = (split == 'train')
            loaders[split] = create_data_loader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=is_train,
                num_workers=4 if is_train else 2
            )
        except FileNotFoundError:
            print(f"No {split} split found, skipping.")
    
    return loaders
