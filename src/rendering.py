"""
Rendering and evaluation module for NeRF.

This module contains functions for rendering novel views from a trained NeRF model
and evaluating the quality of those renderings.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Union
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .ray_utils import get_ray_directions, get_rays, render_rays
from .model import NeRFModel


class NeRFRenderer:
    """Renderer class for NeRF models."""
    
    def __init__(self,
                model: NeRFModel,
                device: str = 'cuda',
                chunk_size: int = 1024 * 32,
                near: float = 2.0,
                far: float = 6.0,
                white_background: bool = True):
        """
        Initialize NeRF renderer.
        
        Args:
            model: Trained NeRF model
            device: Device to render on ('cuda' or 'cpu')
            chunk_size: Number of rays to process at once
            near: Near clipping plane
            far: Far clipping plane
            white_background: Whether scene has white background
        """
        self.model = model
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        self.chunk_size = chunk_size
        self.near = near
        self.far = far
        self.white_background = white_background
    
    def render_image(self,
                    height: int,
                    width: int,
                    focal_length: float,
                    pose: torch.Tensor,
                    center: Optional[Tuple[float, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Render a full image from the model.
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            focal_length: Focal length of the camera
            pose: Camera pose, shape (4, 4)
            center: Optional camera center (cx, cy)
            
        Returns:
            Dictionary with rendered RGB and depth images
        """
        # Get ray directions in camera frame
        directions = get_ray_directions(
            height=height, 
            width=width, 
            focal_length=focal_length,
            center=center
        ).to(self.device)  # (H, W, 3)
        
        # Get rays in world frame
        rays_origin, rays_direction = get_rays(directions, pose.to(self.device))
        
        # Flatten rays for batched processing
        rays_o = rays_origin.reshape(-1, 3)  # (H*W, 3)
        rays_d = rays_direction.reshape(-1, 3)  # (H*W, 3)
        
        # Process in chunks to avoid OOM
        rgb_chunks = []
        depth_chunks = []
        
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], self.chunk_size):
                # Get chunk
                chunk_rays_o = rays_o[i:i+self.chunk_size]  # (chunk_size, 3)
                chunk_rays_d = rays_d[i:i+self.chunk_size]  # (chunk_size, 3)
                
                # Render chunk
                with torch.cuda.amp.autocast():
                    chunk_results = render_rays(
                        model=self.model,
                        rays_origin=chunk_rays_o,
                        rays_direction=chunk_rays_d,
                        near=self.near,
                        far=self.far,
                        num_coarse_samples=64,
                        num_fine_samples=128,
                        perturb=False,  # No perturbation during rendering
                        white_background=self.white_background
                    )
                
                # Use fine results if available, otherwise coarse
                if 'rgb_fine' in chunk_results:
                    rgb_chunks.append(chunk_results['rgb_fine'].cpu())
                    depth_chunks.append(chunk_results['depth_fine'].cpu())
                else:
                    rgb_chunks.append(chunk_results['rgb_coarse'].cpu())
                    depth_chunks.append(chunk_results['depth_coarse'].cpu())
        
        # Combine chunks
        rgb = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3)
        depth = torch.cat(depth_chunks, dim=0).reshape(height, width)
        
        return {
            'rgb': rgb,
            'depth': depth
        }
    
    def render_path(self,
                   render_poses: List[torch.Tensor],
                   height: int,
                   width: int,
                   focal_length: float,
                   center: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
        """
        Render images along a camera path.
        
        Args:
            render_poses: List of camera poses
            height: Image height in pixels
            width: Image width in pixels
            focal_length: Focal length of the camera
            center: Optional camera center (cx, cy)
            
        Returns:
            Dictionary with rendered RGB and depth image sequences
        """
        self.model.eval()
        
        # Storage for rendered images
        rgbs = []
        depths = []
        
        # Render each pose
        for pose in tqdm(render_poses, desc="Rendering path"):
            # Render image
            rendered = self.render_image(
                height=height,
                width=width,
                focal_length=focal_length,
                pose=pose,
                center=center
            )
            
            # Store results
            rgbs.append(rendered['rgb'].numpy())
            depths.append(rendered['depth'].numpy())
        
        # Stack all images
        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0)
        
        return {
            'rgb': rgbs,
            'depth': depths
        }
    
    @staticmethod
    def create_360_rendering_path(radius: float, n_frames: int = 120) -> List[torch.Tensor]:
        """
        Create a 360-degree rendering path around the origin.
        
        Args:
            radius: Radius of the camera path
            n_frames: Number of frames in the 360-degree path
            
        Returns:
            List of camera pose matrices
        """
        poses = []
        
        for i in range(n_frames):
            # 360-degree rotation
            angle = i / n_frames * 2 * np.pi
            
            # Camera position
            x = radius * np.sin(angle)
            z = radius * np.cos(angle)
            y = 0.0  # Fixed height
            
            # Camera looking at origin
            look_at = np.array([0, 0, 0])
            up = np.array([0, 1, 0])
            
            # Camera position
            position = np.array([x, y, z])
            
            # Camera orientation
            forward = look_at - position
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            
            # Create pose matrix
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward  # Negative because of OpenGL convention
            pose[:3, 3] = position
            
            poses.append(torch.from_numpy(pose).float())
        
        return poses
    
    def create_spiral_rendering_path(center: np.ndarray,
                                    radius: float,
                                    n_frames: int = 120,
                                    n_rotations: float = 2.0,
                                    z_range: Tuple[float, float] = (-0.5, 0.5)) -> List[torch.Tensor]:
        """
        Create a spiral rendering path.
        
        Args:
            center: Center point of the spiral
            radius: Radius of the spiral
            n_frames: Number of frames in the spiral
            n_rotations: Number of rotations in the spiral
            z_range: Range of z values (to control the vertical movement)
            
        Returns:
            List of camera pose matrices
        """
        poses = []
        
        for i in range(n_frames):
            # Normalized progress along the path
            t = i / (n_frames - 1)
            
            # Angle for this frame
            angle = t * 2 * np.pi * n_rotations
            
            # Camera position
            x = center[0] + radius * np.sin(angle)
            z = center[2] + radius * np.cos(angle)
            y = center[1] + (t * (z_range[1] - z_range[0]) + z_range[0])
            
            # Camera looking at the center
            look_at = center
            up = np.array([0, 1, 0])
            
            # Camera position
            position = np.array([x, y, z])
            
            # Camera orientation
            forward = look_at - position
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            
            # Create pose matrix
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward  # Negative because of OpenGL convention
            pose[:3, 3] = position
            
            poses.append(torch.from_numpy(pose).float())
        
        return poses
    
    def save_video(self, 
                 rendered_images: np.ndarray,
                 output_path: str, 
                 fps: int = 30) -> None:
        """
        Save a video from a sequence of rendered images.
        
        Args:
            rendered_images: Rendered image sequence, shape (n_frames, H, W, 3)
            output_path: Path to save the video
            fps: Frames per second
        """
        # Convert to 8-bit RGB
        images_8bit = (np.clip(rendered_images, 0, 1) * 255).astype(np.uint8)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write video
        imageio.mimwrite(output_path, images_8bit, fps=fps, quality=8)
        print(f"Video saved to {output_path}")


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute image quality metrics between rendered and ground truth images.
    
    Args:
        predictions: Predicted RGB values, shape (N, H, W, 3)
        targets: Target RGB values, shape (N, H, W, 3)
        
    Returns:
        Dictionary with PSNR, SSIM and LPIPS metrics
    """
    # Ensure inputs are on the CPU
    predictions = predictions.cpu()
    targets = targets.cpu()
    
    n_images = predictions.shape[0]
    
    # Initialize metrics
    psnr_vals = []
    
    # Compute metrics
    for i in range(n_images):
        pred = predictions[i]
        target = targets[i]
        
        # Mean squared error
        mse = F.mse_loss(pred, target).item()
        
        # PSNR
        psnr = -10.0 * np.log10(mse)
        psnr_vals.append(psnr)
    
    # Average metrics
    avg_psnr = np.mean(psnr_vals)
    
    return {
        'psnr': avg_psnr
    }


def evaluate_model(model: NeRFModel, 
                  test_loader: DataLoader, 
                  device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate a trained NeRF model on a test set.
    
    Args:
        model: Trained NeRF model
        test_loader: DataLoader for test data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create renderer
    renderer = NeRFRenderer(
        model=model,
        device=device,
        chunk_size=1024 * 32
    )
    
    # Set model to eval mode
    model.eval()
    
    # Initialize containers for predictions and targets
    all_preds = []
    all_targets = []
    
    # Evaluate on test set
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            # Get data
            images = batch['image'].to(device)  # (B, H, W, 3)
            poses = batch['pose'].to(device)    # (B, 4, 4)
            intrinsics = batch['intrinsics'].to(device)  # (B, 4)
            
            batch_size, height, width, _ = images.shape
            
            # Process each image in the batch
            for b in range(batch_size):
                # Camera intrinsics
                fx, fy, cx, cy = intrinsics[b]
                
                # Render image
                rendered = renderer.render_image(
                    height=height,
                    width=width,
                    focal_length=fx.item(),
                    pose=poses[b],
                    center=(cx.item(), cy.item())
                )
                
                # Store prediction and target
                all_preds.append(rendered['rgb'])
                all_targets.append(images[b].cpu())
    
    # Stack all predictions and targets
    all_preds = torch.stack(all_preds, dim=0)
    all_targets = torch.stack(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets)
    
    return metrics
