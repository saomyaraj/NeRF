"""
Ray sampling and processing module for NeRF.

This module contains functions for generating, sampling, and processing rays
for use in the NeRF rendering pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Union


def get_ray_directions(height: int, 
                      width: int, 
                      focal_length: float,
                      center: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Generate ray directions for a pinhole camera model.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Focal length of the camera
        center: Optional camera center (cx, cy), defaults to image center
        
    Returns:
        Ray directions tensor of shape (height, width, 3)
    """
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    
    # Create pixel coordinates grid
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing='xy'
    )
    
    # Shift by center and normalize by focal length
    directions = torch.stack([
        (i - cx) / focal_length,
        -(j - cy) / focal_length,  # Negative for correct orientation
        -torch.ones_like(i)  # Negative z-direction (camera looks along -z)
    ], dim=-1)
    
    return directions


def get_rays(directions: torch.Tensor, 
            c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get ray origins and normalized directions in world coordinate system.
    
    Args:
        directions: Ray directions in camera frame, shape (..., 3)
        c2w: Camera-to-world transformation matrix, shape (4, 4)
        
    Returns:
        Tuple of:
            - rays_origin: Ray origins in world frame, shape (..., 3)
            - rays_direction: Normalized ray directions in world frame, shape (..., 3)
    """
    # Rotate ray directions from camera to world frame
    rays_direction = torch.sum(
        directions[..., None, :] * c2w[:3, :3], 
        dim=-1
    )
    
    # Get ray origin (camera position in world frame)
    rays_origin = c2w[:3, -1].expand(rays_direction.shape)
    
    # Normalize ray directions
    rays_direction = F.normalize(rays_direction, dim=-1)
    
    return rays_origin, rays_direction


def sample_points_along_rays(rays_origin: torch.Tensor, 
                            rays_direction: torch.Tensor,
                            near: float, 
                            far: float,
                            num_samples: int,
                            perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays between near and far bounds.
    
    Args:
        rays_origin: Ray origins, shape (..., 3)
        rays_direction: Ray directions, shape (..., 3)
        near: Near bound of the sampling
        far: Far bound of the sampling
        num_samples: Number of samples per ray
        perturb: Whether to add random perturbation to samples
        
    Returns:
        Tuple of:
            - points: Sampled 3D points, shape (..., num_samples, 3)
            - z_vals: Depths of sampled points, shape (..., num_samples)
    """
    # Generate evenly spaced samples between near and far along each ray
    t_vals = torch.linspace(near, far, num_samples, device=rays_origin.device)
    
    # Expand to match the shape of rays
    z_vals = t_vals.expand(list(rays_origin.shape[:-1]) + [num_samples])
    
    # Add random perturbations if requested
    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        
        # Stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    
    # Generate points along each ray
    # rays_origin: (..., 3)
    # rays_direction: (..., 3)
    # z_vals: (..., num_samples)
    
    # Expand rays to sample points
    # (..., num_samples, 3) = (..., 1, 3) + (..., num_samples, 1) * (..., 1, 3)
    points = rays_origin[..., None, :] + z_vals[..., :, None] * rays_direction[..., None, :]
    
    return points, z_vals


def sample_pdf(bins: torch.Tensor, 
              weights: torch.Tensor, 
              num_samples: int, 
              perturb: bool = True) -> torch.Tensor:
    """
    Sample points from a probability density function (PDF) defined by weights.
    Used for the hierarchical sampling in NeRF.
    
    Args:
        bins: Bin edges, shape (..., num_bins)
        weights: Weights defining the PDF, shape (..., num_bins-1)
        num_samples: Number of samples to draw
        perturb: Whether to add random perturbation to samples
        
    Returns:
        Sampled points, shape (..., num_samples)
    """
    # Get PDF and CDF from weights
    weights = weights + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (..., num_bins)
    
    # Take uniform samples
    if perturb:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0, 1, num_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    
    # Invert CDF to find samples
    # Find indices of the first element in the CDF that is larger than u
    indices = torch.searchsorted(cdf, u, right=True)
    
    # Clamp the indices to valid range
    below = torch.clamp(indices - 1, min=0)
    above = torch.clamp(indices, max=cdf.shape[-1] - 1)
    
    # Get CDF values at these indices
    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    
    # Get bin values at these indices
    bins_below = torch.gather(bins, -1, below)
    bins_above = torch.gather(bins, -1, above)
    
    # Avoid division by zero
    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    
    # Linear interpolation to get samples
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)
    
    return samples


def hierarchical_sampling(rays_origin: torch.Tensor,
                         rays_direction: torch.Tensor,
                         z_vals: torch.Tensor,
                         weights: torch.Tensor,
                         num_fine_samples: int,
                         perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform hierarchical sampling for the fine network.
    
    Args:
        rays_origin: Ray origins, shape (..., 3)
        rays_direction: Ray directions, shape (..., 3)
        z_vals: Depths of coarse samples, shape (..., num_coarse_samples)
        weights: Weights from the coarse network, shape (..., num_coarse_samples)
        num_fine_samples: Number of fine samples
        perturb: Whether to add random perturbation to samples
        
    Returns:
        Tuple of:
            - fine_points: Fine sampled 3D points, shape (..., num_fine_samples, 3)
            - fine_z_vals: Depths of fine sampled points, shape (..., num_fine_samples)
            - combined_z_vals: Combined depths, shape (..., num_coarse_samples + num_fine_samples)
    """
    # Get the mid points between z_vals
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    
    # Sample new points from PDF defined by weights
    fine_z_vals = sample_pdf(
        bins=z_vals_mid, 
        weights=weights[..., 1:-1],
        num_samples=num_fine_samples, 
        perturb=perturb
    )
    
    # Sort the combined z values
    combined_z_vals, _ = torch.sort(torch.cat([z_vals, fine_z_vals], dim=-1), dim=-1)
    
    # Sample points along the rays using these z values
    fine_points = rays_origin[..., None, :] + combined_z_vals[..., :, None] * rays_direction[..., None, :]
    
    return fine_points, fine_z_vals, combined_z_vals


def volume_rendering(rgb: torch.Tensor, 
                    sigma: torch.Tensor, 
                    z_vals: torch.Tensor,
                    white_background: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform volume rendering to combine samples along a ray.
    
    Args:
        rgb: RGB colors, shape (..., num_samples, 3)
        sigma: Volume densities, shape (..., num_samples, 1)
        z_vals: Depths of samples, shape (..., num_samples)
        white_background: Whether to render with white background
        
    Returns:
        Tuple of:
            - rendered_rgb: Rendered RGB values, shape (..., 3)
            - depth_map: Rendered depths, shape (..., 1)
            - weights: Weights used for rendering, shape (..., num_samples)
    """
    # Get distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    
    # Pad the last distance with the last element
    dists = torch.cat([
        dists, 
        1e10 * torch.ones_like(dists[..., :1])
    ], dim=-1)  # (..., num_samples)
    
    # Multiply by the norm of the ray direction
    # Here we assume rays_directions are already normalized, so this is 1
    
    # Compute alpha (opacity) from density
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (..., num_samples)
    
    # Compute transmittance (probability that ray travels until the sample)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[..., :1]),
            1.0 - alpha[..., :-1]
        ], dim=-1),
        dim=-1
    )  # (..., num_samples)
    
    # Compute weights (contribution of each sample to the final color)
    weights = alpha * transmittance  # (..., num_samples)
    
    # Render RGB and depth
    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)  # (..., 3)
    depth_map = torch.sum(weights * z_vals, dim=-1, keepdim=True)  # (..., 1)
    
    # Add white background if specified
    if white_background:
        acc_map = torch.sum(weights, dim=-1, keepdim=True)  # (..., 1)
        rendered_rgb = rendered_rgb + (1.0 - acc_map) * 1.0
    
    return rendered_rgb, depth_map, weights


def render_rays(model, 
               rays_origin: torch.Tensor, 
               rays_direction: torch.Tensor,
               near: float, 
               far: float,
               num_coarse_samples: int = 64,
               num_fine_samples: int = 128,
               perturb: bool = True,
               white_background: bool = True) -> Dict[str, torch.Tensor]:
    """
    Render rays using NeRF model with hierarchical sampling.
    
    Args:
        model: NeRF model
        rays_origin: Ray origins, shape (batch_size, 3)
        rays_direction: Ray directions, shape (batch_size, 3)
        near: Near bound of the sampling
        far: Far bound of the sampling
        num_coarse_samples: Number of coarse samples per ray
        num_fine_samples: Number of fine samples per ray
        perturb: Whether to add random perturbation to samples
        white_background: Whether to render with white background
        
    Returns:
        Dictionary with:
            - 'rgb_coarse': Coarse rendered RGB
            - 'depth_coarse': Coarse rendered depth
            - 'rgb_fine': Fine rendered RGB (if fine sampling is performed)
            - 'depth_fine': Fine rendered depth (if fine sampling is performed)
    """
    # Sample points along rays for coarse network
    coarse_points, z_vals = sample_points_along_rays(
        rays_origin, 
        rays_direction, 
        near, 
        far, 
        num_coarse_samples, 
        perturb
    )
    
    # Reshape for model input
    batch_size = coarse_points.shape[0]
    coarse_points = coarse_points.reshape(-1, 3)  # (batch_size * num_samples, 3)
    
    # Expand directions to match points
    directions = rays_direction.unsqueeze(1).expand(-1, num_coarse_samples, -1)
    directions = directions.reshape(-1, 3)  # (batch_size * num_samples, 3)
    
    # Forward pass through coarse network
    model_outputs = model(coarse_points, directions)
    
    # Reshape outputs
    coarse_rgb = model_outputs['rgb_coarse'].reshape(batch_size, num_coarse_samples, 3)
    coarse_sigma = model_outputs['sigma_coarse'].reshape(batch_size, num_coarse_samples, 1)
    
    # Volume rendering for coarse samples
    coarse_rgb_rendered, coarse_depth, coarse_weights = volume_rendering(
        coarse_rgb, 
        coarse_sigma, 
        z_vals, 
        white_background
    )
    
    outputs = {
        'rgb_coarse': coarse_rgb_rendered,
        'depth_coarse': coarse_depth
    }
    
    # If fine sampling is requested
    if num_fine_samples > 0:
        # Hierarchical sampling based on coarse weights
        fine_points, fine_z_vals, combined_z_vals = hierarchical_sampling(
            rays_origin, 
            rays_direction, 
            z_vals, 
            coarse_weights, 
            num_fine_samples, 
            perturb
        )
        
        total_samples = num_coarse_samples + num_fine_samples
        fine_points = fine_points.reshape(-1, 3)  # (batch_size * total_samples, 3)
        
        # Expand directions to match points
        directions = rays_direction.unsqueeze(1).expand(-1, total_samples, -1)
        directions = directions.reshape(-1, 3)  # (batch_size * total_samples, 3)
        
        # Forward pass through fine network (actually both networks)
        model_outputs = model(fine_points, directions)
        
        # Reshape outputs
        fine_rgb = model_outputs['rgb_fine'].reshape(batch_size, total_samples, 3)
        fine_sigma = model_outputs['sigma_fine'].reshape(batch_size, total_samples, 1)
        
        # Volume rendering for fine samples
        fine_rgb_rendered, fine_depth, _ = volume_rendering(
            fine_rgb, 
            fine_sigma, 
            combined_z_vals, 
            white_background
        )
        
        outputs.update({
            'rgb_fine': fine_rgb_rendered,
            'depth_fine': fine_depth
        })
    
    return outputs
