"""
Training module for NeRF.

This module contains the training pipeline for the NeRF model, including
the loss function, training loop, and validation.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, List, Optional, Union, Callable

from .model import NeRFModel
from .ray_utils import get_ray_directions, get_rays, render_rays


class NeRFTrainer:
    """Trainer class for NeRF models."""
    
    def __init__(self,
                model: NeRFModel,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                learning_rate: float = 5e-4,
                log_dir: str = './logs',
                ckpt_dir: str = './checkpoints',
                use_amp: bool = False,
                white_background: bool = True,
                device: str = 'cuda'):
        """
        Initialize NeRF trainer.
        
        Args:
            model: NeRF model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            learning_rate: Learning rate for optimizer
            log_dir: Directory for TensorBoard logs
            ckpt_dir: Directory for model checkpoints
            use_amp: Whether to use automatic mixed precision
            white_background: Whether scene has white background
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.use_amp = use_amp
        self.white_background = white_background
        
        # Ensure device is either 'cuda' or 'cpu'
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)
        
        # Setup AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Setup dirs
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Setup TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # State tracking
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
    
    def train_step(self, 
                 batch: Dict[str, torch.Tensor], 
                 chunk_size: int = 1024 * 32) -> Dict[str, torch.Tensor]:
        """
        Execute a single training step.
        
        Args:
            batch: Dictionary of training data
            chunk_size: Number of rays to process at once
            
        Returns:
            Dictionary with loss values
        """
        # Get data
        images = batch['image'].to(self.device)  # (B, H, W, 3)
        poses = batch['pose'].to(self.device)    # (B, 4, 4)
        intrinsics = batch['intrinsics'].to(self.device)  # (B, 4)
        
        batch_size, height, width, _ = images.shape
        
        # Prepare results containers
        coarse_rgb_loss = 0
        fine_rgb_loss = 0
        psnr = 0
        
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Process each image in the batch
            for b in range(batch_size):
                # Camera intrinsics
                fx, fy, cx, cy = intrinsics[b]
                
                # Get ray directions in camera frame
                directions = get_ray_directions(
                    height=height,
                    width=width,
                    focal_length=fx.item(),
                    center=(cx.item(), cy.item())
                ).to(self.device)  # (H, W, 3)
                
                # Get rays in world frame
                rays_origin, rays_direction = get_rays(directions, poses[b])
                
                # Flatten rays for batched processing
                rays_o = rays_origin.reshape(-1, 3)  # (H*W, 3)
                rays_d = rays_direction.reshape(-1, 3)  # (H*W, 3)
                rgb_target = images[b].reshape(-1, 3)  # (H*W, 3)
                
                # Process in chunks to avoid OOM
                results = {}
                for i in range(0, rays_o.shape[0], chunk_size):
                    # Get chunk
                    chunk_rays_o = rays_o[i:i+chunk_size]  # (chunk_size, 3)
                    chunk_rays_d = rays_d[i:i+chunk_size]  # (chunk_size, 3)
                    chunk_rgb_target = rgb_target[i:i+chunk_size]  # (chunk_size, 3)
                    
                    # Render chunk
                    chunk_results = render_rays(
                        model=self.model,
                        rays_origin=chunk_rays_o,
                        rays_direction=chunk_rays_d,
                        near=2.0,  # Near clipping plane
                        far=6.0,   # Far clipping plane
                        num_coarse_samples=64,
                        num_fine_samples=128,
                        perturb=True,
                        white_background=self.white_background
                    )
                    
                    # Compute loss for this chunk
                    coarse_loss = F.mse_loss(
                        chunk_results['rgb_coarse'], 
                        chunk_rgb_target
                    )
                    
                    # Add fine loss if available
                    if 'rgb_fine' in chunk_results:
                        fine_loss = F.mse_loss(
                            chunk_results['rgb_fine'], 
                            chunk_rgb_target
                        )
                        # Update total losses
                        fine_rgb_loss += fine_loss.item()
                    else:
                        fine_loss = 0
                    
                    # Total loss
                    loss = coarse_loss + fine_loss
                    
                    # Accumulate for backward
                    loss = loss / batch_size
                    self.scaler.scale(loss).backward()
                    
                    # Update metrics
                    coarse_rgb_loss += coarse_loss.item()
                    
                    # Compute PSNR for this chunk
                    if 'rgb_fine' in chunk_results:
                        mse = F.mse_loss(chunk_results['rgb_fine'], chunk_rgb_target)
                    else:
                        mse = F.mse_loss(chunk_results['rgb_coarse'], chunk_rgb_target)
                    
                    psnr += -10.0 * torch.log10(mse).item()
        
        # Average metrics over the batch
        coarse_rgb_loss /= batch_size
        fine_rgb_loss /= batch_size
        psnr /= batch_size
        
        # Update model parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Increment step counter
        self.global_step += 1
        
        return {
            'loss/coarse_rgb': coarse_rgb_loss,
            'loss/fine_rgb': fine_rgb_loss,
            'metric/psnr': psnr
        }
    
    def validate(self, 
                chunk_size: int = 1024 * 32) -> Dict[str, torch.Tensor]:
        """
        Validate the model on the validation set.
        
        Args:
            chunk_size: Number of rays to process at once
            
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        # Set model to eval mode
        self.model.eval()
        
        val_loss = 0
        val_psnr = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get data
                images = batch['image'].to(self.device)  # (B, H, W, 3)
                poses = batch['pose'].to(self.device)    # (B, 4, 4)
                intrinsics = batch['intrinsics'].to(self.device)  # (B, 4)
                
                batch_size, height, width, _ = images.shape
                
                # Process each image in the batch
                for b in range(batch_size):
                    # Camera intrinsics
                    fx, fy, cx, cy = intrinsics[b]
                    
                    # Get ray directions in camera frame
                    directions = get_ray_directions(
                        height=height,
                        width=width,
                        focal_length=fx.item(),
                        center=(cx.item(), cy.item())
                    ).to(self.device)  # (H, W, 3)
                    
                    # Get rays in world frame
                    rays_origin, rays_direction = get_rays(directions, poses[b])
                    
                    # Flatten rays for batched processing
                    rays_o = rays_origin.reshape(-1, 3)  # (H*W, 3)
                    rays_d = rays_direction.reshape(-1, 3)  # (H*W, 3)
                    rgb_target = images[b].reshape(-1, 3)  # (H*W, 3)
                    
                    # Process in chunks to avoid OOM
                    rgb_predicted = []
                    for i in range(0, rays_o.shape[0], chunk_size):
                        # Get chunk
                        chunk_rays_o = rays_o[i:i+chunk_size]  # (chunk_size, 3)
                        chunk_rays_d = rays_d[i:i+chunk_size]  # (chunk_size, 3)
                        
                        # Render chunk
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            chunk_results = render_rays(
                                model=self.model,
                                rays_origin=chunk_rays_o,
                                rays_direction=chunk_rays_d,
                                near=2.0,  # Near clipping plane
                                far=6.0,   # Far clipping plane
                                num_coarse_samples=64,
                                num_fine_samples=128,
                                perturb=False,  # No perturbation during validation
                                white_background=self.white_background
                            )
                        
                        # Use fine results if available, otherwise coarse
                        if 'rgb_fine' in chunk_results:
                            rgb_predicted.append(chunk_results['rgb_fine'])
                        else:
                            rgb_predicted.append(chunk_results['rgb_coarse'])
                    
                    # Combine chunks
                    rgb_predicted = torch.cat(rgb_predicted, dim=0)
                    
                    # Compute metrics
                    mse = F.mse_loss(rgb_predicted, rgb_target)
                    val_loss += mse.item()
                    val_psnr += -10.0 * torch.log10(mse).item()
        
        # Average metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        val_psnr /= num_batches
        
        return {
            'val/loss': val_loss,
            'val/psnr': val_psnr
        }
    
    def train(self, 
             num_epochs: int, 
             validate_every: int = 1,
             save_every: int = 10,
             chunk_size: int = 1024 * 32) -> None:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            validate_every: Run validation every N epochs
            save_every: Save checkpoint every N epochs
            chunk_size: Number of rays to process at once
        """
        # Start training
        start_time = time.time()
        
        for epoch in range(self.epoch, self.epoch + num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            epoch_start_time = time.time()
            
            # Train on all batches
            epoch_loss = 0
            epoch_psnr = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Run train step
                metrics = self.train_step(batch, chunk_size=chunk_size)
                
                # Update epoch metrics
                epoch_loss += metrics['loss/fine_rgb'] if metrics['loss/fine_rgb'] > 0 else metrics['loss/coarse_rgb']
                epoch_psnr += metrics['metric/psnr']
                num_batches += 1
                
                # Log batch metrics
                self.writer.add_scalar('train/batch_loss', metrics['loss/coarse_rgb'], self.global_step)
                self.writer.add_scalar('train/batch_fine_loss', metrics['loss/fine_rgb'], self.global_step)
                self.writer.add_scalar('train/batch_psnr', metrics['metric/psnr'], self.global_step)
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}/{self.epoch + num_epochs - 1} "
                          f"[{batch_idx}/{len(self.train_loader)}] "
                          f"Loss: {metrics['loss/coarse_rgb']:.4f} "
                          f"PSNR: {metrics['metric/psnr']:.2f} dB")
            
            # Calculate epoch average metrics
            epoch_loss /= num_batches
            epoch_psnr /= num_batches
            
            # Log epoch metrics
            self.writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            self.writer.add_scalar('train/epoch_psnr', epoch_psnr, epoch)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s. "
                  f"Avg Loss: {epoch_loss:.4f}, "
                  f"Avg PSNR: {epoch_psnr:.2f} dB")
            
            # Validate
            if epoch % validate_every == 0 and self.val_loader is not None:
                val_metrics = self.validate(chunk_size=chunk_size)
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(key, value, epoch)
                
                # Print validation results
                print(f"Validation - Loss: {val_metrics['val/loss']:.4f}, "
                      f"PSNR: {val_metrics['val/psnr']:.2f} dB")
                
                # Save best model
                if val_metrics['val/psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['val/psnr']
                    self.save_checkpoint(os.path.join(self.ckpt_dir, 'best.pt'))
                    print(f"New best model saved with PSNR: {self.best_psnr:.2f} dB")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(os.path.join(self.ckpt_dir, f'epoch_{epoch:04d}.pt'))
        
        # Save final model
        self.save_checkpoint(os.path.join(self.ckpt_dir, 'final.pt'))
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/3600:.2f} hours.")
        print(f"Best validation PSNR: {self.best_psnr:.2f} dB")
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_psnr': self.best_psnr
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found at {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint['best_psnr']
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {self.epoch}, global step {self.global_step}")
        print(f"Best PSNR so far: {self.best_psnr:.2f} dB")
