"""
Main script for NeRF training and evaluation.

This script serves as the entry point for the NeRF pipeline, including data loading,
model creation, training, and rendering novel views.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import time
import sys
import datetime

from src.data_loader import prepare_data_for_nerf
from src.model import NeRFModel
from src.training import NeRFTrainer
from src.rendering import NeRFRenderer, evaluate_model
from src.utils import (
    save_config, 
    load_config, 
    create_folder_structure,
    prepare_output_image,
    visualize_depth
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='NeRF Training and Evaluation')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'render'],
                        help='Operation mode')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 400],
                        help='Image size (width, height) for resizing')
    
    # Model settings
    parser.add_argument('--pos_encoding_freqs', type=int, default=10,
                        help='Number of frequency bands for position encoding')
    parser.add_argument('--dir_encoding_freqs', type=int, default=4,
                        help='Number of frequency bands for direction encoding')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of the MLP')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of layers in the MLP')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training (number of rays)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--white_background', action='store_true',
                        help='Render with white background')
    
    # Rendering settings
    parser.add_argument('--near', type=float, default=2.0,
                        help='Near clipping plane')
    parser.add_argument('--far', type=float, default=6.0,
                        help='Far clipping plane')
    parser.add_argument('--chunk_size', type=int, default=32*1024,
                        help='Chunk size for rendering')
    parser.add_argument('--render_path', type=str, default='spiral',
                        choices=['spiral', '360', 'custom'],
                        help='Path type for novel view rendering')
    parser.add_argument('--render_frames', type=int, default=120,
                        help='Number of frames to render')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for loading weights')
    parser.add_argument('--validate_every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training/inference')
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Load config from file if specified
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            args_dict = vars(args)
            # Update args with config values, but don't overwrite explicitly set args
            for key, value in config.items():
                if key in args_dict and args_dict[key] == parser.get_default(key):
                    args_dict[key] = value
    
    return args


def train_nerf(args):
    """
    Train a NeRF model.
    
    Args:
        args: Command-line arguments
    """
    print("Starting NeRF training...")
    
    # Create folder structure
    folders = create_folder_structure(args.output_dir)
    
    # Save configuration
    config_path = os.path.join(folders['configs'], 'config.yaml')
    save_config(vars(args), config_path)
    
    # Prepare data
    print("Loading data...")
    data_loaders = prepare_data_for_nerf(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=1,  # Process one image at a time
        preload=False
    )
    
    # Initialize model
    print("Creating model...")
    model = NeRFModel(
        pos_encoding_freqs=args.pos_encoding_freqs,
        dir_encoding_freqs=args.dir_encoding_freqs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    # Initialize trainer
    trainer = NeRFTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('val'),  # May be None
        learning_rate=args.learning_rate,
        log_dir=folders['logs'],
        ckpt_dir=folders['checkpoints'],
        use_amp=args.use_amp,
        white_background=args.white_background,
        device=args.device
    )
    
    # Load checkpoint if specified
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train the model
    print(f"Training for {args.num_epochs} epochs...")
    trainer.train(
        num_epochs=args.num_epochs,
        validate_every=args.validate_every,
        save_every=args.save_every,
        chunk_size=args.chunk_size
    )
    
    print("Training completed!")


def evaluate_nerf(args):
    """
    Evaluate a trained NeRF model.
    
    Args:
        args: Command-line arguments
    """
    print("Starting NeRF evaluation...")
    
    # Prepare data
    print("Loading test data...")
    data_loaders = prepare_data_for_nerf(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=1,  # Process one image at a time
        preload=False
    )
    
    if 'test' not in data_loaders:
        print("No test split found, using validation set...")
        test_loader = data_loaders.get('val')
        if test_loader is None:
            print("No validation set found. Exiting...")
            return
    else:
        test_loader = data_loaders['test']
    
    # Initialize model
    print("Creating model...")
    model = NeRFModel(
        pos_encoding_freqs=args.pos_encoding_freqs,
        dir_encoding_freqs=args.dir_encoding_freqs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    # Load checkpoint
    if args.checkpoint is None:
        print("No checkpoint specified. Please provide a checkpoint with --checkpoint.")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=args.device
    )
    
    # Print and save metrics
    print("Evaluation results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save metrics
    folders = create_folder_structure(args.output_dir)
    metrics_path = os.path.join(folders['results'], 'metrics.yaml')
    save_config(metrics, metrics_path)
    
    print("Evaluation completed!")


def render_novel_views(args):
    """
    Render novel views from a trained NeRF model.
    
    Args:
        args: Command-line arguments
    """
    print("Starting novel view rendering...")
    
    # Prepare data (to get camera parameters)
    print("Loading data...")
    data_loaders = prepare_data_for_nerf(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=1,  # Process one image at a time
        preload=False
    )
    
    # Use training set to get camera parameters
    dataset = data_loaders['train'].dataset
    
    # Extract image dimensions and focal length
    sample = dataset[0]
    H, W = sample['image'].shape[:2]
    intrinsics = sample['intrinsics']
    focal = intrinsics[0].item()  # Assuming fx = fy
    
    # Initialize model
    print("Creating model...")
    model = NeRFModel(
        pos_encoding_freqs=args.pos_encoding_freqs,
        dir_encoding_freqs=args.dir_encoding_freqs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    # Load checkpoint
    if args.checkpoint is None:
        print("No checkpoint specified. Please provide a checkpoint with --checkpoint.")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Initialize renderer
    renderer = NeRFRenderer(
        model=model,
        device=args.device,
        chunk_size=args.chunk_size,
        near=args.near,
        far=args.far,
        white_background=args.white_background
    )
    
    # Create rendering path
    if args.render_path == 'spiral':
        # Get scene center and radius from training poses
        train_poses = torch.stack([dataset[i]['pose'] for i in range(len(dataset))], dim=0)
        center = train_poses[:, :3, 3].mean(0).numpy()
        radius = 1.0  # Can be adjusted based on the scene scale
        
        render_poses = NeRFRenderer.create_spiral_rendering_path(
            center=center,
            radius=radius,
            n_frames=args.render_frames
        )
    elif args.render_path == '360':
        render_poses = NeRFRenderer.create_360_rendering_path(
            radius=4.0,  # Can be adjusted based on the scene scale
            n_frames=args.render_frames
        )
    else:
        # Use test poses for rendering
        if 'test' in data_loaders:
            test_dataset = data_loaders['test'].dataset
            render_poses = [test_dataset[i]['pose'] for i in range(len(test_dataset))]
        else:
            print("No test split found, using validation poses...")
            if 'val' in data_loaders:
                val_dataset = data_loaders['val'].dataset
                render_poses = [val_dataset[i]['pose'] for i in range(len(val_dataset))]
            else:
                print("No validation set found. Using training poses...")
                render_poses = [dataset[i]['pose'] for i in range(len(dataset))]
    
    # Render images
    print(f"Rendering {len(render_poses)} novel views...")
    rendered = renderer.render_path(
        render_poses=render_poses,
        height=H,
        width=W,
        focal_length=focal
    )
    
    # Save results
    folders = create_folder_structure(args.output_dir)
    video_path = os.path.join(folders['results'], 'novel_views.mp4')
    renderer.save_video(rendered['rgb'], video_path, fps=30)
    
    # Also save depth video if available
    if 'depth' in rendered:
        depth_colored = np.stack([
            visualize_depth(torch.from_numpy(depth)) 
            for depth in rendered['depth']
        ], axis=0)
        depth_video_path = os.path.join(folders['results'], 'novel_views_depth.mp4')
        renderer.save_video(depth_colored, depth_video_path, fps=30)
    
    print("Rendering completed!")
    print(f"Results saved to {folders['results']}")


def main():
    """Main function."""
    args = parse_args()
    
    # Print configuration
    print("=" * 80)
    print("NeRF Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = 'cpu'
    
    # Run the selected mode
    if args.mode == 'train':
        train_nerf(args)
    elif args.mode == 'eval':
        evaluate_nerf(args)
    elif args.mode == 'render':
        render_novel_views(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
