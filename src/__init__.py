"""
NeRF (Neural Radiance Fields) implementation.

This module contains a complete implementation of Neural Radiance Fields
for 3D scene reconstruction from multiple views.

Modules:
    - data_loader: Data loading and preprocessing
    - model: NeRF model architecture
    - ray_utils: Ray sampling and processing
    - training: Training pipeline
    - rendering: Rendering and evaluation
    - utils: Utility functions
"""

from .data_loader import (
    CameraIntrinsics,
    CameraPose,
    NeRFDataset,
    create_data_loader,
    prepare_data_for_nerf
)

from .model import (
    PositionalEncoding,
    NeRF,
    NeRFModel
)

from .ray_utils import (
    get_ray_directions,
    get_rays,
    sample_points_along_rays,
    sample_pdf,
    hierarchical_sampling,
    volume_rendering,
    render_rays
)

from .training import (
    NeRFTrainer
)

from .rendering import (
    NeRFRenderer,
    compute_metrics,
    evaluate_model
)

from .utils import (
    visualize_depth,
    create_visualization_grid,
    save_config,
    load_config,
    create_folder_structure,
    normalize_poses
)

__all__ = [
    'CameraIntrinsics',
    'CameraPose',
    'NeRFDataset',
    'create_data_loader',
    'prepare_data_for_nerf',
    'PositionalEncoding',
    'NeRF',
    'NeRFModel',
    'get_ray_directions',
    'get_rays',
    'sample_points_along_rays',
    'sample_pdf',
    'hierarchical_sampling',
    'volume_rendering',
    'render_rays',
    'NeRFTrainer',
    'NeRFRenderer',
    'compute_metrics',
    'evaluate_model',
    'visualize_depth',
    'create_visualization_grid',
    'save_config',
    'load_config',
    'create_folder_structure',
    'normalize_poses'
]
