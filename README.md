# NeRF: Neural Radiance Fields

This project implements Neural Radiance Fields (NeRF) for 3D scene reconstruction from multiple images, as introduced in the paper ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934).

## Project Structure

```
NeRF/
├── data/                # Directory for datasets
├── src/                 # Source code
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── model.py         # NeRF model architecture
│   ├── ray_utils.py     # Ray sampling and processing
│   ├── training.py      # Training pipeline
│   ├── rendering.py     # Rendering and evaluation
│   ├── utils.py         # Utility functions
│   ├── main.py          # Main script for running the pipeline
│   └── __init__.py      # Package initialization
├── config.yaml          # Configuration file
└── README.md            # Project documentation
```

## Features

- **Complete NeRF Pipeline**: Data loading, model training, and novel view synthesis
- **Hierarchical Sampling**: Implementation of the coarse-to-fine sampling strategy
- **Flexible Configuration**: Easily configurable through YAML files or command-line arguments
- **Visualization**: Tools for visualizing results, including depth maps and RGB renderings
- **Evaluation Metrics**: PSNR calculation for quantitative evaluation

## Dependencies

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm
- PyYAML
- imageio
- OpenCV (cv2)

## Installation

```bash
# Clone the repository
git clone https://github.com/saomyaraj/NeRF.git
cd nerf

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

The NeRF model expects data in a specific format. For each scene, you need:

1. A set of images from different viewpoints
2. Camera parameters for each image

The expected directory structure is:

```
data/
└── scene_name/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    └── images/
        ├── img_0001.png
        ├── img_0002.png
        └── ...
```

Each JSON file should contain camera intrinsics and extrinsics (poses) for each image in the corresponding split.

Example JSON format:

```json
{
  "camera": {
    "width": 800,
    "height": 800,
    "focal_length": 560.0
  },
  "frames": [
    {
      "file_path": "images/img_0001.png",
      "transform_matrix": {
        "rotation": [[...], [...], [...]],
        "translation": [x, y, z]
      }
    },
    ...
  ]
}
```

## Usage

### Training

```bash
python -m src.main --mode train --config config.yaml
```

### Evaluation

```bash
python -m src.main --mode eval --config config.yaml --checkpoint output/checkpoints/best.pt
```

### Rendering Novel Views

```bash
python -m src.main --mode render --config config.yaml --checkpoint output/checkpoints/best.pt --render_path spiral
```

## Configuration

You can modify `config.yaml` to adjust various parameters:

- **Data settings**: paths, image size
- **Model settings**: encoding frequencies, network size
- **Training settings**: batch size, learning rate, number of epochs
- **Rendering settings**: near/far plane, chunk size, rendering path type

## Results

After training, results are saved in the `output` directory:

- `output/checkpoints/`: Model checkpoints
- `output/logs/`: TensorBoard logs for training monitoring
- `output/results/`: Rendered images and videos
- `output/configs/`: Saved configuration files

## Examples

### Training a model

```bash
python -m src.main --mode train --data_dir ./data/lego --output_dir ./output/lego --num_epochs 100
```

### Rendering a spiral path of novel views

```bash
python -m src.main --mode render --data_dir ./data/lego --output_dir ./output/lego --checkpoint ./output/lego/checkpoints/best.pt --render_path spiral --render_frames 200
```

## License

[MIT License](LICENSE)

## Acknowledgements

- The original NeRF paper: ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934)
- Implementation inspired by various open-source NeRF projects
