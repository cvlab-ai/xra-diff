# XRA-Diff: Guided diffusion for 3D coronaries

"XRA-Diff: Feasibility of guided diffusion towards coronary artery three-dimensional reconstruction"

# Installing dependencies

```
uv sync
uv pip install -e .
```

# Preparing data

```
uv run scripts/prepare_data.py
```

# Scripts

Under script directory:
- `prepare_data.py` - generate projections for training using ImageCAS
- `image2cas.sh` - unpack ImageCAS
- `test_synthetic_right.py` - test a model on the synthetic dataset (right arteries)
- `train_diffusion_right.py` - train diffusion model for right arteries
- `train_diffusion_left.py` - train diffusion model for left arteries
- `visualizer.py` - visualize model output for a given sample

