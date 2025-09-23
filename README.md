# XRA-Diff: Guided diffusion for 3D coronaries

"XRA-Diff: Feasibility of guided diffusion towards coronary artery three-dimensional reconstruction"

# Installing dependencies

```
uv sync
uv pip install -e .
```

# Preparing data

```
uv run scripts/dataset/prepare_data.py
uv run scrpits/dataset/make_split.py
```

# Scripts

Under `scripts` directory:
- `experiments/` - scripts for running experiments
    - `test_synthetic_right.py` - test a model on the synthetic dataset (right arteries)
    - `test_synthetic_baseline_right.py` - test a baseline on the synthetic dataset (right arteries)
    - `test_synthetic_adaptive_right.py` - test a model on the synthetic dataset with adaptive threshold (right arteries)
    - `test_clinical_right.py` - test a model on the clinical dataset (right arteries)
    - `test_clinical_baseline_right.py` - test a baseline on the clinical dataset (right)
- `demos/` - scripts for qualitative evaluation
    - `synthetic_visualizer.py` - synthetic data visualizer (see --help)
    - `clinical_visualizer.py` - clinical data visualizer (see --help)
- `dataset/` - dataset preparation
    - `image2cas.sh` - extraction of ImageCAS 
    - `prepare_data.py` - simulates the image acquisition using cone beam geometry, generates synthetic dataset
    - `make_split.py` - splits the above dataset to train and valitation splits
- `training/` - scripts for training models
    - `train_diffusion_right.py` - train diffusion model for right arteries
    - `train_diffusion_left.py` - train diffusion model for left arteries
    - `train_gan_right.py` - train gan model for right arteries (baseline)
    - `train_gan_left.py` - train gan model for left arteries (baseline)

# Notebooks

Under `notebooks` directory:
- `synthetic_results.ipynb` (3.1. SOTA)
- `clinical_results.ipynb` (3.3. )

