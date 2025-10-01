#!/bin/bash
set -euo pipefail

RUN="uv run"

EXPERIMENTS=(
    # synthetic
    "scripts/experiments/synthetic/test_synthetic_diffusion_right.py"
    "scripts/experiments/synthetic/test_synthetic_diffusion_left.py"
    "scripts/experiments/synthetic/test_synthetic_gan_right.py"
    "scripts/experiments/synthetic/test_synthetic_gan_left.py"
    "scripts/experiments/synthetic/test_synthetic_unet_right.py"
    "scripts/experiments/synthetic/test_synthetic_unet_left.py"

    # synthetic adaptive
    "scripts/experiments/synthetic/test_synthetic_adaptive_diffusion_right.py"
    "scripts/experiments/synthetic/test_synthetic_adaptive_diffusion_left.py"
    "scripts/experiments/synthetic/test_synthetic_adaptive_gan_right.py"
    "scripts/experiments/synthetic/test_synthetic_adaptive_gan_left.py"
    "scripts/experiments/synthetic/test_synthetic_adaptive_unet_right.py"
    "scripts/experiments/synthetic/test_synthetic_adaptive_unet_left.py"

    # clinical
    "scripts/experiments/clinical/test_clinical_diffusion_right.py"
    "scripts/experiments/clinical/test_clinical_diffusion_left.py"
    "scripts/experiments/clinical/test_clinical_gan_right.py"
    "scripts/experiments/clinical/test_clinical_gan_left.py"
    "scripts/experiments/clinical/test_clinical_unet_right.py"
    "scripts/experiments/clinical/test_clinical_unet_left.py"

    # ablation
    "scripts/experiments/ablation/test_more_projections.py"
    "scripts/experiments/ablation/test_guidance.py"
    "scripts/experiments/ablation/test_sampling_steps.py"
    "scripts/experiments/ablation/test_synthetic_adaptive_diffusion_both.py"

    # other
    "scripts/experiments/other/test_synthetic_motion.py"
)

TOTAL=${#EXPERIMENTS[@]}

for i in "${!EXPERIMENTS[@]}"; do
    n=$((i+1))
    echo -e "\n[ $n / $TOTAL ] Running: ${EXPERIMENTS[$i]}"
    eval "$RUN ${EXPERIMENTS[$i]}"
done

echo -e "\nAll $TOTAL experiments finished!"
