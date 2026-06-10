# ST-HF-VVI-ReID

Faithful  reproduction and extension of Spatial-Temporal High-Frequency Learning for Video-based Visible-Infrared Person Re-Identification.

Status: In progress

Current focus:
- Baseline ResNet-50 video ReID model
- Fixed spatial-temporal high-frequency filtering module
- HITSZ-style training/evaluation pipeline
- Reproducible experiments and ablations

Tech stack:
Python, PyTorch, TorchVision, NumPy, YAML, pytest

Repository structure:
- models/ — backbone, baseline, STHF model modules
- losses/ — ID loss, triplet loss, combined ReID loss
- engine/ — training loop
- scripts/ — sanity checks, training, evaluation entry points
- configs/ — baseline/fixed/adaptive experiment configs
- tests/ — forward-pass and loss contract tests

Current validation:
- Model sanity checks pass for baseline/fixed/adaptive configs
- Forward contract tests pass
- Loss tests pass
- Debug training step runs successfully

Planned:
- Dataset loader integration
- Full baseline training
- Fixed STHF training
- Evaluation metrics: Rank-k, mAP
- Ablation comparison
