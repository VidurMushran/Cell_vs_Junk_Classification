# junk_qc

Cross-validated training and comparison of junk-vs-rest classifiers for immunofluorescence images.

## Layout

- `junk_qc/data`: Dataset utilities, index building, CV splits, QUALIFAI-style augmentations.
- `junk_qc/models`: Model architectures and loss functions.
- `junk_qc/train`: Cross-validated training loop (5-fold) + final model on all data.
- `junk_qc/inference`: Inference + persistence of predictions into HDF5.
- `junk_qc/scripts`: CLI entrypoints for training and comparisons.

## Basic usage

From the directory containing the `junk_qc` folder, run:

```bash
# Example: train a simple CNN with QUALIFAI-style augmentations
python -m junk_qc.scripts.main_train \
    --root /path/to/root \    --arch simple_cnn

# Example: train a VGG19-BN backbone, pretrained on ImageNet, last 4 layers unfrozen
python -m junk_qc.scripts.main_train \
    --root /path/to/root \    --arch vgg19_bn \    --pretrained \    --unfreeze_last 4
```

Each run trains 5 cross-validation folds and one final model on all data. Per-fold metrics,
aggregated CV metrics, and final model checkpoints are saved in the experiment output folder.

To compare augmentation vs no-augmentation, after you have trained two runs (one with default
augmentations and one with `--no_qualifai_aug`), run:

```bash
python -m junk_qc.scripts.compare_aug_vs_noaug \    --aug_metrics path/to/experiment_aug/cv_metrics_*.json \    --noaug_metrics path/to/experiment_noaug/cv_metrics_*.json
```

To compare different model families (simple CNN, rotation-invariant CNN, ImageNet-pretrained
backbone with last 4 layers unfrozen), run:

```bash
python -m junk_qc.scripts.compare_models \    --simple path/to/simple_run/cv_metrics_*.json \    --rot path/to/rot_run/cv_metrics_*.json \    --pretrained path/to/pre_run/cv_metrics_*.json
```

For inference, you can use:

```bash
python -m junk_qc.scripts.run_inference \    --checkpoint path/to/fold_or_final_checkpoint.pt \    --root /path/to/root
```

Predictions and scores are saved into the `/features` table in each HDF5 file with column names:

- `label_{model_type}_{timestamp}_{fold}`
- `score_{model_type}_{timestamp}_{fold}`

where `model_type` encodes the architecture and augmentation (e.g. `simple_cnn_aug`),
`timestamp` matches the experiment folder timestamp, and `fold` is `1`–`5` for CV models or `all`
for the final model.
