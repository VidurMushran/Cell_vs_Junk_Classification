"""
junk_qc: Cross-validated training and comparison of junk-vs-rest classifiers.

This package is designed to work with rare-cell imaging HDF5 files and provides:
- Dataset + augmentation utilities
- Model architectures (simple CNN, rotation-invariant CNN, ImageNet-pretrained backbones)
- Cross-validated training (5-fold)
- Inference helpers that persist predictions back into HDF5 feature tables
- Scripts to compare augmentation vs no-augmentation and different model families
"""
