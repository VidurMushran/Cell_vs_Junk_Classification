# add_mask_and_BLUE_features.py

This script takes an existing HDF5 that already contains cropped images in an `images` dataset and a pandas `features` table, generates center-cell masks using the same Cellpose + `channels_to_bgr` approach used in your pipeline, calculates basic masked features and background regionprops features, and writes a new HDF5 that looks more like your `ctcs.hdf5` style files.

It is designed for HDF5 files like:
- `Common_Cell.final.hdf5`
- `ctcs.hdf5`

where the file structure is:
- `images`: shape `(N, 75, 75, 4)`
- `channels`: 4 channel names
- `features`: pandas HDF table with metadata columns such as `slide_id`, `frame_id`, `cell_id`, and either `x/y` or `cell_x/cell_y`

The output HDF5 will contain:
- `images`
- `channels`
- `masks`: shape `(N, 75, 75)`
- `features`: original metadata plus mask-derived features and BLUE embeddings

## What the script adds

For each crop, the script does the following:

1. Reads the 4-channel 75x75 image crop from the input HDF5.
2. Converts channels to BGR using the same helper import style as your current code:
   - blue from `[0, 3]`
   - green from `[2, 3]`
   - red from `[1, 3]`
3. Runs the Cellpose model.
4. Keeps only the object whose label matches the center pixel, just like your existing `get_center_mask` helper.
5. Calculates cell features from the masked event:
   - `area`
   - `eccentricity`
   - `DAPI_mean`
   - `TRITC_mean`
   - `CY5_mean`
   - `FITC_mean`
6. Calculates many `skimage.regionprops`-derived features for the background inside the crop by inverting the cell mask. These are written with a `background_` prefix.
7. Runs the representation learning encoder to generate BLUE features `z0` through `z127` if `encode_model_path` is set.
8. Optionally runs the WBC classifier and/or outlier detection if enabled in the config.

## Expected input columns

The script expects the input HDF5 `features` table to have:
- `slide_id`
- `frame_id`
- either `x` or `cell_x`
- either `y` or `cell_y`

It will preserve all existing metadata columns and standardize coordinates so the output has both:
- `x`, `y`
- `cell_x`, `cell_y`

If `image_id` is missing, it will be created.
If `cell_id` is missing, it will also be created.

## Files included

- `add_mask_and_BLUE_features.py`
- `mask_and_BLUE_config.yml`

## How to run

```bash
python add_mask_and_BLUE_features.py --config mask_and_BLUE_config.yml
```

## Example config

```yaml
input_h5: /mnt/deepstore/Vidur/Junk_Classification/data/Common_Cell/Common_Cell.final.hdf5
output_h5: /mnt/deepstore/Vidur/Junk_Classification/data/Common_Cell/Common_Cell.with_masks_and_BLUE.hdf5

image_dataset_key: images
channels_dataset_key: channels
channels: [DAPI, TRITC, CY5, FITC]

mask_model_path: /mnt/deepstore/shared_models/purple/cellpose_model
encode_model_path: /mnt/deepstore/shared_models/purple/representation_learning_04_28.pth

enable_wbc: false
classifier_path: /mnt/deepstore/shared_models/purple/wbc_model_0225.pth
enable_outlier_detection: false

device: cuda:0
workers: 1
feature_workers: 8
dataloader_workers: 0

cellpose_diameter: 20
cellpose_batch_size: 8
mask_chunk_size: 256
feature_chunk_size: 512
inference_batch: 4096
```

## Important notes on workers and speed

### Segmentation workers

Each segmentation worker loads its own Cellpose model.
That means:
- `workers: 1` is the safest default for GPU use
- on CPU, you can usually increase `workers`
- on GPU, using `workers > 1` may help in some environments, but it can also use much more VRAM because each worker holds a model copy

A good starting point is:
- GPU: `workers: 1`
- CPU: `workers: 4` to `workers: 16`, depending on the machine

### Feature workers

`feature_workers` controls only the CPU-side basic feature and background feature extraction. This is usually safe to increase.

### Dataloader workers

`dataloader_workers` is only used for BLUE embedding inference and optional WBC scoring.

## Output format details

The output HDF5 is written fresh and contains:
- a copy of the input `images`
- a copy of the input `channels`
- a new `masks` dataset with binary center-object masks
- a new `features` table with original metadata plus added columns

That means the script does not modify your original input HDF5. The script requires `output_h5` to be a different path from `input_h5`.

## Background feature behavior

The script inverts the binary cell mask and runs `skimage.regionprops` on the background region. The resulting columns are prefixed with `background_`.

Examples include properties such as:
- background area-like measurements
- centroid and bbox values
- eccentricity and axis length values
- intensity statistics for each channel
- moment-derived scalar features when available

Very large non-scalar regionprops outputs such as raw pixel coordinate arrays or full binary images are intentionally skipped because they are not practical to store as HDF table columns.

## Consistency with your existing pipeline

This script intentionally mirrors the existing codebase where practical:
- same import style
- same `channels_to_bgr` helper
- same `load_model` and `get_embeddings` pathway
- same Cellpose center-mask logic
- same general HDF5 feature-table layout

The main difference is that this script starts from an HDF5 of pre-extracted 75x75 crops instead of going back to raw slide tile directories.

## Recommended validation

After running, validate the structure with commands like:

```bash
h5ls /path/to/output.hdf5
python -c "import pandas as pd; print(pd.read_hdf('/path/to/output.hdf5', key='features').head())"
```

You should see:
- `images`
- `channels`
- `masks`
- `features`

## Common issues

### 1. Input `features` rows do not match image count

The script requires:
- one metadata row per image crop

If the `features` table has a different number of rows than `images.shape[0]`, the script will stop with an error.

### 2. Wrong image shape

The script expects:
- `(N, 75, 75, 4)`

If your images are a different crop size or channel count, update the script before using it.

### 3. GPU memory issues

Reduce:
- `workers`
- `mask_chunk_size`
- `cellpose_batch_size`
- `inference_batch`

### 4. Missing encoder model

If you only want masks and basic/background features, you can leave `encode_model_path` blank or remove it from the config. In that case, the script will skip BLUE embeddings.
