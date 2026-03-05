# rebuild_outlier_hdf5.py

This script rebuilds a full HDF5 file with `images`, `masks`, `channels`, and a `features` table from parquet outputs produced by either:

1. The BLUE pipeline outlier parquet, such as `0AD5601_copod_10000.parquet`
2. The RED-BLUE pipeline coordinate parquet, such as `rarest_coords_filtered.parquet`

The output HDF5 is meant to look more like datasets such as `ctcs.hdf5`, where the HDF5 contains:

- `images`: `(N, 75, 75, 4)`
- `masks`: `(N, 75, 75, 1)`
- `channels`: `['DAPI', 'TRITC', 'CY5', 'FITC']`
- `features`: pandas HDF table with metadata, basic features, regionprops-derived features, and optional BLUE embeddings

## What the script does

For each input row:

1. Reads the source slide tiles from the Oncoscope slide directory
2. Rebuilds the 75x75 crop around the provided `frame_id`, `x`, and `y`
3. Segments the crop with the Cellpose model
4. Keeps only the object closest to the crop center using the center-label rule
5. Saves the crop mask to the `masks` dataset
6. Computes masked basic features:
   - `area`
   - `eccentricity`
   - `DAPI_mean`
   - `TRITC_mean`
   - `CY5_mean`
   - `FITC_mean`
7. Computes additional `cell_prop_*` skimage regionprops features on the foreground mask
8. Computes `background_*` regionprops and intensity summary features on the inverted mask inside the crop
9. Optionally runs the BLUE encoder model to add `z0..z127`
10. Writes everything into one HDF5

## Supported input styles

### 1) BLUE outlier parquet

Example columns:

- `z0..z127`
- `area`, `eccentricity`, `DAPI_mean`, `TRITC_mean`, `CY5_mean`, `FITC_mean`
- `frame_id`, `cell_id`, `x`, `y`, `slide_id`
- `copod_score`

The script preserves the original columns and overwrites or fills in computed mask-based features from the rebuilt crops.

### 2) RED-BLUE rare coordinate parquet

Example columns:

- `frame_id`, `x`, `y`
- `recon_err_DAPI`, `recon_err_TRITC`, `recon_err_CY5`, `recon_err_FITC`
- `filtered_out`, `reconstruction_error`

If `cell_id` is missing, the script creates one from the row order.
If `slide_id` is missing, the script tries to infer it from:

1. The YAML config `slide_id`
2. The parquet filename stem before the first underscore
3. The parent directory name

## Source data assumptions

The script assumes the source imagery is available in the same tile layout used by your extraction code.
It searches for slide directories like:

`/mnt/<something>/Oncoscope/tubeID_<first5digits>/*/slideID_<slide_id>/bzScanner/proc`

Inside that directory it expects either:

- `Tile000001.tif`, `Tile000002.tif`, ...
- or `Tile000001.jpg`, `Tile000002.jpg`, ...

If you already know the exact image directory, set `images_path` in the YAML config and the script will use that directly.

## Files

- `rebuild_outlier_hdf5.py` - main script
- `rebuild_outlier_hdf5_config.yml` - example config

## How to run

```bash
python rebuild_outlier_hdf5.py --config rebuild_outlier_hdf5_config.yml
```

## Important config fields

### Required

- `input_parquet`: input BLUE outlier parquet or RED-BLUE coordinate parquet
- `output_h5`: output HDF5 path to create
- `mask_model_path`: Cellpose model path

### Optional but commonly needed

- `slide_id`: required when it cannot be inferred from the parquet or path
- `images_path`: explicit tile directory for the slide
- `encode_model_path`: BLUE embedding model path if you want `z0..z127`
- `device`: `cuda:0`, `cuda:1`, or `cpu`
- `workers`: multiprocessing workers for crop extraction
- `feature_workers`: multiprocessing workers for regionprops calculation
- `embedding_workers`: dataloader workers for embedding inference

## Output feature columns

The output `features` table keeps the original parquet columns and adds or updates:

- `image_id`
- `slide_id`
- `frame_id`
- `cell_id`
- `x`
- `y`
- `area`
- `eccentricity`
- `DAPI_mean`
- `TRITC_mean`
- `CY5_mean`
- `FITC_mean`
- `cell_prop_*`
- `background_*`
- `z0..z127` if `encode_model_path` is provided

## Notes about consistency

This script intentionally copies the same overall design used by your existing codebase:

- uses `slideutils.utils.frame.Frame` for tile reading and crop extraction
- uses the same channel order and tile starts
- uses the same `channels_to_bgr`, `load_model`, and `get_embeddings` utilities from `Final_DeepPhenotyping`
- uses the same Cellpose style of segmentation followed by center-object selection

## Example BLUE usage

```yaml
input_parquet: /mnt/deepstore/Vidur/Junk_Classification/test_data/BLUE_output/outlier_outputs/0AD5601_copod_10000.parquet
output_h5: /mnt/deepstore/Vidur/Junk_Classification/test_data/BLUE_output/outlier_hdf5/0AD5601_copod_10000.full.hdf5
slide_id: null
images_path: null
mask_model_path: /mnt/deepstore/shared_models/purple/cellpose_model
encode_model_path: /mnt/deepstore/shared_models/purple/representation_learning_04_28.pth
device: cuda:0
workers: 8
feature_workers: 8
```

## Example RED-BLUE usage

```yaml
input_parquet: /mnt/deepstore/Vidur/Junk_Classification/test_data/RED_BLUE_output/0AD5601/rarest_coords_filtered.parquet
output_h5: /mnt/deepstore/Vidur/Junk_Classification/test_data/RED_BLUE_output/0AD5601/rarest_coords_filtered.full.hdf5
slide_id: 0AD5601
images_path: null
mask_model_path: /mnt/deepstore/shared_models/purple/cellpose_model
encode_model_path: /mnt/deepstore/shared_models/purple/representation_learning_04_28.pth
device: cuda:0
workers: 8
feature_workers: 8
```

## Suggested validation checks

After the script finishes, inspect the output with:

```bash
h5ls /path/to/output.full.hdf5
python -c "import pandas as pd; print(pd.read_hdf('/path/to/output.full.hdf5', key='features').head())"
```

You should see datasets similar to:

- `channels`
- `images`
- `masks`
- `features`

