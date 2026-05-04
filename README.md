# SRMIL

[![Paper](https://img.shields.io/badge/Paper-WACV%202026-b31b1b?logo=adobeacrobatreader&logoColor=white)](https://openaccess.thecvf.com/content/WACV2026/papers/Wu_Exploiting_Label-Independent_Regularization_from_Spatial_Patterns_for_Whole_Slide_Image_WACV_2026_paper.pdf)

## Resources

- [CLAM-preprocessed Camelyon16 folder link](https://www.dropbox.com/scl/fo/6zxmzennp2pqdzcpf8qiy/AJOHZgdSBH1v-9qbeQC0EYQ?rlkey=2vhdar7nol7zeiryyca1p8o3i&st=lvtsdux7&dl=0)
  - The CluBYOL and PLIP `.pkl` feature files in this folder can be used as SRMIL input files.
  - These files were generated with a modified [CLAM preprocessing branch](https://github.com/wwyi1828/CLAM) that aligns patch coordinates to a global 224-step grid.

### Difference from standard CLAM preprocessing

Original CLAM starts the coordinate grid from each contour's own bounding box:

```python
start_x, start_y, w, h = cv2.boundingRect(cont)

step_size_x = step_size * patch_downsample[0]
step_size_y = step_size * patch_downsample[1]

x_range = np.arange(start_x, stop_x, step=step_size_x)
y_range = np.arange(start_y, stop_y, step=step_size_y)
```

The modified branch expands each contour box to the nearest global grid origin before candidate patch coordinates are generated:

```python
step_size_x = read_step_size * patch_downsample[0]
step_size_y = read_step_size * patch_downsample[1]

start_x, start_y, w, h = cv2.boundingRect(cont)

w += start_x % step_size_x
h += start_y % step_size_y
start_x -= start_x % step_size_x
start_y -= start_y % step_size_y
```

With the default `patch_size=224` and `step_size=224`, this makes patch coordinates fall on the same global 224-grid instead of using a different local grid for each contour.

## Quick Run

```bash
python -m srmil.cli \
  --dataset C16 \
  --train_path /path/to/C16_R50.pkl \
  --test_path /path/to/C16_R50_test.pkl \
  --w_ce 1.0 \
  --seed 1 \
  --save_path mil_results/run
```

Outputs:

- `<save_path>.json`

Path requirements:

- `C16`: `--train_path`, `--test_path`
- `BRACS`: `--train_path`, `--val_path`, `--test_path`
- other dataset values (TCGA split branch): `--train_path` only
