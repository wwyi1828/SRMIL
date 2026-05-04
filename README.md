# (Updating)

[![Paper](https://img.shields.io/badge/Paper-WACV%202026-b31b1b?logo=adobeacrobatreader&logoColor=white)](https://openaccess.thecvf.com/content/WACV2026/papers/Wu_Exploiting_Label-Independent_Regularization_from_Spatial_Patterns_for_Whole_Slide_Image_WACV_2026_paper.pdf)

## Resources

- [CLAM-preprocessed Camelyon16 folder link](https://www.dropbox.com/scl/fo/6zxmzennp2pqdzcpf8qiy/AJOHZgdSBH1v-9qbeQC0EYQ?rlkey=2vhdar7nol7zeiryyca1p8o3i&st=lvtsdux7&dl=0)
  - The CluBYOL and PLIP `.pkl` feature files in this folder can be used as SRMIL input files.

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
