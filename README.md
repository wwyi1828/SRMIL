# (Updating)
# Quick Run 

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
