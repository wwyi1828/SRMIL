import os

from .config import build_runtime_config, parse_args
from .data import convert2dgl, load_dataset, set_seed
from .trainer import train_and_evaluate


os.environ["DGLBACKEND"] = "pytorch"


def main():
    args = parse_args()
    num_classes, train_loader, val_loader, test_loader = load_dataset(
        args.dataset,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )

    if args.seed is not None:
        set_seed(int(args.seed))

    cfg = build_runtime_config(args)
    train_loader = convert2dgl(train_loader, w_size=cfg["window_size"])
    val_loader = convert2dgl(val_loader, w_size=cfg["window_size"])
    test_loader = convert2dgl(test_loader, w_size=cfg["window_size"])

    train_and_evaluate(args, cfg, train_loader, val_loader, test_loader, num_classes)


if __name__ == "__main__":
    main()
