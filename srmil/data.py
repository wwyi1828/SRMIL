import os
import pickle
import random

import dgl
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


def convert2dgl(input_loader, w_size=1):
    output_loader = []
    for item in tqdm(input_loader):
        if isinstance(item, dgl.DGLGraph):
            output_loader.append(item)
        else:
            g = dgl.radius_graph(item.pos.float() / 224, w_size * 1.42)
            g = dgl.add_self_loop(g)

            g.ndata["feat"] = item.x
            g.ndata["pos"] = item.pos
            if getattr(item, "y", None) is not None:
                num_classes = len(torch.unique(item.y))
                g.ndata["label"] = F.one_hot(item.y.to(torch.int64), num_classes=num_classes)

            g.gdata = {"label": item.graph_y, "WSI_ID": item.slide_index}
            output_loader.append(g)
    return output_loader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pickle.load(open(path, "rb"))


def _require_path(path_value, path_name, dataset_name):
    if path_value is None:
        raise ValueError(f"`{path_name}` is required for dataset `{dataset_name}`.")
    return path_value


def load_dataset(dataset_name, train_path=None, val_path=None, test_path=None):
    if dataset_name == "C16":
        num_classes = 2
        train_objs = _require_path(train_path, "--train_path", dataset_name)
        test_objs = _require_path(test_path, "--test_path", dataset_name)
        train_loader = _load_pickle(train_objs)
        test_loader = _load_pickle(test_objs)
        split = 0.15
        random.seed(42)
        random.shuffle(train_loader)
        train_samples = int(len(train_loader) * (1 - split))
        train_loader, val_loader = train_loader[:train_samples], train_loader[train_samples:]
        class_weights = torch.tensor([1, 1])
    elif dataset_name == "BRACS":
        num_classes = 3
        train_objs = _require_path(train_path, "--train_path", dataset_name)
        val_objs = _require_path(val_path, "--val_path", dataset_name)
        test_objs = _require_path(test_path, "--test_path", dataset_name)
        train_loader = _load_pickle(train_objs)
        val_loader = _load_pickle(val_objs)
        test_loader = _load_pickle(test_objs)
    else:
        if val_path is not None or test_path is not None:
            raise ValueError(
                "For non-C16/BRACS datasets, provide only `--train_path`; "
                "`--val_path` and `--test_path` are not used in this split logic."
            )
        num_classes = 2
        train_objs = _require_path(train_path, "--train_path", dataset_name)
        train_loader = _load_pickle(train_objs)
        random.seed(42)
        random.shuffle(train_loader)
        split_idx = int(2 / 3 * len(train_loader))
        test_loader = train_loader[split_idx:]
        train_loader = train_loader[:split_idx]
        split = 0.15
        train_samples = int(len(train_loader) * (1 - split))
        train_loader, val_loader = train_loader[:train_samples], train_loader[train_samples:]

    return num_classes, train_loader, val_loader, test_loader
