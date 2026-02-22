import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .modeling import compute_graph_logits


def evaluate_split(model, data_loader, criterion, num_heads, enc_num_hidden, num_classes, use_supernode):
    test_loss = 0
    correct_preds = 0
    all_scores = []
    all_labels = []

    for item in data_loader:
        with torch.no_grad():
            g = item.to("cuda")
            if use_supernode:
                g = model.add_supernodes(g)

            logits = compute_graph_logits(model, g, num_heads, enc_num_hidden, use_supernode)
            if criterion is not None:
                test_loss += criterion(logits, g.gdata["label"].cuda())

            all_scores.extend(F.softmax(logits, dim=1).detach().cpu().numpy().tolist())
            all_labels.extend(g.gdata["label"].numpy().tolist())
            _, predicted = logits.max(1)
            correct_preds += (predicted == g.gdata["label"].argmax(axis=1).cuda()).sum().item()

    accuracy = correct_preds / len(data_loader)
    auc_list = [roc_auc_score(np.array(all_labels)[:, i], np.array(all_scores)[:, i]) for i in range(num_classes)]
    avg_loss = (test_loss / len(data_loader)) if criterion is not None else None
    return accuracy, auc_list, avg_loss
