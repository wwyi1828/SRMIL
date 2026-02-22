import copy

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from tqdm import trange

from .evaluation import evaluate_split
from .io_utils import update_results
from .modeling import build_model_and_optim, compute_graph_logits, compute_masked_logits, sce_loss


def train_and_evaluate(args, cfg, train_loader, val_loader, test_loader, num_classes):
    model, criterion, optimizer, scheduler, enc_num_hidden, num_heads, num_classes = build_model_and_optim(
        cfg, num_classes
    )

    best_model_state = None
    best_val_accuracy = 0
    lowest_val_loss = float("inf")
    best_val_auc = 0

    for epoch in trange(cfg["epochs"]):
        model = model.train()
        train_loss_ce_0 = 0
        train_loss_ce_1 = 0
        train_loss_sce = 0
        correct_preds = 0
        all_scores = []
        all_labels = []

        for item in train_loader:
            g = item.to("cuda")
            recon_loss = 0
            ce_loss = 0

            mask_nodes, keep_nodes = model.encoding_mask_noise(g, g.ndata["feat"], mask_rate=cfg["mask_rate"])
            if cfg["use_supernode"]:
                if cfg["add_node_first"]:
                    g = model.add_supernodes(g)

            out_x = g.ndata["feat"].clone().cuda()
            out_x[mask_nodes] = model.enc_mask_token
            ins_embed = model.encoder(g, model.reformer(out_x))
            latent_embed = model.encoder_to_decoder(ins_embed)

            for _ in range(cfg["num_remasking"]):
                rep = latent_embed.clone()
                if not cfg["fix_remask"]:
                    rep, remask_nodes, rekeep_nodes = model.random_remask(g, rep, remask_rate=cfg["remask_rate"])
                else:
                    rep = model.fixed_remask(rep, mask_nodes, remask=cfg["remask"])
                recon_x = model.decoder(g, rep)
                x_init = g.ndata["feat"][mask_nodes]
                x_rec = recon_x[mask_nodes]
                recon_loss += sce_loss(x_rec, x_init, alpha=cfg["alpha"])
            recon_loss = recon_loss / cfg["num_remasking"]

            if cfg["use_supernode"]:
                if not cfg["add_node_first"]:
                    g = model.add_supernodes(g)

            logits = compute_masked_logits(
                model,
                g,
                mask_nodes,
                num_heads,
                enc_num_hidden,
                cfg["use_supernode"],
                cfg["predict_mask"],
            )
            corrupt_ce_loss = criterion(logits, g.gdata["label"].cuda())
            train_loss_ce_0 += corrupt_ce_loss.item()
            ce_loss += (2 - cfg["w_ce"]) * corrupt_ce_loss

            logits = compute_graph_logits(model, g, num_heads, enc_num_hidden, cfg["use_supernode"])
            complete_ce_loss = criterion(logits, g.gdata["label"].cuda())
            train_loss_ce_1 += complete_ce_loss.item()
            ce_loss += cfg["w_ce"] * complete_ce_loss

            all_scores.extend(F.softmax(logits, dim=1).detach().cpu().numpy().tolist())
            all_labels.extend(g.gdata["label"].numpy().tolist())

            loss = (2 - cfg["w_gen"]) * ce_loss + cfg["w_gen"] * recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sce += recon_loss.item()
            _, predicted = logits.max(1)
            correct_preds += (predicted == g.gdata["label"].argmax(axis=1).cuda()).sum().item()

        scheduler.step()
        auc_list = [roc_auc_score(np.array(all_labels)[:, i], np.array(all_scores)[:, i]) for i in range(num_classes)]
        print(f"Train AUC: {auc_list}")
        print(
            f"CE loss: {train_loss_ce_0/len(train_loader)}, "
            f"SCE loss: {train_loss_sce/len(train_loader)}, "
            f"train acc:{correct_preds/len(train_loader)}"
        )

        model = model.eval()
        val_accuracy, val_auc_list, val_loss = evaluate_split(
            model, val_loader, criterion, num_heads, enc_num_hidden, num_classes, cfg["use_supernode"]
        )
        print(f"val acc:{val_accuracy}, Val AUC: {val_auc_list}")
        val_avg_auc = sum(val_auc_list) / len(val_auc_list)

        if epoch > cfg["epochs"] * 0.6:
            if (val_accuracy > best_val_accuracy) or (val_avg_auc >= best_val_auc and val_accuracy == best_val_accuracy):
                lowest_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_val_auc = val_avg_auc
                best_model_state = copy.deepcopy(model.state_dict())
                print(
                    "Updated the best model in memory with "
                    f"Val Accuracy: {best_val_accuracy:.3f}, Val AUC: {best_val_auc:.2f}"
                )

        test_accuracy, test_auc_list, test_loss = evaluate_split(
            model, test_loader, criterion, num_heads, enc_num_hidden, num_classes, cfg["use_supernode"]
        )
        print(f"test acc:{test_accuracy}, Test AUC: {test_auc_list}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

        final_accuracy, final_auc_list, _ = evaluate_split(
            model, test_loader, None, num_heads, enc_num_hidden, num_classes, cfg["use_supernode"]
        )
        print(f"test acc:{final_accuracy}, Test AUC: {final_auc_list}")

        if args.save_path is not None:
            json_path = f"{args.save_path}.json"
            dataset_name = args.dataset
            baseline_name = cfg["w_ce"]
            update_results(
                json_path,
                dataset_name,
                f"mr{cfg['mask_rate']}_bidir_w_ce:{baseline_name}",
                final_accuracy,
                final_auc_list,
            )
