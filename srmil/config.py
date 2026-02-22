import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MIL Baselines training")
    parser.add_argument("--dataset", default="C16", type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--w_ce", default=None, type=float)
    parser.add_argument(
        "--train_path",
        default=None,
        type=str,
        help="Path to train pickle. Required for all datasets.",
    )
    parser.add_argument(
        "--val_path",
        default=None,
        type=str,
        help="Path to val pickle. Required only for BRACS.",
    )
    parser.add_argument(
        "--test_path",
        default=None,
        type=str,
        help="Path to test pickle. Required for C16/BRACS.",
    )
    return parser.parse_args()


def build_runtime_config(args):
    encoder_type = "gat"
    decoder_type = "gat"
    mask_type = "mask"
    mask_rate = 0.7
    remask_rate = 0.2
    num_remasking = 1
    in_drop = feat_drop = 0.11
    attn_drop = 0.15
    num_layers = 2
    num_dec_layers = 2
    num_hidden = enc_num_hidden = 256
    nhead = num_heads = 4
    nhead_out = num_out_heads = 1
    residual = True
    norm = "layernorm"
    activation = "prelu"
    negative_slope = 0.2
    alpha = 1.0
    epochs = 100
    num_superndoes = 1
    w_gen = 1.85
    window_size = 2
    cls_drop = 0.059
    cls_hidden = 512
    cls_hidden = 128
    lr = 0.0005
    weight_decay = 5e-4
    w_ce = 0.9036074632462564
    remask = "Token"
    add_node_first = False
    predict_mask = "predict_mask"
    fix_remask = True
    use_supernode = True
    uni_direct = False
    init_token = "0.01"
    reformer_type = "Identity"
    reformer_type = "Linear"

    weight_decay = 6.21006126812768e-05
    weight_decay = 1e-5
    w_ce = 1.199397927845541
    w_ce = args.w_ce if args.w_ce is not None else 1
    w_gen = 1.7543373911243918
    w_gen = 1.8
    encoder_to_decoder_bias = False
    reformer_bias = True
    norm_func = "Bag"

    return {
        "encoder_type": encoder_type,
        "decoder_type": decoder_type,
        "mask_type": mask_type,
        "mask_rate": mask_rate,
        "remask_rate": remask_rate,
        "num_remasking": num_remasking,
        "in_drop": in_drop,
        "feat_drop": feat_drop,
        "attn_drop": attn_drop,
        "num_layers": num_layers,
        "num_dec_layers": num_dec_layers,
        "num_hidden": num_hidden,
        "nhead": nhead,
        "nhead_out": nhead_out,
        "residual": residual,
        "norm": norm,
        "activation": activation,
        "negative_slope": negative_slope,
        "alpha": alpha,
        "epochs": epochs,
        "num_superndoes": num_superndoes,
        "w_gen": w_gen,
        "window_size": window_size,
        "cls_drop": cls_drop,
        "cls_hidden": cls_hidden,
        "lr": lr,
        "weight_decay": weight_decay,
        "w_ce": w_ce,
        "remask": remask,
        "add_node_first": add_node_first,
        "predict_mask": predict_mask,
        "fix_remask": fix_remask,
        "use_supernode": use_supernode,
        "uni_direct": uni_direct,
        "init_token": init_token,
        "reformer_type": reformer_type,
        "encoder_to_decoder_bias": encoder_to_decoder_bias,
        "reformer_bias": reformer_bias,
        "norm_func": norm_func,
    }
