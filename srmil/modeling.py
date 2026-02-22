import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .models.edcoder import setup_module
from .models.mil import BagNorm


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.414)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class CombinedModel(nn.Module):
    def __init__(self, k=1, in_dim=1024, num_hidden=256, init_token="0.01", uni_direct=False, **modules):
        super().__init__()
        for name, module in modules.items():
            setattr(self, name, module)

        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.init_token = init_token
        self.uni_direct = uni_direct

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden))
        self.super_tokens = nn.Parameter(torch.zeros(k, in_dim))

        if self.init_token == "0.01":
            nn.init.normal_(self.super_tokens, mean=0.0, std=0.01)
            nn.init.normal_(self.enc_mask_token, mean=0.0, std=0.01)
            nn.init.normal_(self.dec_mask_token, mean=0.0, std=0.01)
        if self.init_token == "0.02":
            nn.init.normal_(self.super_tokens, mean=0.0, std=0.02)
            nn.init.normal_(self.enc_mask_token, mean=0.0, std=0.02)
            nn.init.normal_(self.dec_mask_token, mean=0.0, std=0.02)
        if self.init_token == "Xavier":
            nn.init.xavier_normal_(self.super_tokens, gain=1.414)
            nn.init.xavier_normal_(self.enc_mask_token, gain=1.414)
            nn.init.xavier_normal_(self.dec_mask_token, gain=1.414)

        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)
        self.reformer.apply(initialize_weights)

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        return mask_nodes, keep_nodes

    def random_remask(self, g, rep, remask_rate=0.5):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[:num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, rep, masked_nodes, remask="None"):
        if remask == "None":
            rep = rep
        elif remask == "Zero":
            rep[masked_nodes] = 0
        elif remask == "Token":
            rep[masked_nodes] = self.dec_mask_token
        return rep

    def add_supernodes(self, g):
        g = g.clone()
        k = self.super_tokens.size(0)
        g.add_nodes(k)

        supernode_ids = torch.arange(g.number_of_nodes() - k, g.number_of_nodes(), device=g.device)
        nodes = torch.arange(g.number_of_nodes() - k, device=g.device)

        for supernode_id in supernode_ids:
            if self.uni_direct:
                g.add_edges(nodes, supernode_id)
            else:
                g.add_edges(supernode_id, nodes)
                g.add_edges(nodes, supernode_id)

        g.ndata["feat"][-k:] = self.super_tokens
        g.ndata["pos"][-k:] = torch.full((k, 2), torch.inf)
        if "label" in g.ndata:
            num_classes = g.ndata["label"].shape[-1]
            g.ndata["label"][-k:] = torch.full((k, num_classes), torch.inf)
        return g

    def drop_nodes(self, g, keep_indices):
        subg = g.subgraph(keep_indices)
        return subg

    def recover_nodes(self, g, subg_feats, keep_indices, mask_indices):
        pass


def build_model_and_optim(cfg, num_classes):
    num_hidden = cfg["num_hidden"]
    nhead = cfg["nhead"]
    nhead_out = cfg["nhead_out"]
    feat_drop = cfg["feat_drop"]
    attn_drop = cfg["attn_drop"]
    num_layers = cfg["num_layers"]
    num_dec_layers = cfg["num_dec_layers"]
    activation = cfg["activation"]
    negative_slope = cfg["negative_slope"]
    residual = cfg["residual"]
    norm = cfg["norm"]
    cls_hidden = cfg["cls_hidden"]
    cls_drop = cfg["cls_drop"]
    encoder_to_decoder_bias = cfg["encoder_to_decoder_bias"]
    reformer_bias = cfg["reformer_bias"]
    reformer_type = cfg["reformer_type"]
    norm_func_name = cfg["norm_func"]

    enc_num_hidden = num_hidden // nhead
    dec_in_dim = num_hidden
    dec_num_hidden = num_hidden // nhead if cfg["decoder_type"] in ("gat",) else num_hidden
    enc_nhead = nhead
    in_dim = num_features = 1024

    encoder = setup_module(
        m_type=cfg["encoder_type"],
        enc_dec="encoding",
        in_dim=in_dim,
        num_hidden=enc_num_hidden,
        out_dim=enc_num_hidden,
        num_layers=num_layers,
        nhead=enc_nhead,
        nhead_out=enc_nhead,
        concat_out=True,
        activation=activation,
        dropout=feat_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        norm=norm,
    )

    decoder = setup_module(
        m_type=cfg["decoder_type"],
        enc_dec="decoding",
        in_dim=dec_in_dim,
        num_hidden=dec_num_hidden,
        out_dim=in_dim,
        nhead_out=nhead_out,
        num_layers=num_dec_layers,
        nhead=nhead,
        activation=activation,
        dropout=feat_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        norm=norm,
        concat_out=True,
    )

    attender = setup_module(
        m_type="gat",
        enc_dec="decoding",
        in_dim=enc_num_hidden,
        num_hidden=cls_hidden,
        out_dim=1,
        nhead_out=1,
        num_layers=2,
        nhead=1,
        activation="relu",
        dropout=feat_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        norm=None,
        concat_out=True,
    )

    norm_func = {"Batch": nn.BatchNorm1d, "Bag": BagNorm}[norm_func_name]
    if reformer_type == "MLP":
        reformer = nn.Sequential(
            nn.Linear(in_dim, 256),
            norm_func(256),
            nn.ReLU(),
            nn.Linear(256, in_dim),
            norm_func(in_dim),
        )
    if reformer_type == "Linear":
        reformer = nn.Sequential(nn.Linear(in_dim, in_dim, bias=reformer_bias))
    if reformer_type == "Identity":
        reformer = nn.Sequential(nn.Identity())

    encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=encoder_to_decoder_bias)
    classifier = nn.Sequential(
        nn.Linear(enc_num_hidden, cls_hidden),
        nn.ReLU(),
        nn.Dropout(cls_drop),
        nn.Linear(cls_hidden, num_classes),
    )

    model = CombinedModel(
        encoder=encoder,
        decoder=decoder,
        classifier=classifier,
        encoder_to_decoder=encoder_to_decoder,
        k=cfg["num_superndoes"],
        attender=attender,
        reformer=reformer,
        in_dim=in_dim,
        num_hidden=num_hidden,
        init_token=cfg["init_token"],
        uni_direct=cfg["uni_direct"],
    )

    encoder = encoder.cuda()
    decoder = decoder.cuda()
    classifier = classifier.cuda()
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, cfg["epochs"])

    return model, criterion, optimizer, scheduler, enc_num_hidden, nhead, num_classes


def compute_graph_logits(model, g, num_heads, enc_num_hidden, use_supernode):
    ins_embed = model.encoder(g, model.reformer(g.ndata["feat"].cuda()))
    ins_embed = ins_embed.view(-1, num_heads, enc_num_hidden)
    ins_embed = torch.mean(ins_embed, dim=1)

    if use_supernode:
        supernode_representation = ins_embed[-1, None]
    else:
        ins_weights = F.softmax(model.attender(g, ins_embed), dim=0)
        supernode_representation = (ins_weights * ins_embed).sum(dim=0, keepdim=True)
    return model.classifier(supernode_representation)


def compute_masked_logits(model, g, mask_nodes, num_heads, enc_num_hidden, use_supernode, predict_mask):
    out_x = g.ndata["feat"].clone().cuda()
    out_x[mask_nodes] = model.enc_mask_token
    if predict_mask == "predict_mask":
        ins_embed = model.encoder(g, model.reformer(out_x))
    else:
        ins_embed = model.encoder(g, model.reformer(out_x))

    ins_embed = ins_embed.view(-1, num_heads, enc_num_hidden)
    ins_embed = torch.mean(ins_embed, dim=1)

    if use_supernode:
        supernode_representation = ins_embed[-1, None]
    else:
        ins_weights = F.softmax(model.attender(g, ins_embed), dim=0)
        supernode_representation = (ins_weights * ins_embed).sum(dim=0, keepdim=True)
    return model.classifier(supernode_representation)
