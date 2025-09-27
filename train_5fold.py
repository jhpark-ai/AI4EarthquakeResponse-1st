# train_5fold.py
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import os
import math
import random
import argparse
import logging
from datetime import datetime
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models

from dataset import CropsSplit

# ===========================
# Utils
# ===========================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logger(log_dir: str, filename_prefix: str):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{filename_prefix}_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
    )
    logging.info(f"Log file: {log_path}")
    return log_path

# ===========================
# Losses (Imbalance)
# ===========================


class BCEWithLogitsLossPosWeight(nn.Module):
    def __init__(self, pos_weight: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("pw", torch.tensor(
            [pos_weight], dtype=torch.float32))
        self.ls = label_smoothing

    def forward(self, logits, targets):
        # targets: (B,1) in {0,1}
        if self.ls > 0:
            targets = targets * (1 - self.ls) + 0.5 * self.ls
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pw)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B,1), targets: (B,1)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        loss = self.alpha * (1-pt).pow(self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

# ===========================
# MixUp / CutMix (Batch-level)
# ===========================


@dataclass
class MixCfg:
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    prob: float = 0.5
    switch_prob: float = 0.5  # choose between mixup/cutmix


def rand_bbox(H, W, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mix_batch(x, y, cfg: MixCfg):
    if random.random() > cfg.prob:
        return x, y, None
    B, C, H, W = x.shape
    perm = torch.randperm(B, device=x.device)
    if random.random() < cfg.switch_prob:
        # MixUp
        lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
        x_m = lam * x + (1-lam) * x[perm]
        y_m = lam * y + (1-lam) * y[perm]
        return x_m, y_m, ('mixup', lam)
    else:
        # CutMix
        lam = np.random.beta(cfg.cutmix_alpha, cfg.cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(H, W, lam)
        x_m = x.clone()
        x_m[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam_adj = 1 - ((x2-x1)*(y2-y1) / (H*W))
        y_m = lam_adj * y + (1-lam_adj) * y[perm]
        return x_m, y_m, ('cutmix', lam_adj)


# ===========================
# Model
# ===========================


def build_model(in_ch=3, pretrained=True, model_name='convnextv2_large', mixstyle=False):
    # Use user-defined CustomModel if available
    try:
        from model import CustomModel
        m = CustomModel(
            in_channels=in_ch, num_classes=1, pretrained=pretrained, model_name=model_name, mixstyle=mixstyle)
        return m
    except Exception:
        logging.warning(
            "CustomModel not found, using torchvision ConvNeXt-Tiny")
        # Fallback: torchvision ConvNeXt-Tiny


# ===========================
# Train/Eval
# ===========================


@torch.no_grad()
def evaluate(model, loader, device, thresh: float = 0.5):
    """
    Returns:
        avg_loss (float), acc (float), f1 (float), precision (float), recall (float)
    """
    model.eval()
    total, correct = 0, 0
    losses = []

    tp = fp = fn = 0  # for F1
    bce = nn.BCEWithLogitsLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = bce(logits, y)
        losses.append(loss.item() * x.size(0))

        # preds/labels
        probs = torch.sigmoid(logits)
        preds = (probs >= thresh).long()      # (B,1)
        t = y.long()                          # (B,1)

        # accuracy
        correct += (preds == t).sum().item()
        total += x.size(0)

        # confusion terms (binary)
        tp += ((preds == 1) & (t == 1)).sum().item()
        fp += ((preds == 1) & (t == 0)).sum().item()
        fn += ((preds == 0) & (t == 1)).sum().item()

    avg_loss = sum(losses) / max(1, total)
    acc = correct / max(1, total)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          ) if (precision + recall) > 0 else 0.0

    return avg_loss, acc, f1, precision, recall


def stratified_split_indices(ds, val_ratio=0.15, seed=42):
    """Return stratified train/val indices based on labels in CropsSplit.ds.rows."""
    rnd = random.Random(seed)
    pos_idx = [i for i, r in enumerate(ds.rows) if r.label == 1]
    neg_idx = [i for i, r in enumerate(ds.rows) if r.label == 0]
    rnd.shuffle(pos_idx)
    rnd.shuffle(neg_idx)

    n_pos_val = max(1, int(len(pos_idx) * val_ratio)
                    ) if len(pos_idx) > 0 else 0
    n_neg_val = max(1, int(len(neg_idx) * val_ratio)
                    ) if len(neg_idx) > 0 else 0

    val_idx = pos_idx[:n_pos_val] + neg_idx[:n_neg_val]
    train_idx = pos_idx[n_pos_val:] + neg_idx[n_neg_val:]
    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    return train_idx, val_idx


def class_counts_on_indices(ds, indices):
    pos = sum(1 for i in indices if ds.rows[i].label == 1)
    neg = sum(1 for i in indices if ds.rows[i].label == 0)
    return pos, neg


def get_labels_from_dataset(ds):
    # Assume CropsSplit.rows[i].label exists and is 0/1
    return np.array([r.label for r in ds.rows], dtype=np.int64)


def iter_stratified_kfold_indices(labels, n_splits=5, seed=42, only_fold=None):
    """Yield (fold_idx, train_idx, val_idx) for stratified K-fold."""
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        all_pairs = list(skf.split(np.zeros(len(labels)), labels))
        for fi, (tr, va) in enumerate(all_pairs):
            if (only_fold is None) or (fi == only_fold):
                yield fi, tr.tolist(), va.tolist()
    except ImportError:
        # Simple fallback: split evenly by class
        logging.warning("Sklearn not found, using simple stratified split")
        raise ImportError("Sklearn not found")


def run_one_fold(args, fold_idx, train_idx, val_idx, device, train_dir):
    # Use separate dataset instances for train/val to allow different aug/normalization
    ds_train_full = CropsSplit(train_dir, img_size=args.img_size,
                               label_source=args.label_source, ignore_unknown=True,
                               use_masks=True, train=True)
    ds_val_full = CropsSplit(train_dir, img_size=args.img_size,
                             label_source=args.label_source, ignore_unknown=True,
                             use_masks=True, train=False, valid=True)

    train_ds = Subset(ds_train_full, train_idx)
    val_ds = Subset(ds_val_full,   val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    logging.info(f"[Fold {fold_idx}] train_loader: {len(train_loader)}")
    logging.info(f"[Fold {fold_idx}] val_loader: {len(val_loader)}")
    # Re-compute pos_weight per fold

    def class_counts_on_indices_local(ds, indices):
        pos = sum(1 for i in indices if ds.rows[i].label == 1)
        neg = sum(1 for i in indices if ds.rows[i].label == 0)
        return pos, neg
    pos, neg = class_counts_on_indices_local(ds_train_full, train_idx)
    pos_weight = float(neg / max(1, pos)) if pos > 0 else 1.0
    logging.info(
        f"[Fold {fold_idx}] class counts (train): pos={pos}, neg={neg}, pos_weight={pos_weight:.3f}")

    # Initialize model/loss/optimizer/scheduler
    model = build_model(in_ch=3, pretrained=args.pretrained,
                        model_name=args.model_name, mixstyle=args.mixstyle).to(device)

    if args.loss == "bce":
        criterion = BCEWithLogitsLossPosWeight(
            pos_weight=pos_weight, label_smoothing=args.label_smoothing).to(device)
    else:
        alpha = 0.75 if pos < neg else 0.25
        criterion = FocalLoss(alpha=alpha, gamma=2.0).to(device)

    base_opt_ctor = torch.optim.AdamW
    optimizer = base_opt_ctor(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.use_amp

    warmup_epochs = max(1, args.epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / \
            max(1, (args.epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lr_lambda)

    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_start = max(1, min(args.swa_start, args.epochs-1))
        swa_scheduler = SWALR(  # cos annealing
            optimizer, swa_lr=args.lr*0.1)
    else:
        swa_model = None

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mix_cfg = None
    if args.mixup or args.cutmix:
        mix_cfg = MixCfg(mixup_alpha=0.4 if args.mixup else 0.0,
                         cutmix_alpha=1.0 if args.cutmix else 0.0,
                         prob=0.7, switch_prob=0.5)

    # Per-fold checkpoint paths
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_best = os.path.join(
        args.out_dir, f"best_{args.use_swa}_{args.mixup}_{args.cutmix}_{args.mixstyle}_{args.loss}_{args.seed}_model{args.model_name}_aug_{args.data_root}_lr{args.lr}_data_aug_type{args.data_aug_type}_swa_start{args.swa_start}_fold{fold_idx}.pt")
    ckpt_best_swa = os.path.join(
        args.out_dir, f"best_swa_{args.use_swa}_{args.mixup}_{args.cutmix}_{args.mixstyle}_{args.loss}_{args.seed}_model{args.model_name}_aug_{args.data_root}_lr{args.lr}_data_aug_type{args.data_aug_type}_swa_start{args.swa_start}_fold{fold_idx}.pt")

    # Training
    best_val = (-1.0, -1.0)  # Save best by (f1, acc) if desired
    for epoch in range(args.epochs):
        set_seed(args.seed + fold_idx)  # Keep reproducibility
        model.train()
        run_loss, seen = 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mix_cfg is not None:
                x, y, _ = mix_batch(x, y, mix_cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_loss = loss.detach().item()

            run_loss += batch_loss * x.size(0)
            seen += x.size(0)

        val_loss, val_acc, val_f1, val_p, val_r = evaluate(
            model, val_loader, device)
        # Scheduler/SWA
        if args.use_swa and epoch >= args.swa_start:
            swa_scheduler.step()
            if (args.epochs)*0.85 < epoch:
                swa_model.update_parameters(model)
        else:
            scheduler.step()
        lr = (swa_scheduler if (args.use_swa and epoch >=
              args.swa_start) else scheduler).get_last_lr()[0]

        logging.info(f"[Fold {fold_idx}] Epoch {epoch+1:03d}/{args.epochs:03d} | "
                     f"TrainLoss {run_loss/seen:.4f} | ValLoss {val_loss:.4f} | "
                     f"ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f} (P {val_p:.4f}/R {val_r:.4f}) | "
                     f"LR {lr:.2e}")

        # Save best (by F1 here)
        if val_f1 >= best_val[0] and epoch > int(args.epochs*0.5):
            best_val = (val_f1, val_acc)
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_f1": val_f1}, ckpt_best)

    # Optionally evaluate SWA model
    if args.use_swa and swa_model is not None:
        update_bn(train_loader, swa_model, device=device)
        # torch.save({"model": swa_model.state_dict()}, ckpt_best_swa)
        # logging.info(f"[Fold {fold_idx}] Saved SWA model to {ckpt_best_swa}")

        val_loss, val_acc, val_f1, val_p, val_r = evaluate(
            swa_model, val_loader, device)
        logging.info(
            f"SWA ValLoss {val_loss:.4f} | SWA ValAcc {val_acc:.4f} | SWA ValF1 {val_f1:.4f} (SWA P {val_p:.4f}/SWA R {val_r:.4f})")

    return best_val  # (best_f1, best_acc)


'''
python3 train_5fold.py --kfolds 5 --fold all --epochs 50 --pretrained --mixup --cutmix --use_amp --use_swa --model_name eva --img_size 448 --batch_size 32 --lr 5e-5 --data_aug_type test --device 0
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=f"CROPS_50_all")
    ap.add_argument("--img_size", type=int, default=448)  # 384, 224, 448
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--label_source", type=str,
                    default="filename", choices=["filename", "csv"])
    ap.add_argument("--ignore_unknown", action="store_true")
    ap.add_argument("--use_swa", action="store_true")
    ap.add_argument("--swa_start", type=int, default=30)
    ap.add_argument("--loss", type=str, default="bce",
                    choices=["bce", "focal"])
    ap.add_argument("--data_aug_type", type=str, default="test",
                    choices=["test"])
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--cutmix", action="store_true")
    ap.add_argument("--mixstyle", action="store_true")
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kfolds", type=int, default=5,
                    help="number of folds for CV")
    ap.add_argument("--fold", type=str, default="all", choices=["all", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    help="run all folds or a specific fold index (0-based)")
    ap.add_argument("--out_dir", type=str,
                    default="saved_model", help="ckpt/output dir")
    ap.add_argument("--log_dir", type=str, default="log", help="log directory")
    ap.add_argument("--model_name", type=str, default="eva", choices=["convnextv2_base", "resnext", "convnextv2_large", "eva", "dinov3_vitl", "dinov3_convnext_large"],
                    help="model name")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # initialize logging
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_prefix = (
        f"{script_name}_"
        f"swa{int(args.use_swa)}_"
        f"mix{int(args.mixup)}_cut{int(args.cutmix)}_"
        f"mixstyle{int(args.mixstyle)}_"
        f"loss{args.loss}_seed{args.seed}_model{args.model_name}_norm_aug_{args.data_root}_lr{args.lr}_data_aug_type{args.data_aug_type}_swa_start{args.swa_start}"
    )
    init_logger(args.log_dir, log_prefix)
    try:
        logging.info("Args: %s", json.dumps(
            vars(args), indent=2, ensure_ascii=False))
    except Exception:
        logging.info("Args(dict): %s", str(vars(args)))

    train_dir = os.path.join(args.data_root, "train")

    # Load all labels once
    if args.model_name == 'dinov3_vitl':
        imagenet_norm = True
        print("dinov3_vitl")
    else:
        imagenet_norm = True
    full_for_labels = CropsSplit(train_dir, img_size=args.img_size,
                                 label_source=args.label_source, ignore_unknown=True,
                                 use_masks=True, train=True, imagenet_norm=imagenet_norm)
    labels = get_labels_from_dataset(full_for_labels)

    only_fold = None if args.fold == "all" else int(args.fold)
    results = []  # (fold, best_f1, best_acc)

    for fold_idx, tr_idx, va_idx in iter_stratified_kfold_indices(labels, n_splits=args.kfolds,
                                                                  seed=args.seed, only_fold=only_fold):
        logging.info(
            f"\n========== Fold {fold_idx} / {args.kfolds} ==========")
        best_f1, best_acc = run_one_fold(
            args, fold_idx, tr_idx, va_idx, device, train_dir)
        logging.info(
            f"Fold {fold_idx} best F1={best_f1:.4f}, best Acc={best_acc:.4f}")
        results.append((fold_idx, best_f1, best_acc))

    # Aggregation
    if results:
        results = sorted(results, key=lambda x: x[0])
        f1s = [x[1] for x in results]
        accs = [x[2] for x in results]
        logging.info("\n===== Cross-Validation Summary =====")
        for fi, (fold_idx, f1, acc) in enumerate(results):
            logging.info(
                f"Fold {fold_idx}: best F1={f1:.4f}, best Acc={acc:.4f}")
        logging.info(f"Mean F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
                     f"Mean Acc={np.mean(accs):.4f} ± {np.std(accs):.4f}")


if __name__ == "__main__":
    main()
