import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from datebase import PainHRDataset
from model import HRSingleModalModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HR pain model on validation set")
    parser.add_argument("--val-csv", default="data/hr/val_hr.csv", help="Path to validation CSV")
    parser.add_argument("--ckpt", default="checkpoints_hr/best_hr_model.pth", help="Path to checkpoint")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--output-json", default="checkpoints_hr/eval_metrics.json", help="Where to save metrics JSON")
    parser.add_argument("--loss-type", default="auto", choices=["auto", "ce", "coral"], help="Loss/output mode")
    parser.add_argument("--model-arch", default="auto", choices=["auto", "causal_gru", "legacy_bilstm"], help="Model architecture")
    parser.add_argument("--hidden-dim", type=int, default=-1, help="Hidden dim for GRU/LSTM; -1 means auto")
    parser.add_argument("--feature-mode", default="auto", choices=["auto", "basic", "enhanced"], help="Feature mode")
    parser.add_argument("--normalize-mode", default="auto", choices=["auto", "minmax", "subject"], help="Normalization mode")
    return parser.parse_args()


def load_checkpoint_raw(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))

    return torch.load(ckpt_path, map_location=device)


def labels_to_levels(labels, num_classes):
    levels = []
    for k in range(num_classes - 1):
        levels.append((labels > k).float())
    return torch.stack(levels, dim=1)


def logits_to_predictions(logits, loss_type):
    if loss_type == "coral":
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
    return torch.argmax(logits, dim=1)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_ckpt = load_checkpoint_raw(args.ckpt, device)
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
        meta = raw_ckpt
    else:
        state_dict = raw_ckpt
        meta = {}

    loss_type = args.loss_type if args.loss_type != "auto" else str(meta.get("loss_type", "ce")).lower()
    model_arch = args.model_arch if args.model_arch != "auto" else str(meta.get("model_arch", "legacy_bilstm")).lower()
    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else int(meta.get("hidden_dim", 64))
    feature_mode = args.feature_mode if args.feature_mode != "auto" else str(meta.get("feature_mode", "basic")).lower()
    normalize_mode = args.normalize_mode if args.normalize_mode != "auto" else str(meta.get("normalize_mode", "minmax")).lower()
    in_channels = int(meta.get("in_channels", 1 if feature_mode == "basic" else 3))

    os.environ["FEATURE_MODE"] = feature_mode
    os.environ["NORMALIZE_MODE"] = normalize_mode

    dataset = PainHRDataset(csv_path=args.val_csv, split="val", seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = HRSingleModalModel(
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        in_channels=in_channels,
        output_mode=loss_type,
        model_arch=model_arch,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict)

    if loss_type == "coral":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for hr_data, labels in loader:
            hr_data = hr_data.to(device)
            labels = labels.to(device)

            outputs = model(hr_data)
            if loss_type == "coral":
                levels = labels_to_levels(labels, args.num_classes)
                loss = criterion(outputs, levels)
            else:
                loss = criterion(outputs, labels)

            total_loss += loss.item() * hr_data.size(0)
            preds = logits_to_predictions(outputs, loss_type)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=["mild_1_to_3", "moderate_4_to_6", "severe_7_to_8"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "num_samples": len(loader.dataset),
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "kappa": float(kappa),
        "best_kappa_from_ckpt": float(meta.get("best_kappa")) if "best_kappa" in meta else None,
        "loss_type": loss_type,
        "model_arch": model_arch,
        "hidden_dim": hidden_dim,
        "feature_mode": feature_mode,
        "normalize_mode": normalize_mode,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Device:", device)
    print("Validation samples:", len(loader.dataset))
    print("Loss: {:.4f}".format(avg_loss))
    print("Accuracy: {:.4f}".format(acc))
    print("F1(weighted): {:.4f}".format(f1))
    print("Kappa: {:.4f}".format(kappa))
    print("Loss type:", loss_type)
    print("Model arch:", model_arch)
    print("Feature mode:", feature_mode)
    print("Normalize mode:", normalize_mode)
    print("Confusion matrix [rows=true, cols=pred]:")
    print(cm)
    print("Metrics JSON saved to:", args.output_json)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
