import argparse

import numpy as np
import pandas as pd
import torch

from model import HRSingleModalModel


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pain level from HR sequence")
    parser.add_argument("--ckpt", default="checkpoints_hr/best_hr_model.pth", help="Path to checkpoint")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--hr-seq", default=None, help="Comma-separated HR values")
    parser.add_argument("--csv", default=None, help="CSV path containing hr_sequence column")
    parser.add_argument("--row", type=int, default=0, help="Row index when reading from --csv")
    parser.add_argument("--loss-type", default="auto", choices=["auto", "ce", "coral"])
    parser.add_argument("--model-arch", default="auto", choices=["auto", "causal_gru", "legacy_bilstm"])
    parser.add_argument("--hidden-dim", type=int, default=-1)
    parser.add_argument("--feature-mode", default="auto", choices=["auto", "basic", "enhanced"])
    return parser.parse_args()


def parse_sequence(seq_text, seq_len):
    values = [float(x.strip()) for x in seq_text.split(",") if x.strip()]
    if not values:
        raise ValueError("Empty sequence")

    if len(values) < seq_len:
        values += [values[-1]] * (seq_len - len(values))
    else:
        values = values[:seq_len]

    return np.array(values, dtype=np.float32)


def to_minmax(seq):
    arr = (seq - 60.0) / (120.0 - 60.0)
    return np.clip(arr, 0.0, 1.0)


def tanh_unit(values, scale):
    return (np.tanh(values / scale) + 1.0) / 2.0


def build_features(raw_seq, feature_mode):
    base = to_minmax(raw_seq)
    if feature_mode == "enhanced":
        diff = np.diff(raw_seq, prepend=raw_seq[0])
        kernel = np.ones(5, dtype=np.float32) / 5.0
        moving_avg = np.convolve(raw_seq, kernel, mode="same")
        dev = raw_seq - moving_avg
        features = np.stack([base, tanh_unit(diff, 5.0), tanh_unit(dev, 5.0)], axis=0)
        return features.astype(np.float32)
    return np.expand_dims(base, axis=0).astype(np.float32)


def load_input_sequence(args):
    if args.hr_seq:
        return args.hr_seq

    if args.csv:
        df = pd.read_csv(args.csv)
        if "hr_sequence" not in df.columns:
            raise ValueError("CSV must contain hr_sequence column")
        if args.row < 0 or args.row >= len(df):
            raise IndexError("row out of range")
        return str(df.iloc[args.row]["hr_sequence"])

    raise ValueError("Provide either --hr-seq or --csv")


def main():
    args = parse_args()
    seq_text = load_input_sequence(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = ckpt
    else:
        state = ckpt
        meta = {}

    loss_type = args.loss_type if args.loss_type != "auto" else str(meta.get("loss_type", "ce")).lower()
    model_arch = args.model_arch if args.model_arch != "auto" else str(meta.get("model_arch", "legacy_bilstm")).lower()
    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else int(meta.get("hidden_dim", 64))
    feature_mode = args.feature_mode if args.feature_mode != "auto" else str(meta.get("feature_mode", "basic")).lower()
    in_channels = int(meta.get("in_channels", 1 if feature_mode == "basic" else 3))

    model = HRSingleModalModel(
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        in_channels=in_channels,
        output_mode=loss_type,
        model_arch=model_arch,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    raw_seq = parse_sequence(seq_text, args.seq_len)
    features = build_features(raw_seq, feature_mode)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        if loss_type == "coral":
            level_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            pred = int((level_probs > 0.5).sum())
            # 将 level 概率近似映射为类别概率，便于展示。
            probs = np.zeros(args.num_classes, dtype=np.float32)
            probs[0] = 1.0 - level_probs[0]
            for i in range(1, args.num_classes - 1):
                probs[i] = max(0.0, level_probs[i - 1] - level_probs[i])
            probs[-1] = level_probs[-1]
            s = probs.sum()
            if s > 0:
                probs = probs / s
        else:
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred = int(np.argmax(probs))

    label_map = {
        0: "mild (NRS 1-3)",
        1: "moderate (NRS 4-6)",
        2: "severe (NRS 7-8)",
    }

    print("Device:", device)
    print("Loss type:", loss_type)
    print("Model arch:", model_arch)
    print("Feature mode:", feature_mode)
    print("Predicted class:", pred, "->", label_map.get(pred, "unknown"))
    print("Class probabilities [mild, moderate, severe]:")
    print([round(float(p), 6) for p in probs])


if __name__ == "__main__":
    main()
