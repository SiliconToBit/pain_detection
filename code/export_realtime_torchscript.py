import argparse

import torch
import torch.nn as nn

from model import HRSingleModalModel


class RealtimeCausalWrapper(nn.Module):
    """TorchScript-friendly wrapper: fixed signature for streaming export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, h_0):
        logits, h_n, feat = self.model.forward_with_state(x, h_0)
        return logits, h_n, feat


def parse_args():
    parser = argparse.ArgumentParser(description="Export causal HR model to TorchScript")
    parser.add_argument("--ckpt", default="checkpoints_hr/best_hr_model.pth")
    parser.add_argument("--output", default="checkpoints_hr/hr_realtime_causal.pt")
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = ckpt
    else:
        state = ckpt
        meta = {}

    loss_type = str(meta.get("loss_type", "ce")).lower()
    model_arch = str(meta.get("model_arch", "causal_gru")).lower()
    hidden_dim = int(meta.get("hidden_dim", 64))
    in_channels = int(meta.get("in_channels", 1))
    num_classes = int(meta.get("num_classes", args.num_classes))

    if model_arch != "causal_gru":
        raise ValueError("TorchScript streaming export expects model_arch=causal_gru")

    model = HRSingleModalModel(
        seq_len=args.seq_len,
        num_classes=num_classes,
        in_channels=in_channels,
        output_mode=loss_type,
        model_arch=model_arch,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    wrapper = RealtimeCausalWrapper(model).to(device)
    wrapper.eval()

    example_x = torch.zeros(1, in_channels, args.seq_len, device=device)
    example_h = torch.zeros(1, 1, hidden_dim, device=device)

    traced = torch.jit.trace(wrapper, (example_x, example_h), strict=True)
    traced.save(args.output)

    print("Exported TorchScript model to:", args.output)


if __name__ == "__main__":
    main()
