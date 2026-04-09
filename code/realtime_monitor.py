import argparse
import time
from collections import deque

import numpy as np
import pandas as pd
import torch

from model import HRSingleModalModel


class OnlineHRProcessor:
    """Streaming HR processor with EMA normalization and multi-factor SQI."""

    def __init__(
        self,
        fs=4,
        window_sec=10,
        alpha=0.05,
        hr_min=30.0,
        hr_max=180.0,
        min_var=4.0,
    ):
        self.fs = fs
        self.window_sec = window_sec
        self.window_samples = int(fs * window_sec)
        self.alpha = alpha
        self.hr_min = float(hr_min)
        self.hr_max = float(hr_max)
        self.min_var = float(min_var)

        self.norm_buffer = deque(maxlen=self.window_samples)
        self.raw_buffer = deque(maxlen=self.window_samples)
        self.imputed_flags = deque(maxlen=self.window_samples)

        self.ema_mean = None
        self.ema_var = None
        self.last_valid_raw = None

    def _sanitize(self, raw_hr):
        if raw_hr is None:
            if self.last_valid_raw is None:
                return None, "MISSING"
            return self.last_valid_raw, "IMPUTED_MISSING"

        value = float(raw_hr)
        if np.isnan(value):
            if self.last_valid_raw is None:
                return None, "NAN"
            return self.last_valid_raw, "IMPUTED_NAN"

        if value < self.hr_min or value > self.hr_max:
            if self.last_valid_raw is None:
                return None, "OUTLIER"
            return self.last_valid_raw, "IMPUTED_OUTLIER"

        self.last_valid_raw = value
        return value, "OK"

    def _update_ema_stats(self, value):
        if self.ema_mean is None:
            self.ema_mean = value
            self.ema_var = max(self.min_var, 1.0)
            return

        prev_mean = self.ema_mean
        self.ema_mean = (1.0 - self.alpha) * self.ema_mean + self.alpha * value

        # Numerically stable online variance update.
        delta = value - prev_mean
        delta2 = value - self.ema_mean
        self.ema_var = (1.0 - self.alpha) * self.ema_var + self.alpha * (delta * delta2)
        self.ema_var = max(self.ema_var, self.min_var)

    def _compute_sqi(self):
        raw = np.asarray(self.raw_buffer, dtype=np.float32)
        imputed = np.asarray(self.imputed_flags, dtype=np.float32)

        if raw.size < 2:
            return "WARMUP", {
                "missing_ratio": float(imputed.mean()) if imputed.size else 0.0,
                "jump_rate": 0.0,
                "diff_std": 0.0,
            }

        diff = np.diff(raw)
        missing_ratio = float(imputed.mean()) if imputed.size else 0.0
        jump_rate = float(np.mean(np.abs(diff) > 12.0))
        diff_std = float(np.std(diff))

        # Rule-based quality gate for clinical robustness.
        if missing_ratio > 0.30:
            quality = "BAD_CONTACT"
        elif jump_rate > 0.25 or diff_std > 10.0:
            quality = "MOTION"
        elif diff_std < 0.15:
            quality = "FLATLINE"
        else:
            quality = "GOOD"

        return quality, {
            "missing_ratio": missing_ratio,
            "jump_rate": jump_rate,
            "diff_std": diff_std,
        }

    def update(self, raw_hr):
        value, ingest_status = self._sanitize(raw_hr)
        if value is None:
            return None, ingest_status, {}

        self._update_ema_stats(value)

        norm = (value - self.ema_mean) / (np.sqrt(self.ema_var) + 1e-5)
        norm = float(np.clip(norm, -6.0, 6.0))

        self.raw_buffer.append(value)
        self.norm_buffer.append(norm)
        self.imputed_flags.append(1.0 if ingest_status.startswith("IMPUTED") else 0.0)

        if len(self.norm_buffer) < self.window_samples:
            return None, "WARMUP", {
                "warmup_progress": round(len(self.norm_buffer) / self.window_samples, 4),
                "ingest_status": ingest_status,
            }

        quality, sqi = self._compute_sqi()
        sqi["ingest_status"] = ingest_status

        # Model input shape: [B, C, T]
        window = np.asarray(self.norm_buffer, dtype=np.float32)[None, None, :]
        return window, quality, sqi


class RealtimeHRMonitor:
    """Realtime pain monitor with stateful GRU inference and hysteresis decisioning."""

    def __init__(
        self,
        ckpt_path,
        device="cpu",
        fs=4,
        window_sec=10,
        num_classes=3,
        prob_ema_alpha=0.6,
        switch_entry=0.55,
        switch_exit=0.45,
        min_switch_steps=3,
    ):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            meta = ckpt
        else:
            state = ckpt
            meta = {}

        self.loss_type = str(meta.get("loss_type", "ce")).lower()
        self.model_arch = str(meta.get("model_arch", "legacy_bilstm")).lower()
        self.hidden_dim = int(meta.get("hidden_dim", 64))
        self.in_channels = int(meta.get("in_channels", 1))
        self.num_classes = int(meta.get("num_classes", num_classes))

        seq_len = int(round(fs * window_sec))
        self.model = HRSingleModalModel(
            seq_len=seq_len,
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            output_mode=self.loss_type,
            model_arch=self.model_arch,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.processor = OnlineHRProcessor(fs=fs, window_sec=window_sec)
        self.h_state = None

        self.prob_ema_alpha = float(prob_ema_alpha)
        self.pred_ema = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)

        self.switch_entry = float(switch_entry)
        self.switch_exit = float(switch_exit)
        self.min_switch_steps = int(min_switch_steps)
        self.current_level = None
        self.switch_counter = 0

    def _logits_to_probs(self, logits):
        if self.loss_type == "coral":
            level_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            probs = np.zeros(self.num_classes, dtype=np.float32)
            probs[0] = 1.0 - level_probs[0]
            for i in range(1, self.num_classes - 1):
                probs[i] = max(0.0, level_probs[i - 1] - level_probs[i])
            probs[-1] = level_probs[-1]
            s = float(probs.sum())
            if s > 0:
                probs /= s
            return probs

        return torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    def _hysteresis_decision(self):
        candidate = int(np.argmax(self.pred_ema))

        if self.current_level is None:
            self.current_level = candidate
            self.switch_counter = 0
            return self.current_level

        if candidate == self.current_level:
            self.switch_counter = 0
            return self.current_level

        candidate_conf = float(self.pred_ema[candidate])
        current_conf = float(self.pred_ema[self.current_level])

        if candidate_conf >= self.switch_entry and current_conf <= self.switch_exit:
            self.switch_counter += 1
            if self.switch_counter >= self.min_switch_steps:
                self.current_level = candidate
                self.switch_counter = 0
        else:
            self.switch_counter = 0

        return self.current_level

    @staticmethod
    def _trend(prev_probs, cur_probs, severe_idx=2):
        if severe_idx >= len(cur_probs):
            severe_idx = len(cur_probs) - 1
        delta = float(cur_probs[severe_idx] - prev_probs[severe_idx])
        if delta > 0.03:
            return "RISING"
        if delta < -0.03:
            return "FALLING"
        return "STABLE"

    def step(self, raw_hr, timestamp=None):
        window, quality, sqi = self.processor.update(raw_hr)
        if window is None:
            return {
                "status": quality,
                "pain_level": None,
                "confidence": 0.0,
                "trend": "UNKNOWN",
                "timestamp": float(time.time() if timestamp is None else timestamp),
                "sqi": sqi,
            }

        x = torch.from_numpy(window).to(self.device)

        with torch.no_grad():
            if self.model_arch == "causal_gru":
                logits, h_n, feat = self.model.forward_with_state(x, self.h_state)
                self.h_state = h_n.detach() if h_n is not None else None
            else:
                logits = self.model(x)
                feat = None
                self.h_state = None

        probs = self._logits_to_probs(logits)

        # Use previous EMA for trend, then update EMA (fixes trend timing bug).
        prev_ema = self.pred_ema.copy()
        self.pred_ema = self.prob_ema_alpha * probs + (1.0 - self.prob_ema_alpha) * self.pred_ema

        level = self._hysteresis_decision()
        trend = self._trend(prev_ema, self.pred_ema, severe_idx=min(2, self.num_classes - 1))

        feature_vec = []
        if feat is not None:
            feature_vec = feat.squeeze(0).cpu().numpy().astype(np.float32).tolist()

        return {
            "status": quality,
            "pain_level": int(level),
            "confidence": float(np.max(self.pred_ema)),
            "trend": trend,
            "timestamp": float(time.time() if timestamp is None else timestamp),
            "sqi": sqi,
            "feature_vec": feature_vec,
            "prediction": {
                "pain_level": int(level),
                "confidence": float(np.max(self.pred_ema)),
                "trend": trend,
            },
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run realtime HR pain monitor demo")
    parser.add_argument("--ckpt", default="checkpoints_hr/best_hr_model.pth")
    parser.add_argument("--csv", default="data/hr/val_hr.csv", help="CSV with hr_sequence column")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fs", type=int, default=4)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    monitor = RealtimeHRMonitor(
        ckpt_path=args.ckpt,
        device=args.device,
        fs=args.fs,
        window_sec=args.window_sec,
    )

    df = pd.read_csv(args.csv)
    seq_text = str(df.iloc[args.row]["hr_sequence"])
    hr_values = [float(x.strip()) for x in seq_text.split(",") if x.strip()]

    print("Realtime demo start")
    for i, hr in enumerate(hr_values[: args.max_steps]):
        result = monitor.step(hr)
        if result["pain_level"] is not None:
            print(
                "step={} hr={:.2f} status={} level={} conf={:.3f} trend={}".format(
                    i,
                    hr,
                    result["status"],
                    result["pain_level"],
                    result["confidence"],
                    result["trend"],
                )
            )


if __name__ == "__main__":
    main()
