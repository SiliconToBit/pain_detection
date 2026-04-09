import json
import os
import subprocess
import time


def run_command(cmd, env):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_experiment(name, env_overrides):
    save_dir = os.path.join("checkpoints_compare", name)
    os.makedirs(save_dir, exist_ok=True)

    env = os.environ.copy()
    # Ensure per-experiment env is isolated from accidental shell leftovers.
    env.pop("CLASS_WEIGHTS", None)
    env.update(env_overrides)
    env["SAVE_DIR"] = save_dir

    start = time.time()

    run_command(["python3", "code/train.py"], env)

    ckpt = os.path.join(save_dir, "best_hr_model.pth")
    eval_json = os.path.join(save_dir, "eval_metrics.json")
    run_command(
        [
            "python3",
            "code/evaluate.py",
            "--ckpt",
            ckpt,
            "--output-json",
            eval_json,
        ],
        env,
    )

    elapsed = time.time() - start
    with open(eval_json, "r") as f:
        metrics = json.load(f)

    result = {
        "name": name,
        "save_dir": save_dir,
        "elapsed_sec": round(elapsed, 2),
        "kappa": metrics.get("kappa"),
        "accuracy": metrics.get("accuracy"),
        "f1_weighted": metrics.get("f1_weighted"),
        "loss": metrics.get("loss"),
        "loss_type": metrics.get("loss_type"),
        "feature_mode": metrics.get("feature_mode"),
        "normalize_mode": metrics.get("normalize_mode"),
    }
    return result


def main():
    # 快速对比配置：控制训练时长，优先探索稳健提升 Kappa 的方向。
    experiments = [
        {
            "name": "baseline_ce",
            "env": {
                "NUM_EPOCHS": "16",
                "MIN_EPOCHS": "8",
                "EARLY_STOP_PATIENCE": "4",
                "SCHEDULER_PATIENCE": "3",
                "LEARNING_RATE": "3e-4",
                "BATCH_SIZE": "16",
                "NUM_WORKERS": "4",
                "LOSS_TYPE": "ce",
                "MODEL_ARCH": "causal_gru",
                "FEATURE_MODE": "basic",
                "NORMALIZE_MODE": "minmax",
                "USE_WEIGHTED_SAMPLER": "0",
                "USE_FOCAL_LOSS": "0",
            },
        },
        {
            "name": "ce_enhanced_minmax",
            "env": {
                "NUM_EPOCHS": "16",
                "MIN_EPOCHS": "8",
                "EARLY_STOP_PATIENCE": "4",
                "SCHEDULER_PATIENCE": "3",
                "LEARNING_RATE": "2.5e-4",
                "BATCH_SIZE": "16",
                "NUM_WORKERS": "4",
                "LOSS_TYPE": "ce",
                "MODEL_ARCH": "causal_gru",
                "FEATURE_MODE": "enhanced",
                "NORMALIZE_MODE": "minmax",
                "USE_WEIGHTED_SAMPLER": "0",
                "USE_FOCAL_LOSS": "0",
            },
        },
        {
            "name": "ce_subject_basic",
            "env": {
                "NUM_EPOCHS": "16",
                "MIN_EPOCHS": "8",
                "EARLY_STOP_PATIENCE": "4",
                "SCHEDULER_PATIENCE": "3",
                "LEARNING_RATE": "2.5e-4",
                "BATCH_SIZE": "16",
                "NUM_WORKERS": "4",
                "LOSS_TYPE": "ce",
                "MODEL_ARCH": "causal_gru",
                "FEATURE_MODE": "basic",
                "NORMALIZE_MODE": "subject",
                "USE_WEIGHTED_SAMPLER": "0",
                "USE_FOCAL_LOSS": "0",
            },
        },
        {
            "name": "ce_manual_mid_weight",
            "env": {
                "NUM_EPOCHS": "16",
                "MIN_EPOCHS": "8",
                "EARLY_STOP_PATIENCE": "4",
                "SCHEDULER_PATIENCE": "3",
                "LEARNING_RATE": "2.5e-4",
                "BATCH_SIZE": "16",
                "NUM_WORKERS": "4",
                "LOSS_TYPE": "ce",
                "MODEL_ARCH": "causal_gru",
                "FEATURE_MODE": "basic",
                "NORMALIZE_MODE": "minmax",
                "USE_WEIGHTED_SAMPLER": "0",
                "USE_FOCAL_LOSS": "0",
                "CLASS_WEIGHTS": "0.9,1.4,1.2",
            },
        },
        {
            "name": "enhanced_coral",
            "env": {
                "NUM_EPOCHS": "16",
                "MIN_EPOCHS": "8",
                "EARLY_STOP_PATIENCE": "4",
                "SCHEDULER_PATIENCE": "3",
                "LEARNING_RATE": "2e-4",
                "BATCH_SIZE": "16",
                "NUM_WORKERS": "4",
                "LOSS_TYPE": "coral",
                "MODEL_ARCH": "causal_gru",
                "FEATURE_MODE": "enhanced",
                "NORMALIZE_MODE": "subject",
                "USE_WEIGHTED_SAMPLER": "0",
                "USE_FOCAL_LOSS": "0",
            },
        },
    ]

    all_results = []
    failed = []

    for exp in experiments:
        print("\n" + "=" * 80)
        print("Experiment:", exp["name"])
        print("=" * 80)
        try:
            result = run_experiment(exp["name"], exp["env"])
            all_results.append(result)
            print("Completed:", exp["name"])
            print(
                "kappa={:.4f} acc={:.4f} f1={:.4f} loss={:.4f}".format(
                    result["kappa"],
                    result["accuracy"],
                    result["f1_weighted"],
                    result["loss"],
                )
            )
        except subprocess.CalledProcessError as e:
            failed.append({"name": exp["name"], "returncode": e.returncode})
            print("Failed:", exp["name"], "returncode=", e.returncode)

    summary = {
        "timestamp": int(time.time()),
        "results": all_results,
        "failed": failed,
    }

    if len(all_results) >= 2:
        baseline = all_results[0]
        best = max(all_results, key=lambda x: x["kappa"])
        summary["best"] = best
        summary["delta_vs_baseline"] = {
            "kappa": round(best["kappa"] - baseline["kappa"], 6),
            "accuracy": round(best["accuracy"] - baseline["accuracy"], 6),
            "f1_weighted": round(best["f1_weighted"] - baseline["f1_weighted"], 6),
            "loss": round(best["loss"] - baseline["loss"], 6),
        }

    os.makedirs("checkpoints_compare", exist_ok=True)
    out_path = "checkpoints_compare/comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Comparison summary saved to:", out_path)
    if all_results:
        print("Results:")
        for r in sorted(all_results, key=lambda x: x["kappa"], reverse=True):
            print(
                "- {} | kappa={:.4f} acc={:.4f} f1={:.4f} loss={:.4f} | mode=({},{},{})".format(
                    r["name"],
                    r["kappa"],
                    r["accuracy"],
                    r["f1_weighted"],
                    r["loss"],
                    r.get("loss_type"),
                    r.get("feature_mode"),
                    r.get("normalize_mode"),
                )
            )
    if failed:
        print("Failed experiments:", failed)


if __name__ == "__main__":
    main()
