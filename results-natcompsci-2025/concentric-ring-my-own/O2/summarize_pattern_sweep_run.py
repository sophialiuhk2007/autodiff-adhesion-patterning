#!/usr/bin/env python
import csv
import json
import sys
from pathlib import Path


def read_json(path):
    with open(path) as f:
        return json.load(f)


def summarize_run(run_dir):
    progress_path = run_dir / "training_progress.json"
    j_path = run_dir / "j_parameters.json"
    if not progress_path.exists():
        return None

    progress = read_json(progress_path)
    if not progress:
        return None

    losses = [row["train_loss"] for row in progress]
    finite_losses = [loss for loss in losses if loss == loss]
    final = progress[-1]
    best_loss = min(finite_losses) if finite_losses else None
    best_epoch = next((row["epoch"] for row in progress if row["train_loss"] == best_loss), None)

    summary = {
        "run": run_dir.name,
        "n_epochs_logged": len(progress),
        "final_epoch": final["epoch"],
        "final_train_loss": final["train_loss"],
        "best_train_loss": best_loss,
        "best_train_loss_epoch": best_epoch,
        "final_learning_rate": final["learning_rate"],
        "final_j": final["j"],
        "final_type_fractions": final["type_fractions"],
        "final_type_counts": final["type_counts"],
        "stopped_reason": final.get("stopped_reason"),
    }

    if j_path.exists():
        j_params = read_json(j_path)
        summary["initial_j"] = j_params.get("initial_j")
        summary["trained_j"] = j_params.get("trained_j")
        summary["initial_type_fractions"] = j_params.get("initial_type_fractions")
        summary["trained_type_fractions"] = j_params.get("trained_type_fractions")

    return summary


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_pattern_sweep_run.py OUTDIR")

    outdir = Path(sys.argv[1])
    hyperparams_path = outdir / "train-pattern-opt-hyperparams.json"
    hyperparams = read_json(hyperparams_path) if hyperparams_path.exists() else {}

    run_summaries = []
    for run_dir in sorted(outdir.glob("train-pattern-*")):
        if run_dir.is_dir():
            summary = summarize_run(run_dir)
            if summary is not None:
                run_summaries.append(summary)

    sweep_summary = {
        "outdir": str(outdir),
        "hyperparams": hyperparams,
        "runs": run_summaries,
    }

    summary_json = outdir / "sweep_run_summary.json"
    with open(summary_json, "w") as f:
        json.dump(sweep_summary, f, indent=2)

    summary_csv = outdir / "sweep_run_summary.csv"
    fieldnames = [
        "run",
        "n_epochs_logged",
        "final_epoch",
        "final_train_loss",
        "best_train_loss",
        "best_train_loss_epoch",
        "final_learning_rate",
        "final_type_fractions",
        "final_type_counts",
        "final_j",
        "stopped_reason",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in run_summaries:
            row = {key: summary.get(key) for key in fieldnames}
            for key in ("final_type_fractions", "final_type_counts", "final_j"):
                row[key] = json.dumps(row[key])
            writer.writerow(row)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
