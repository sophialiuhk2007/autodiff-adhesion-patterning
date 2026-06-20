#!/usr/bin/env python3
import argparse
import csv
import html
import importlib.util
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as np
import jax_morph as jxm  # type: ignore
import numpy as onp

jax.config.update("jax_enable_x64", True)

SCRIPT_DIR = Path(__file__).resolve().parents[1]
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

import pattern_istate_and_model as pattern_model

jax.config.update("jax_disable_jit", True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_plotter():
    plot_path = SCRIPT_DIR / "translucent-3d-plotting.py"
    spec = importlib.util.spec_from_file_location("translucent_3d_plotting", plot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_translucent_spheres


def visible_cell_mask(state, hidden_types=None):
    hidden_types = set(hidden_types or [])
    if not hidden_types:
        return None
    type_idx = onp.asarray(onp.argmax(state.celltype, axis=1)) + 1
    return ~onp.isin(type_idx, list(hidden_types))


def final_state_metrics(state, pattern, loss_kwargs):
    type_idx = onp.asarray(onp.argmax(state.celltype, axis=1))
    metrics = {
        "loss": float(pattern_model.pattern_loss(state, pattern=pattern, **loss_kwargs)),
        "f12": float(pattern_model.contact_frequency(state, 0, 1)),
        "f33": float(pattern_model.contact_frequency(state, 2, 2)),
        "m1": float(pattern_model.media_contact_frequency(state, 0)),
        "m2": float(pattern_model.media_contact_frequency(state, 1)),
        "m3": float(pattern_model.media_contact_frequency(state, 2)),
        "shell_distance": float(pattern_model.shell_distance_loss(state)),
    }
    positions = onp.asarray(state.position)
    for i in range(3):
        mask = type_idx == i
        metrics[f"count_type_{i + 1}"] = int(onp.sum(mask))
        if onp.any(mask):
            type_positions = positions[mask]
            center = type_positions.mean(axis=0)
            dist = onp.sqrt(onp.sum((type_positions - center) ** 2, axis=-1) + 1e-8)
            metrics[f"mean_radius_type_{i + 1}"] = float(onp.mean(dist))
        else:
            metrics[f"mean_radius_type_{i + 1}"] = 0.0
    return metrics


def final_state_for_run(run_dir, hyperparams, seed):
    j_params = read_json(run_dir / "j_parameters.json")
    trained_j = np.asarray(j_params["trained_j"])
    trained_fractions = np.asarray(j_params["trained_type_fractions"])

    n_cells = int(hyperparams["n_cells"])
    type_ratios = np.asarray(hyperparams["type_ratios"])
    n_steps = int(hyperparams["n_steps"])

    base_key = jax.random.PRNGKey(seed)
    init_key, type_key, sim_key = jax.random.split(base_key, 3)
    base_state = pattern_model.build_istate(init_key, n_cells=n_cells, type_ratios=type_ratios)
    model = pattern_model.build_model(trained_j, initial_type_fractions=trained_fractions)
    state, _ = pattern_model.sample_initial_celltypes(model, base_state, type_key)
    trajectory = jxm.simulate(model, state, sim_key, n_steps, history=True)
    if isinstance(trajectory, tuple):
        trajectory = trajectory[0]
    final_state = jax.tree_util.tree_map(lambda x: x[-1], trajectory)
    return final_state, trained_j, trained_fractions


def summarize_progress(run_dir):
    progress_path = run_dir / "training_progress.json"
    if not progress_path.exists():
        return {}
    progress = read_json(progress_path)
    if not progress:
        return {}
    finite_losses = [row["train_loss"] for row in progress if row["train_loss"] == row["train_loss"]]
    best_loss = min(finite_losses) if finite_losses else None
    best_epoch = next((row["epoch"] for row in progress if row["train_loss"] == best_loss), None)
    final = progress[-1]
    return {
        "final_train_loss": final["train_loss"],
        "best_train_loss": best_loss,
        "best_train_loss_epoch": best_epoch,
        "final_type_counts_from_training": final.get("type_counts"),
        "final_type_fractions_from_training": final.get("type_fractions"),
    }


def make_gallery(rows, gallery_path):
    rows = sorted(rows, key=lambda row: (row.get("best_train_loss") is None, row.get("best_train_loss", 1e9)))
    parts = [
        "<html><head><meta charset='utf-8'><title>Pattern sweep gallery</title></head>",
        "<body style='font-family: sans-serif'>",
        "<h1>Pattern Sweep Gallery</h1>",
        "<div style='display:flex; flex-wrap:wrap; gap:14px'>",
    ]
    gallery_dir = gallery_path.parent
    for row in rows:
        full_src = html.escape(os.path.relpath(row["full_image"], gallery_dir))
        hidden_src = html.escape(os.path.relpath(row["hidden_image"], gallery_dir))
        label = html.escape(
            f"{row['combo']} / {row['run']} | best={row.get('best_train_loss')} | "
            f"visual_loss={row['visual_loss']:.3f} | counts={row['visual_counts']}"
        )
        j_text = html.escape("J=" + onp.array2string(onp.asarray(row["trained_j"]), precision=2))
        component_text = html.escape(
            f"m=({row['m1']:.2f},{row['m2']:.2f},{row['m3']:.2f}), "
            f"f12={row['f12']:.2f}, f33={row['f33']:.2f}, shell={row['shell_distance']:.2f}"
        )
        parts.append(
            "<div style='width:430px; border:1px solid #ccc; padding:8px'>"
            f"<div><b>{label}</b></div>"
            "<div style='display:flex; gap:6px'>"
            f"<div><div>all types</div><img src='{full_src}' style='width:205px'></div>"
            f"<div><div>type 3 hidden</div><img src='{hidden_src}' style='width:205px'></div>"
            "</div>"
            f"<div style='font-size:12px'>{j_text}</div>"
            f"<div style='font-size:12px'>{component_text}</div>"
            "</div>"
        )
    parts.append("</div></body></html>")
    gallery_path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    sweep_root = args.sweep_root.expanduser().resolve()
    plotter = load_plotter()
    rows = []
    skipped = []

    for hyper_path in sorted(sweep_root.glob("wfrac_*/w12_*/ratio_*/ncells_*/train-pattern-opt-hyperparams.json")):
        combo_dir = hyper_path.parent
        hyperparams = read_json(hyper_path)
        pattern = hyperparams.get("pattern", "salt-pepper-shell")
        loss_kwargs = {
            "shell_distance_weight": float(hyperparams.get("shell_distance_weight", 1.0)),
            "w12": float(hyperparams.get("w12", 0.0)),
            "w33": float(hyperparams.get("w33", 1.0)),
            "w_media1": float(hyperparams.get("w_media1", 10.0)),
            "w_media2": float(hyperparams.get("w_media2", 10.0)),
            "w_media3": float(hyperparams.get("w_media3", 10.0)),
        }
        combo_label = str(combo_dir.relative_to(sweep_root))
        image_dir = combo_dir / "visualizations"
        image_dir.mkdir(exist_ok=True)
        print(f"Visualizing combo {combo_label}", flush=True)

        run_dirs = sorted([p for p in combo_dir.glob("train-pattern-*") if p.is_dir()])
        if args.max_runs is not None:
            run_dirs = run_dirs[: args.max_runs]

        for run_dir in run_dirs:
            missing = [
                name
                for name in ("j_parameters.json", "training_progress.json")
                if not (run_dir / name).exists()
            ]
            if missing:
                skipped.append({"run_dir": str(run_dir), "missing": missing})
                print(f"Skipping incomplete run {run_dir}: missing {', '.join(missing)}", flush=True)
                continue

            run_id = run_dir.name.split("-")[-1]
            print(f"  visualizing {combo_label}/{run_dir.name}", flush=True)
            final_state, trained_j, trained_fractions = final_state_for_run(
                run_dir,
                hyperparams,
                seed=args.seed + int(run_id),
            )
            metrics = final_state_metrics(final_state, pattern, loss_kwargs)
            full_image = image_dir / f"{run_dir.name}-final.png"
            hidden_image = image_dir / f"{run_dir.name}-final-hide-type-3.png"
            title_prefix = combo_label.replace("/", ", ")
            plotter(final_state, full_image, f"{title_prefix}, {run_dir.name}", metrics, trained_j)
            plotter(
                final_state,
                hidden_image,
                f"{title_prefix}, {run_dir.name}, type 3 hidden",
                metrics,
                trained_j,
                cell_mask=visible_cell_mask(final_state, hidden_types=[3]),
            )
            progress_summary = summarize_progress(run_dir)
            rows.append(
                {
                    "combo": combo_label,
                    "run": run_dir.name,
                    "trained_j": onp.asarray(trained_j).tolist(),
                    "trained_type_fractions": onp.asarray(trained_fractions).tolist(),
                    "visual_loss": metrics["loss"],
                    "visual_counts": [metrics["count_type_1"], metrics["count_type_2"], metrics["count_type_3"]],
                    "full_image": str(full_image),
                    "hidden_image": str(hidden_image),
                    **{key: metrics[key] for key in ("m1", "m2", "m3", "f12", "f33", "shell_distance")},
                    **progress_summary,
                }
            )

    summary_csv = sweep_root / "visualization_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "combo",
            "run",
            "best_train_loss",
            "final_train_loss",
            "visual_loss",
            "visual_counts",
            "trained_type_fractions",
            "trained_j",
            "m1",
            "m2",
            "m3",
            "f12",
            "f33",
            "shell_distance",
            "full_image",
            "hidden_image",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (row.get("best_train_loss") is None, row.get("best_train_loss", 1e9))):
            writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})

    gallery_path = sweep_root / "visualization_gallery.html"
    make_gallery(rows, gallery_path)
    skipped_path = sweep_root / "visualization_skipped_runs.json"
    with open(skipped_path, "w") as f:
        json.dump(skipped, f, indent=2)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {gallery_path}")
    print(f"Wrote {skipped_path}")
    print(f"Generated {len(rows)} visualized runs")
    print(f"Skipped {len(skipped)} incomplete runs")


if __name__ == "__main__":
    main()
