#!/usr/bin/env python3
import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def run_label(full_image, sweep_root):
    combo_dir = full_image.parent.parent
    run_name = full_image.name.removesuffix("-final.png")
    return combo_dir.relative_to(sweep_root), run_name


def image_pairs(sweep_root):
    pairs = []
    for full_image in sorted(sweep_root.glob("wfrac_*/w12_*/ratio_*/ncells_*/visualizations/*-final.png")):
        if full_image.name.endswith("-hide-type-3.png"):
            continue
        hidden_image = full_image.with_name(full_image.name.replace("-final.png", "-final-hide-type-3.png"))
        combo_rel, run_name = run_label(full_image, sweep_root)
        run_dir = sweep_root / combo_rel / run_name
        j_params = read_json(run_dir / "j_parameters.json") or {}
        progress = read_json(run_dir / "training_progress.json") or []
        final_progress = progress[-1] if progress else {}
        best_loss = None
        if progress:
            finite_losses = [row["train_loss"] for row in progress if row["train_loss"] == row["train_loss"]]
            best_loss = min(finite_losses) if finite_losses else None

        pairs.append(
            {
                "combo": str(combo_rel),
                "run": run_name,
                "full_image": full_image,
                "hidden_image": hidden_image if hidden_image.exists() else None,
                "trained_j": j_params.get("trained_j"),
                "trained_type_fractions": j_params.get("trained_type_fractions"),
                "final_train_loss": final_progress.get("train_loss"),
                "best_train_loss": best_loss,
                "final_type_counts": final_progress.get("type_counts"),
            }
        )
    return pairs


def rel(path, base):
    return html.escape(os.path.relpath(path, base))


def fmt(value, precision=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def combo_sort_key(combo):
    values = {}
    for part in combo.split("/"):
        if "_" in part:
            key, value = part.split("_", 1)
            values[key] = value
    ratio = values.get("ratio", "")
    ratio_key = tuple(int(x) for x in ratio.split("-")) if ratio else ()
    return (
        float(values.get("wfrac", 0)),
        float(values.get("w12", 0)),
        ratio_key,
        int(values.get("ncells", 0)),
    )


def run_sort_key(run_name):
    try:
        return int(run_name.split("-")[-1])
    except ValueError:
        return run_name


def make_gallery(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["combo"]].append(row)

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Existing Pattern Visualizations</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 20px; }",
        ".combo { margin: 22px 0 30px; }",
        ".combo h2 { font-size: 18px; margin: 0 0 8px; }",
        ".run-row { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; }",
        ".card { flex: 0 0 360px; border: 1px solid #ccc; padding: 8px; }",
        ".imgs { display: flex; gap: 6px; align-items: flex-start; }",
        ".imgs img { width: 172px; }",
        ".meta { font-size: 12px; overflow-wrap: anywhere; margin-top: 4px; }",
        "</style></head><body>",
        "<h1>Existing Pattern Visualizations</h1>",
        f"<p>Found {len(rows)} full final-state images. This gallery uses already-generated PNGs only.</p>",
    ]

    for combo in sorted(grouped, key=combo_sort_key):
        combo_rows = sorted(grouped[combo], key=lambda row: run_sort_key(row["run"]))
        parts.append("<section class='combo'>")
        parts.append(f"<h2>{html.escape(combo)}</h2>")
        parts.append("<div class='run-row'>")

        for row in combo_rows:
            title = html.escape(row["run"])
            loss_text = html.escape(
                f"best loss: {fmt(row['best_train_loss'])}; final loss: {fmt(row['final_train_loss'])}; "
                f"counts: {row['final_type_counts']}"
            )
            j_text = html.escape(f"trained J: {row['trained_j']}")
            frac_text = html.escape(f"trained fractions: {row['trained_type_fractions']}")
            full_src = rel(row["full_image"], output_path.parent)
            if row["hidden_image"] is not None:
                hidden_html = f"<div><div>type 3 hidden</div><img src='{rel(row['hidden_image'], output_path.parent)}'></div>"
            else:
                hidden_html = "<div><div>type 3 hidden</div><p>missing</p></div>"

            parts.append(
                "<div class='card'>"
                f"<div><b>{title}</b></div>"
                "<div class='imgs'>"
                f"<div><div>all types</div><img src='{full_src}'></div>"
                f"{hidden_html}"
                "</div>"
                f"<div class='meta'>{loss_text}</div>"
                f"<div class='meta'>{j_text}</div>"
                f"<div class='meta'>{frac_text}</div>"
                "</div>"
            )

        parts.append("</div></section>")

    parts.append("</body></html>")
    output_path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sweep_root = args.sweep_root.expanduser().resolve()
    output_path = args.output
    if output_path is None:
        output_path = sweep_root / "existing_visualization_gallery.html"
    else:
        output_path = output_path.expanduser().resolve()

    rows = image_pairs(sweep_root)
    make_gallery(rows, output_path)
    print(f"Wrote {output_path}")
    print(f"Included {len(rows)} existing visualized runs")


if __name__ == "__main__":
    main()
