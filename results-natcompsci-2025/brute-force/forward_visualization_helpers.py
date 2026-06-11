import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.animation import FuncAnimation

try:
    from IPython.display import HTML
except ModuleNotFoundError:
    HTML = None


def _load_translucent_sphere_plotter(notebook_dir):
    plot_path = Path(notebook_dir) / "translucent-3d-plotting.py"
    spec = importlib.util.spec_from_file_location("translucent_3d_plotting", plot_path)
    sphere_plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sphere_plotting)
    return sphere_plotting.plot_translucent_spheres


def adhesion_metrics(state, target_radii, loss_fn):
    type_idx = onp.asarray(onp.argmax(state.celltype, axis=1))
    target_radii = onp.asarray(target_radii)
    core_type = int(onp.argmin(target_radii))
    core = state.celltype[:, core_type]
    center = (state.position * core[:, None]).sum(axis=0) / (core.sum() + 1e-8)
    dist = onp.sqrt(onp.sum((onp.asarray(state.position) - onp.asarray(center)) ** 2, axis=-1) + 1e-8)
    n_types = int(state.celltype.shape[1])
    loss_args = [float(r) for r in target_radii[:n_types]]

    metrics = {"loss": float(loss_fn(state, *loss_args))}
    for i in range(n_types):
        mask = type_idx == i
        metrics[f"mean_radius_type_{i + 1}"] = float(onp.mean(dist[mask])) if onp.any(mask) else 0.0
        metrics[f"count_type_{i + 1}"] = int(onp.sum(mask))
    return metrics


def animate_adhesion_trajectory(
    trajectory,
    istate,
    *,
    path=None,
    frame_stride=1,
    interval=120,
    figsize=(8.5, 4.8),
):
    # Same idea as tutorials/01 - JAX-morph Simulation Basics.ipynb:
    # FuncAnimation builds an interactive HTML animation from the trajectory.
    positions = onp.concatenate(
        [onp.asarray(istate.position)[None, :, :], onp.asarray(trajectory.position)],
        axis=0,
    )
    celltypes = onp.concatenate(
        [onp.asarray(istate.celltype)[None, :, :], onp.asarray(trajectory.celltype)],
        axis=0,
    )
    radii = onp.concatenate(
        [onp.asarray(istate.radius)[None, :, :], onp.asarray(trajectory.radius)],
        axis=0,
    ).reshape(positions.shape[0], positions.shape[1])
    frame_indices = list(range(0, int(positions.shape[0]), frame_stride))
    if frame_indices[-1] != int(positions.shape[0]) - 1:
        frame_indices.append(int(positions.shape[0]) - 1)

    mins = positions.min(axis=(0, 1))
    maxs = positions.max(axis=(0, 1))
    center = 0.5 * (mins + maxs)
    span = float(onp.max(maxs - mins)) + 2.0 * float(onp.max(radii))
    colors = ["#3452b8", "#d4ad4f", "#b4002c"]
    alphas = [0.56, 0.44, 0.34]

    u = onp.linspace(0, 2 * onp.pi, 18)
    v = onp.linspace(0, onp.pi, 10)
    sphere_x = onp.outer(onp.cos(u), onp.sin(v))
    sphere_y = onp.outer(onp.sin(u), onp.sin(v))
    sphere_z = onp.outer(onp.ones_like(u), onp.cos(v))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def animate(frame_i):
        t = frame_indices[frame_i]
        ax.clear()
        pos = positions[t]
        type_idx = onp.argmax(celltypes[t], axis=1)
        order = onp.argsort(type_idx)[::-1]
        for cell_i in order:
            radius = float(radii[t, cell_i])
            ctype = int(type_idx[cell_i])
            ax.plot_surface(
                pos[cell_i, 0] + radius * sphere_x,
                pos[cell_i, 1] + radius * sphere_y,
                pos[cell_i, 2] + radius * sphere_z,
                color=colors[ctype],
                alpha=alphas[ctype],
                linewidth=0.0,
                shade=True,
                antialiased=True,
            )
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
        ax.view_init(elev=18, azim=35)
        ax.set_axis_off()
        ax.set_title(f"adhesion-only simulation t={t}", fontsize=16)

    anim = FuncAnimation(fig, animate, frames=len(frame_indices), interval=interval)
    html_text = anim.to_jshtml()
    if path is not None:
        Path(path).write_text(html_text)
    plt.close(fig)
    return HTML(html_text) if HTML is not None else html_text


def save_initial_final_visualizations(
    *,
    notebook_dir,
    outdir,
    istate,
    fstate,
    target_j,
    target_radii,
    loss_fn,
    hidden_final_type=None,
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    plot_translucent_spheres = _load_translucent_sphere_plotter(notebook_dir)
    initial_metrics = adhesion_metrics(istate, target_radii, loss_fn)
    final_metrics = adhesion_metrics(fstate, target_radii, loss_fn)

    plot_translucent_spheres(
        istate,
        outdir / "adhesion_initial_state.png",
        "adhesion-only initial state",
        initial_metrics,
        target_j,
    )
    final_cell_mask = None
    final_filename = "adhesion_forward_final_state.png"
    if hidden_final_type is not None:
        type_idx = onp.asarray(onp.argmax(fstate.celltype, axis=1))
        final_cell_mask = type_idx != int(hidden_final_type)
        final_filename = f"adhesion_forward_final_state_without_type_{int(hidden_final_type) + 1}.png"

    plot_translucent_spheres(
        fstate,
        outdir / final_filename,
        "adhesion-only forward final state",
        final_metrics,
        target_j,
        cell_mask=final_cell_mask,
    )

    return {
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "initial_path": str(outdir / "adhesion_initial_state.png"),
        "final_path": str(outdir / final_filename),
    }


def save_state_visualization(
    *,
    notebook_dir,
    outdir,
    state,
    title,
    filename,
    target_j,
    target_radii,
    loss_fn,
    hidden_type=None,
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    plot_translucent_spheres = _load_translucent_sphere_plotter(notebook_dir)
    metrics = adhesion_metrics(state, target_radii, loss_fn)
    path = outdir / filename
    cell_mask = None
    if hidden_type is not None:
        type_idx = onp.asarray(onp.argmax(state.celltype, axis=1))
        cell_mask = type_idx != int(hidden_type)
    plot_translucent_spheres(state, path, title, metrics, target_j, cell_mask=cell_mask)
    return {"path": str(path), "metrics": metrics}


def save_forward_animation(
    *,
    outdir,
    istate,
    trajectory,
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    animation_path = outdir / "adhesion_forward_animation.html"
    animation = animate_adhesion_trajectory(
        trajectory,
        istate,
        path=animation_path,
        frame_stride=1,
        interval=120,
        figsize=(8.5, 4.8),
    )
    return {"animation": animation, "animation_html": str(animation_path)}


def save_forward_summary(
    *,
    outdir,
    target_j,
    target_radii,
    n_type_1,
    n_type_2,
    n_type_3,
    n_steps,
    seed,
    initial_metrics,
    final_metrics,
    animation_html,
    hidden_final_type=None,
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    summary = {
        "target_j": onp.asarray(target_j).tolist(),
        "n_type_1": int(n_type_1),
        "n_type_2": int(n_type_2),
        "n_type_3": int(n_type_3),
        "n_steps": int(n_steps),
        "target_r1": float(target_radii[0]),
        "target_r2": float(target_radii[1]),
        "target_r3": float(target_radii[2]) if len(target_radii) > 2 else None,
        "seed": int(seed),
        "hidden_final_type": None if hidden_final_type is None else int(hidden_final_type) + 1,
        "animation_html": str(animation_html),
        "initialization": "one cell -> division + growth + mechanical relaxation; colors assigned after initialization",
        "parameter_sources": {
            "initialization": "MOESM1 Forward Simulation: one cell at origin, n-1 iterations of division/growth/mechanical relaxation",
            "adhesion_experiment": "MOESM1 Chemical Regulation of Cellular Adhesion: 3D, 50 time steps, 200 Brownian relaxation steps, no division after initialization",
            "core_shell_loss": "MOESM1 core-shell loss equation: target radii 1/2/3 for three cell types",
            "target_j": "Visible [J_11, J_12, J_13, J_22, J_23, J_33]; internally adapted to the species Morse matrix expected by JAX-Morph",
            "model_overrides": "Only epsilon scaling 0.8-3.8 and Brownian relaxation_steps=200 are passed because they differ from stock JAX-Morph defaults and are specified for the adhesion experiment",
            "model_defaults": "alpha, r_onset, r_cutoff, and Brownian dt/kT/gamma use stock JAX-Morph defaults unless the adhesion experiment specifies otherwise",
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
    }
    with open(outdir / "adhesion_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def save_forward_visualizations(
    *,
    notebook_dir,
    outdir,
    istate,
    trajectory,
    fstate,
    target_j,
    target_radii,
    loss_fn,
    n_type_1,
    n_type_2,
    n_type_3,
    n_steps,
    seed,
    hidden_final_type=None,
):
    preview = save_initial_final_visualizations(
        notebook_dir=notebook_dir,
        outdir=outdir,
        istate=istate,
        fstate=fstate,
        target_j=target_j,
        target_radii=target_radii,
        loss_fn=loss_fn,
        hidden_final_type=hidden_final_type,
    )
    animation_output = save_forward_animation(
        outdir=outdir,
        istate=istate,
        trajectory=trajectory,
    )
    summary = save_forward_summary(
        outdir=outdir,
        target_j=target_j,
        target_radii=target_radii,
        n_type_1=n_type_1,
        n_type_2=n_type_2,
        n_type_3=n_type_3,
        n_steps=n_steps,
        seed=seed,
        initial_metrics=preview["initial_metrics"],
        final_metrics=preview["final_metrics"],
        animation_html=animation_output["animation_html"],
        hidden_final_type=hidden_final_type,
    )

    return {
        "animation": animation_output["animation"],
        "summary": summary,
        "initial_path": preview["initial_path"],
        "final_path": preview["final_path"],
        "animation_html": animation_output["animation_html"],
        "initial_metrics": preview["initial_metrics"],
        "final_metrics": preview["final_metrics"],
    }
