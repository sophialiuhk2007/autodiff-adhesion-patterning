import argparse
import json
import os
from dataclasses import asdict, dataclass

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax_md

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache", "matplotlib")
)
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax_morph as jxm


@dataclass
class ThreeTypeConfig:
    n_cells: int = 60
    n_type_1: int = 10
    n_type_2: int = 20
    n_type_3: int = 30
    n_dim: int = 3
    cell_radius: float = 0.5
    init_spread: float = 1.0
    target_r1: float = 1.0
    target_r2: float = 2.0
    target_r3: float = 3.0
    n_steps: int = 50
    relaxation_steps: int = 200
    relaxation_dt: float = 1e-4
    mechanics: str = "brownian"
    brownian_kT: float = 1.0
    brownian_gamma: float = 0.8
    gradient_discount: float = 1.0
    alpha: float = 2.8
    j_min: float = 0.8
    j_max: float = 3.8
    r_onset: float = 1.7
    r_cutoff: float = 2.0
    seed: int = 2


class ThreeTypeCellState(jxm.BaseCellState):
    pass


def logit(x):
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x / (1.0 - x))


def raw_species_matrix_from_effective_j(effective_j, cfg: ThreeTypeConfig):
    """Convert desired effective J values into MorsePotentialSpecies raw params.

    MorsePotentialSpecies symmetrizes its raw matrix and halves diagonal raw
    entries before applying sigmoid scaling. Doubling diagonals here makes the
    effective diagonal values match the requested J matrix.
    """
    effective_j = jnp.asarray(effective_j)
    scaled = (effective_j - cfg.j_min) / (cfg.j_max - cfg.j_min)
    raw = logit(scaled)
    diag = jnp.diag(raw)
    return raw.at[jnp.diag_indices(3)].set(2.0 * diag)


def build_initial_state(key, cfg: ThreeTypeConfig):
    if cfg.n_type_1 + cfg.n_type_2 + cfg.n_type_3 != cfg.n_cells:
        raise ValueError("type counts must sum to n_cells")

    disp, shift = jax_md.space.free()
    key_pos, key_perm = jax.random.split(key, 2)
    position = cfg.init_spread * jax.random.normal(
        key_pos, shape=(cfg.n_cells, cfg.n_dim)
    )

    celltype = jnp.concatenate(
        [
            jnp.tile(jnp.array([[1.0, 0.0, 0.0]]), (cfg.n_type_1, 1)),
            jnp.tile(jnp.array([[0.0, 1.0, 0.0]]), (cfg.n_type_2, 1)),
            jnp.tile(jnp.array([[0.0, 0.0, 1.0]]), (cfg.n_type_3, 1)),
        ],
        axis=0,
    )
    perm = jax.random.permutation(key_perm, cfg.n_cells)

    return ThreeTypeCellState(
        displacement=disp,
        shift=shift,
        position=position[perm],
        celltype=celltype[perm],
        radius=cfg.cell_radius * jnp.ones((cfg.n_cells, 1)),
    )


def build_model(effective_j, cfg: ThreeTypeConfig):
    potential = jxm.env.mechanics.MorsePotentialSpecies(
        epsilon=raw_species_matrix_from_effective_j(effective_j, cfg),
        alpha=jnp.zeros((3, 3)),
        epsilon_min=cfg.j_min,
        epsilon_max=cfg.j_max - cfg.j_min,
        alpha_min=cfg.alpha,
        alpha_max=0.0,
        r_onset=cfg.r_onset,
        r_cutoff=cfg.r_cutoff,
    )
    if cfg.mechanics == "brownian":
        relaxation = jxm.env.mechanics.BrownianMechanicalRelaxation(
            potential,
            relaxation_steps=cfg.relaxation_steps,
            dt=cfg.relaxation_dt,
            kT=cfg.brownian_kT,
            gamma=cfg.brownian_gamma,
            discount=cfg.gradient_discount,
        )
    elif cfg.mechanics == "sgd":
        relaxation = jxm.env.mechanics.SGDMechanicalRelaxation(
            potential,
            relaxation_steps=cfg.relaxation_steps,
            dt=cfg.relaxation_dt,
        )
    else:
        raise ValueError("--mechanics must be 'brownian' or 'sgd'")
    return jxm.Sequential([relaxation])


def simulate(effective_j, istate, key, cfg: ThreeTypeConfig):
    model = build_model(effective_j, cfg)
    return jxm.simulate(model, istate, key=key, n_steps=cfg.n_steps)


def core_shell_loss_three_type(state, cfg: ThreeTypeConfig):
    core_weight = state.celltype[:, 0]
    alive = state.celltype.sum(axis=-1)
    center = (state.position * core_weight[:, None]).sum(axis=0) / (
        core_weight.sum() + 1e-8
    )
    dist = jnp.sqrt(jnp.sum((state.position - center) ** 2, axis=-1) + 1e-8)
    target_radius = state.celltype @ jnp.array(
        [cfg.target_r1, cfg.target_r2, cfg.target_r3]
    )
    return jnp.sum(alive * (dist - target_radius) ** 2) / (alive.sum() + 1e-8)


def improved_core_shell_loss_three_type(
    state,
    cfg: ThreeTypeConfig,
    *,
    shell_thickness_weight=0.25,
    ordering_weight=0.5,
    coverage_weight=0.1,
    margin=0.25,
):
    """Core-shell loss with shell thickness, ordering, and coverage terms."""
    type_idx = jnp.argmax(state.celltype, axis=1)
    core_weight = state.celltype[:, 0]
    center = (state.position * core_weight[:, None]).sum(axis=0) / (
        core_weight.sum() + 1e-8
    )
    disp = state.position - center
    dist = jnp.sqrt(jnp.sum(disp**2, axis=-1) + 1e-8)
    unit = disp / dist[:, None]

    radial = core_shell_loss_three_type(state, cfg)
    means = []
    variances = []
    coverage_terms = []
    eye = jnp.eye(3) / 3.0
    for i in range(3):
        mask = (type_idx == i).astype(jnp.float64)
        denom = mask.sum() + 1e-8
        mean_r = jnp.sum(mask * dist) / denom
        centered_r = dist - mean_r
        means.append(mean_r)
        variances.append(jnp.sum(mask * centered_r**2) / denom)

        mean_u = jnp.sum(unit * mask[:, None], axis=0) / denom
        centered_u = unit - mean_u
        cov = (centered_u * mask[:, None]).T @ centered_u / denom
        coverage_terms.append(jnp.mean((cov - eye) ** 2))

    means = jnp.asarray(means)
    shell_thickness = jnp.mean(jnp.asarray(variances))
    ordering = jax.nn.softplus(margin + means[0] - means[1]) + jax.nn.softplus(
        margin + means[1] - means[2]
    )
    coverage = jnp.mean(jnp.asarray(coverage_terms))
    return (
        radial
        + shell_thickness_weight * shell_thickness
        + ordering_weight * ordering
        + coverage_weight * coverage
    )


def radial_diagnostics(state, cfg: ThreeTypeConfig):
    type_idx = jnp.argmax(state.celltype, axis=1)
    core_weight = state.celltype[:, 0]
    center = (state.position * core_weight[:, None]).sum(axis=0) / (
        core_weight.sum() + 1e-8
    )
    dist = jnp.sqrt(jnp.sum((state.position - center) ** 2, axis=-1) + 1e-8)
    metrics = {
        "loss": float(core_shell_loss_three_type(state, cfg)),
        "improved_loss": float(improved_core_shell_loss_three_type(state, cfg)),
    }
    for i in range(3):
        mask = type_idx == i
        metrics[f"count_type_{i + 1}"] = int(jnp.sum(mask))
        metrics[f"mean_radius_type_{i + 1}"] = float(jnp.mean(dist[mask]))
        metrics[f"std_radius_type_{i + 1}"] = float(jnp.std(dist[mask]))
    return metrics


def contact_diagnostics(state, contact_factor=2.5):
    type_idx = jnp.argmax(state.celltype, axis=1)
    pair_dist = jnp.sqrt(
        jnp.sum(
            (state.position[:, None, :] - state.position[None, :, :]) ** 2,
            axis=-1,
        )
        + 1e-8
    )
    threshold = contact_factor * float(state.radius[0, 0])
    contact = (pair_dist < threshold) & (~jnp.eye(state.position.shape[0], dtype=bool))
    metrics = {}
    same_total = 0
    unlike_total = 0
    for i in range(3):
        for j in range(i, 3):
            mask = (type_idx[:, None] == i) & (type_idx[None, :] == j) & contact
            if i == j:
                count = int(jnp.sum(mask) // 2)
                same_total += count
            else:
                count = int(jnp.sum(mask))
                unlike_total += count
            metrics[f"contacts_type_{i + 1}_type_{j + 1}"] = count
    metrics["contacts_same_type"] = same_total
    metrics["contacts_unlike_type"] = unlike_total
    metrics["contact_threshold"] = threshold
    return metrics


def set_axes_equal(ax, positions, radius):
    mins = np.asarray(positions).min(axis=0) - radius
    maxs = np.asarray(positions).max(axis=0) + radius
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def plot_translucent_spheres(state, path, title, metrics, effective_j, cell_mask=None):
    positions = np.asarray(state.position)
    type_idx = np.asarray(jnp.argmax(state.celltype, axis=1))
    radius = float(state.radius[0, 0])
    colors = ["#3452b8", "#d4ad4f", "#b4002c"]
    alphas = [0.56, 0.44, 0.34]

    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 15)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    visible = np.ones(positions.shape[0], dtype=bool)
    if cell_mask is not None:
        visible = np.asarray(cell_mask, dtype=bool)
    order = np.asarray([i for i in np.argsort(type_idx)[::-1] if visible[i]])
    for i in order:
        color = colors[type_idx[i]]
        ax.plot_surface(
            positions[i, 0] + radius * sphere_x,
            positions[i, 1] + radius * sphere_y,
            positions[i, 2] + radius * sphere_z,
            color=color,
            alpha=alphas[type_idx[i]],
            linewidth=0.0,
            shade=True,
            antialiased=True,
        )

    set_axes_equal(ax, positions[visible], radius)
    ax.view_init(elev=18, azim=35)
    ax.set_axis_off()
    ax.set_title(title, fontsize=18, pad=14)
    text = (
        "J =\n"
        f"{np.array2string(np.asarray(effective_j), precision=2)}\n"
        f"loss: {metrics['loss']:.3f}\n"
        f"mean r: {metrics['mean_radius_type_1']:.2f}, "
        f"{metrics['mean_radius_type_2']:.2f}, "
        f"{metrics['mean_radius_type_3']:.2f}"
    )
    ax.text2D(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_radial_summary(state, path, title, metrics, cfg: ThreeTypeConfig, effective_j):
    positions = np.asarray(state.position)
    type_idx = np.asarray(jnp.argmax(state.celltype, axis=1))
    core = np.asarray(state.celltype[:, 0])
    center = (positions * core[:, None]).sum(axis=0) / (core.sum() + 1e-8)
    radial_distance = np.sqrt(np.sum((positions - center) ** 2, axis=-1))
    colors = ["#3452b8", "#d4ad4f", "#b4002c"]
    targets = [cfg.target_r1, cfg.target_r2, cfg.target_r3]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i in range(3):
        ax.hist(
            radial_distance[type_idx == i],
            bins=12,
            alpha=0.62,
            color=colors[i],
            label=f"type {i + 1}",
        )
        ax.axvline(targets[i], color=colors[i], linestyle="--", linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel("distance from type-1 center")
    ax.set_ylabel("cell count")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.96,
        "J =\n"
        f"{np.array2string(np.asarray(effective_j), precision=2)}\n"
        f"loss: {metrics['loss']:.3f}\n"
        f"mean r1: {metrics['mean_radius_type_1']:.3f}\n"
        f"mean r2: {metrics['mean_radius_type_2']:.3f}\n"
        f"mean r3: {metrics['mean_radius_type_3']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_contact_summary(state, path, title, metrics, effective_j):
    contacts = contact_diagnostics(state)
    pair_labels = ["1-1", "1-2", "1-3", "2-2", "2-3", "3-3"]
    pair_keys = [
        "contacts_type_1_type_1",
        "contacts_type_1_type_2",
        "contacts_type_1_type_3",
        "contacts_type_2_type_2",
        "contacts_type_2_type_3",
        "contacts_type_3_type_3",
    ]
    values = [contacts[k] for k in pair_keys]
    colors = ["#3452b8", "#7681a8", "#7f2f65", "#d4ad4f", "#bd6c3d", "#b4002c"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(pair_labels, values, color=colors, alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel("cell-type pair")
    ax.set_ylabel("contact count")
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.98,
        0.96,
        "J =\n"
        f"{np.array2string(np.asarray(effective_j), precision=2)}\n"
        f"same-type contacts: {contacts['contacts_same_type']}\n"
        f"unlike-type contacts: {contacts['contacts_unlike_type']}\n"
        f"threshold: {contacts['contact_threshold']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Temporary three-type core-shell forward simulation."
    )
    parser.add_argument(
        "--outdir",
        default="results-natcompsci-2025/concentric_ring/three_type_core_shell_temp_run",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n-type-1", type=int, default=10)
    parser.add_argument("--n-type-2", type=int, default=20)
    parser.add_argument("--n-type-3", type=int, default=30)
    parser.add_argument("--init-spread", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--relaxation-steps", type=int, default=200)
    parser.add_argument("--relaxation-dt", type=float, default=1e-4)
    parser.add_argument("--mechanics", choices=["brownian", "sgd"], default="brownian")
    parser.add_argument("--brownian-kT", type=float, default=1.0)
    parser.add_argument("--brownian-gamma", type=float, default=0.8)
    parser.add_argument(
        "--j",
        nargs=9,
        type=float,
        default=[
            3.6,
            1.8,
            0.8,
            1.8,
            2.4,
            2.8,
            0.8,
            2.8,
            1.2,
        ],
        help="Row-major 3x3 effective J matrix.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = ThreeTypeConfig(
        n_cells=args.n_type_1 + args.n_type_2 + args.n_type_3,
        n_type_1=args.n_type_1,
        n_type_2=args.n_type_2,
        n_type_3=args.n_type_3,
        seed=args.seed,
        init_spread=args.init_spread,
        n_steps=args.n_steps,
        relaxation_steps=args.relaxation_steps,
        relaxation_dt=args.relaxation_dt,
        mechanics=args.mechanics,
        brownian_kT=args.brownian_kT,
        brownian_gamma=args.brownian_gamma,
    )
    effective_j = jnp.asarray(args.j, dtype=jnp.float64).reshape(3, 3)
    effective_j = 0.5 * (effective_j + effective_j.T)

    key = jax.random.PRNGKey(cfg.seed)
    key, init_key, sim_key = jax.random.split(key, 3)
    istate = build_initial_state(init_key, cfg)
    fstate = simulate(effective_j, istate, sim_key, cfg)
    metrics = radial_diagnostics(fstate, cfg)

    sphere_path = os.path.join(args.outdir, "three_type_core_shell_spheres.png")
    cutaway_path = os.path.join(args.outdir, "three_type_core_shell_cutaway.png")
    no_red_path = os.path.join(args.outdir, "three_type_core_shell_no_red.png")
    radial_path = os.path.join(args.outdir, "three_type_core_shell_radial.png")
    contact_path = os.path.join(args.outdir, "three_type_core_shell_contacts.png")
    plot_translucent_spheres(
        fstate, sphere_path, "three-type core-shell", metrics, effective_j
    )
    positions = np.asarray(fstate.position)
    type_idx = np.asarray(jnp.argmax(fstate.celltype, axis=1))
    plot_translucent_spheres(
        fstate,
        no_red_path,
        "three-type core-shell without red cells",
        metrics,
        effective_j,
        cell_mask=type_idx != 2,
    )
    plot_translucent_spheres(
        fstate,
        cutaway_path,
        "three-type core-shell cutaway",
        metrics,
        effective_j,
        cell_mask=positions[:, 1] <= np.median(positions[:, 1]),
    )
    plot_radial_summary(
        fstate, radial_path, "three-type core-shell radial target", metrics, cfg, effective_j
    )
    plot_contact_summary(
        fstate,
        contact_path,
        "three-type core-shell contact counts",
        metrics,
        effective_j,
    )

    result = {
        "config": asdict(cfg),
        "effective_j": np.asarray(effective_j).tolist(),
        "cell_counts": {
            "type_1": cfg.n_type_1,
            "type_2": cfg.n_type_2,
            "type_3": cfg.n_type_3,
            "total": cfg.n_cells,
        },
        "metrics": metrics,
        "contact_metrics": contact_diagnostics(fstate),
        "sphere_plot": sphere_path,
        "cutaway_plot": cutaway_path,
        "no_red_plot": no_red_path,
        "radial_plot": radial_path,
        "contact_plot": contact_path,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
