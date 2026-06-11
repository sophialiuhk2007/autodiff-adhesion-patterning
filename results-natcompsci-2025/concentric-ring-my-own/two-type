import argparse
import json
import os
from dataclasses import asdict, dataclass

import jax

jax.config.update("jax_enable_x64", True)

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

from three_type_core_shell_temp import plot_translucent_spheres, set_axes_equal


@dataclass
class TwoTypeConfig:
    n_type_1: int = 10
    n_type_2: int = 30
    n_dim: int = 3
    cell_radius: float = 0.5
    init_spread: float = 0.7
    target_r1: float = 0.75
    target_r2: float = 1.5
    n_steps: int = 50
    relaxation_steps: int = 200
    relaxation_dt: float = 1e-4
    brownian_kT: float = 0.6
    brownian_gamma: float = 0.8
    alpha: float = 2.8
    j_min: float = 0.8
    j_max: float = 3.8
    r_onset: float = 1.7
    r_cutoff: float = 2.0
    seed: int = 2

    @property
    def n_cells(self):
        return self.n_type_1 + self.n_type_2


class TwoTypeCellState(jxm.BaseCellState):
    pass


def logit(x):
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x / (1.0 - x))


def raw_species_matrix_from_effective_j(effective_j, cfg):
    effective_j = jnp.asarray(effective_j)
    scaled = (effective_j - cfg.j_min) / (cfg.j_max - cfg.j_min)
    raw = logit(scaled)
    diag = jnp.diag(raw)
    return raw.at[jnp.diag_indices(2)].set(2.0 * diag)


def build_initial_state(key, cfg):
    disp, shift = jax_md.space.free()
    key_pos, key_perm = jax.random.split(key, 2)
    position = cfg.init_spread * jax.random.normal(
        key_pos, shape=(cfg.n_cells, cfg.n_dim)
    )
    celltype = jnp.concatenate(
        [
            jnp.tile(jnp.array([[1.0, 0.0]]), (cfg.n_type_1, 1)),
            jnp.tile(jnp.array([[0.0, 1.0]]), (cfg.n_type_2, 1)),
        ],
        axis=0,
    )
    perm = jax.random.permutation(key_perm, cfg.n_cells)
    return TwoTypeCellState(
        displacement=disp,
        shift=shift,
        position=position[perm],
        celltype=celltype[perm],
        radius=cfg.cell_radius * jnp.ones((cfg.n_cells, 1)),
    )


def build_model(effective_j, cfg):
    potential = jxm.env.mechanics.MorsePotentialSpecies(
        epsilon=raw_species_matrix_from_effective_j(effective_j, cfg),
        alpha=jnp.zeros((2, 2)),
        epsilon_min=cfg.j_min,
        epsilon_max=cfg.j_max - cfg.j_min,
        alpha_min=cfg.alpha,
        alpha_max=0.0,
        r_onset=cfg.r_onset,
        r_cutoff=cfg.r_cutoff,
    )
    return jxm.Sequential(
        [
            jxm.env.mechanics.BrownianMechanicalRelaxation(
                potential,
                relaxation_steps=cfg.relaxation_steps,
                dt=cfg.relaxation_dt,
                kT=cfg.brownian_kT,
                gamma=cfg.brownian_gamma,
            )
        ]
    )


def simulate(effective_j, istate, key, cfg):
    return jxm.simulate(build_model(effective_j, cfg), istate, key=key, n_steps=cfg.n_steps)


def simulate_history(effective_j, istate, key, cfg):
    return jxm.simulate(
        build_model(effective_j, cfg), istate, key=key, n_steps=cfg.n_steps, history=True
    )


def paper_core_shell_loss(state, cfg):
    core = state.celltype[:, 0]
    center = (state.position * core[:, None]).sum(axis=0) / (core.sum() + 1e-8)
    dist = jnp.sqrt(jnp.sum((state.position - center) ** 2, axis=-1) + 1e-8)
    target = state.celltype @ jnp.array([cfg.target_r1, cfg.target_r2])
    alive = state.celltype.sum(axis=-1)
    return jnp.sum(alive * (dist - target) ** 2) / (alive.sum() + 1e-8)


def time_averaged_paper_loss(trajectory, cfg):
    return jnp.mean(jax.vmap(lambda state: paper_core_shell_loss(state, cfg))(trajectory))


def diagnostics(state, cfg):
    type_idx = jnp.argmax(state.celltype, axis=1)
    core = state.celltype[:, 0]
    center = (state.position * core[:, None]).sum(axis=0) / (core.sum() + 1e-8)
    dist = jnp.sqrt(jnp.sum((state.position - center) ** 2, axis=-1) + 1e-8)
    target = state.celltype @ jnp.array([cfg.target_r1, cfg.target_r2])
    loss = paper_core_shell_loss(state, cfg)

    pair_dist = jnp.sqrt(
        jnp.sum((state.position[:, None, :] - state.position[None, :, :]) ** 2, axis=-1)
        + 1e-8
    )
    contact = (pair_dist < 2.5 * float(state.radius[0, 0])) & (
        ~jnp.eye(state.position.shape[0], dtype=bool)
    )
    c11 = int(jnp.sum(contact & (type_idx[:, None] == 0) & (type_idx[None, :] == 0)) // 2)
    c12 = int(jnp.sum(contact & (type_idx[:, None] == 0) & (type_idx[None, :] == 1)))
    c22 = int(jnp.sum(contact & (type_idx[:, None] == 1) & (type_idx[None, :] == 1)) // 2)

    return {
        "loss": float(loss),
        "paper_final_loss": float(loss),
        "count_type_1": int(jnp.sum(type_idx == 0)),
        "count_type_2": int(jnp.sum(type_idx == 1)),
        "mean_radius_type_1": float(jnp.mean(dist[type_idx == 0])),
        "mean_radius_type_2": float(jnp.mean(dist[type_idx == 1])),
        "std_radius_type_1": float(jnp.std(dist[type_idx == 0])),
        "std_radius_type_2": float(jnp.std(dist[type_idx == 1])),
        "contacts_type_1_type_1": c11,
        "contacts_type_1_type_2": c12,
        "contacts_type_2_type_2": c22,
    }


def plot_radial(state, path, metrics, cfg, effective_j):
    type_idx = np.asarray(jnp.argmax(state.celltype, axis=1))
    core = np.asarray(state.celltype[:, 0])
    pos = np.asarray(state.position)
    center = (pos * core[:, None]).sum(axis=0) / (core.sum() + 1e-8)
    dist = np.sqrt(np.sum((pos - center) ** 2, axis=-1))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(dist[type_idx == 0], bins=10, alpha=0.65, color="#3452b8", label="type 1")
    ax.hist(dist[type_idx == 1], bins=10, alpha=0.65, color="#d4ad4f", label="type 2")
    ax.axvline(cfg.target_r1, color="#3452b8", linestyle="--", linewidth=2.0)
    ax.axvline(cfg.target_r2, color="#d4ad4f", linestyle="--", linewidth=2.0)
    ax.set_xlabel("distance from type-1 center")
    ax.set_ylabel("cell count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.text(
        0.98,
        0.96,
        "J =\n"
        f"{np.array2string(np.asarray(effective_j), precision=3)}\n"
        f"loss: {metrics['loss']:.3f}\n"
        f"mean r1: {metrics['mean_radius_type_1']:.3f}\n"
        f"mean r2: {metrics['mean_radius_type_2']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_contacts(path, metrics, effective_j):
    labels = ["1-1", "1-2", "2-2"]
    values = [
        metrics["contacts_type_1_type_1"],
        metrics["contacts_type_1_type_2"],
        metrics["contacts_type_2_type_2"],
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#3452b8", "#7681a8", "#d4ad4f"], alpha=0.84)
    ax.set_xlabel("cell-type pair")
    ax.set_ylabel("contact count")
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.98,
        0.96,
        "J =\n"
        f"{np.array2string(np.asarray(effective_j), precision=3)}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="results-natcompsci-2025/concentric_ring/two_type_core_shell_ratio_run",
    )
    parser.add_argument("--n-type-1", type=int, default=10)
    parser.add_argument("--n-type-2", type=int, default=30)
    parser.add_argument("--init-spread", type=float, default=TwoTypeConfig.init_spread)
    parser.add_argument("--brownian-kT", type=float, default=TwoTypeConfig.brownian_kT)
    parser.add_argument("--target-r1", type=float, default=TwoTypeConfig.target_r1)
    parser.add_argument("--target-r2", type=float, default=TwoTypeConfig.target_r2)
    parser.add_argument(
        "--j",
        nargs=4,
        type=float,
        default=[3.0, 2.5, 2.5, 1.2],
        help="Row-major 2x2 effective J matrix.",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = TwoTypeConfig(
        n_type_1=args.n_type_1,
        n_type_2=args.n_type_2,
        init_spread=args.init_spread,
        brownian_kT=args.brownian_kT,
        target_r1=args.target_r1,
        target_r2=args.target_r2,
    )
    effective_j = jnp.asarray(args.j, dtype=jnp.float64).reshape(2, 2)
    effective_j = 0.5 * (effective_j + effective_j.T)

    key = jax.random.PRNGKey(cfg.seed)
    key, init_key, sim_key = jax.random.split(key, 3)
    istate = build_initial_state(init_key, cfg)
    trajectory = simulate_history(effective_j, istate, sim_key, cfg)
    fstate = jax.tree_util.tree_map(lambda x: x[-1], trajectory)
    metrics = diagnostics(fstate, cfg)
    metrics["paper_time_averaged_loss"] = float(time_averaged_paper_loss(trajectory, cfg))

    sphere_path = os.path.join(args.outdir, "two_type_core_shell_spheres.png")
    radial_path = os.path.join(args.outdir, "two_type_core_shell_radial.png")
    contact_path = os.path.join(args.outdir, "two_type_core_shell_contacts.png")
    sphere_metrics = {**metrics, "mean_radius_type_3": 0.0}
    plot_translucent_spheres(
        fstate, sphere_path, "two-type core-shell ratio", sphere_metrics, effective_j
    )
    plot_radial(fstate, radial_path, metrics, cfg, effective_j)
    plot_contacts(contact_path, metrics, effective_j)

    result = {
        "config": asdict(cfg),
        "effective_j": np.asarray(effective_j).tolist(),
        "metrics": metrics,
        "sphere_plot": sphere_path,
        "radial_plot": radial_path,
        "contact_plot": contact_path,
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
