import argparse
import itertools
import json
import os
import pickle
import time
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
import optax
from tqdm import trange

import jax_morph as jxm
from three_type_core_shell_temp import plot_translucent_spheres


@dataclass
class ExperimentConfig:
    n_cells: int = 60
    n_type_a: int = 20
    n_dim: int = 3
    cell_radius: float = 0.5
    init_spread: float = 1.75
    inner_radius: float = 1.0
    outer_radius: float = 2.0
    n_steps: int = 150
    relaxation_steps: int = 25
    relaxation_dt: float = 8e-4
    epochs: int = 1000
    checkpoint_interval: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 4
    validation_batch_size: int = 64
    brownian_kT: float = 1.0
    brownian_gamma: float = 0.8
    gradient_discount: float = 0.9
    alpha: float = 2.8
    j_min: float = 0.8
    j_max: float = 3.8
    max_grad_norm: float = 1.0
    r_onset: float = 1.7
    r_cutoff: float = 2.0
    seed: int = 2


def logit(x):
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x / (1.0 - x))


def raw_from_effective_j(j_values, j_min, j_max):
    j_values = jnp.asarray(j_values)
    scaled = (j_values - j_min) / (j_max - j_min)
    return logit(scaled)


def raw_species_matrix_from_three(raw_j):
    """Map three free params to the raw matrix expected by MorsePotentialSpecies.

    The NatCompSci species potential halves diagonal raw entries during its
    symmetry transform, so the diagonal entries are doubled here.
    """
    return jnp.array([[2.0 * raw_j[0], raw_j[1]], [raw_j[1], 2.0 * raw_j[2]]])


def effective_j_from_raw(raw_j, cfg: ExperimentConfig):
    return cfg.j_min + (cfg.j_max - cfg.j_min) * jax.nn.sigmoid(raw_j)


class RingCellState(jxm.BaseCellState):
    pass


def build_initial_state(key, cfg: ExperimentConfig):
    if not 0 < cfg.n_type_a < cfg.n_cells:
        raise ValueError("n_type_a must be between 1 and n_cells - 1.")

    disp, shift = jax_md.space.free()
    key_dir, key_radius, key_perm = jax.random.split(key, 3)

    position = jax.random.normal(key_dir, shape=(cfg.n_cells, cfg.n_dim))
    position = position / jnp.linalg.norm(position, axis=1, keepdims=True)
    radii = cfg.init_spread * jax.random.uniform(
        key_radius, shape=(cfg.n_cells, 1), minval=0.15, maxval=1.0
    )
    position = position * radii

    n_a = cfg.n_type_a
    celltype = jnp.concatenate(
        [
            jnp.tile(jnp.array([[1.0, 0.0]]), (n_a, 1)),
            jnp.tile(jnp.array([[0.0, 1.0]]), (cfg.n_cells - n_a, 1)),
        ],
        axis=0,
    )
    perm = jax.random.permutation(key_perm, cfg.n_cells)

    return RingCellState(
        displacement=disp,
        shift=shift,
        position=position[perm],
        celltype=celltype[perm],
        radius=cfg.cell_radius * jnp.ones((cfg.n_cells, 1)),
    )


def build_model(raw_j, cfg: ExperimentConfig):
    # This follows the NatCompSci branch's MorsePotentialSpecies design:
    # sigmoid-bounded species interaction matrices plus Brownian relaxation.
    potential = jxm.env.mechanics.MorsePotentialSpecies(
        epsilon=raw_species_matrix_from_three(raw_j),
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
            jxm.env.mechanics.SGDMechanicalRelaxation(
                potential,
                relaxation_steps=cfg.relaxation_steps,
                dt=cfg.relaxation_dt,
            )
        ]
    )


def concentric_ring_loss(state, inner_radius=1.0, outer_radius=2.0):
    """Two-type version of the supplied core-shell radius loss."""
    type_a = state.celltype[:, 0]
    alive = state.celltype.sum(axis=-1)
    center = (state.position * type_a[:, None]).sum(axis=0) / (
        type_a.sum() + 1e-8
    )

    dist = jnp.sqrt(jnp.sum((state.position - center) ** 2, axis=-1) + 1e-8)
    target_radius = state.celltype @ jnp.array([inner_radius, outer_radius])
    squared_error = alive * (dist - target_radius) ** 2

    return squared_error.sum() / (alive.sum() + 1e-8)


def pairwise_pattern_loss(state, *, same_type: bool):
    """Supplemental Fig. 1 pairwise loss for lobe/lattice targets.

    Lobe-like target: minimize same-type pairwise distances.
    Lattice-like target: minimize unlike-type pairwise distances.
    """
    alive = state.celltype.sum(axis=-1) > 0.0
    type_idx = jnp.argmax(state.celltype, axis=1)
    pair_dist = jnp.sqrt(
        jnp.sum(
            (state.position[:, None, :] - state.position[None, :, :]) ** 2,
            axis=-1,
        )
        + 1e-8
    )
    alive_pairs = alive[:, None] & alive[None, :]
    non_self = ~jnp.eye(state.position.shape[0], dtype=bool)
    type_match = type_idx[:, None] == type_idx[None, :]
    pattern_mask = type_match if same_type else ~type_match
    mask = alive_pairs & non_self & pattern_mask
    return jnp.sum(jnp.where(mask, pair_dist, 0.0)) / (jnp.sum(alive) + 1e-8)


def lobe_loss(state):
    return pairwise_pattern_loss(state, same_type=True)


def lattice_loss(state):
    return pairwise_pattern_loss(state, same_type=False)


def state_diagnostics(state, inner_radius=1.0, outer_radius=2.0):
    type_idx = jnp.argmax(state.celltype, axis=1)
    type_a = state.celltype[:, 0]
    type_b = state.celltype[:, 1]
    center = (state.position * type_a[:, None]).sum(axis=0) / (
        type_a.sum() + 1e-8
    )
    dist = jnp.sqrt(jnp.sum((state.position - center) ** 2, axis=-1) + 1e-8)

    pair_dist = jnp.sqrt(
        jnp.sum(
            (state.position[:, None, :] - state.position[None, :, :]) ** 2,
            axis=-1,
        )
        + 1e-8
    )
    neighbor_mask = (
        (pair_dist < 2.5 * float(state.radius[0, 0]))
        & (~jnp.eye(state.position.shape[0], dtype=bool))
    )
    same_type = type_idx[:, None] == type_idx[None, :]
    contacts = neighbor_mask.sum() + 1e-8

    return {
        "loss": float(concentric_ring_loss(state, inner_radius, outer_radius)),
        "core_shell_loss": float(
            concentric_ring_loss(state, inner_radius, outer_radius)
        ),
        "lobe_loss": float(lobe_loss(state)),
        "lattice_loss": float(lattice_loss(state)),
        "mean_radius_A": float((dist * type_a).sum() / (type_a.sum() + 1e-8)),
        "mean_radius_B": float((dist * type_b).sum() / (type_b.sum() + 1e-8)),
        "same_type_contact_fraction": float((neighbor_mask & same_type).sum() / contacts),
        "cross_type_contact_fraction": float((neighbor_mask & (~same_type)).sum() / contacts),
    }


def simulate_final_state(raw_j, istate, key, cfg: ExperimentConfig):
    model = build_model(raw_j, cfg)
    return jxm.simulate(model, istate, key=key, n_steps=cfg.n_steps)


def objective(raw_j, istate, key, cfg: ExperimentConfig):
    keys = jax.random.split(key, cfg.batch_size)

    def one_sim(sim_key):
        fstate = simulate_final_state(raw_j, istate, sim_key, cfg)
        loss = concentric_ring_loss(fstate, cfg.inner_radius, cfg.outer_radius)
        return loss, fstate

    losses, fstates = jax.vmap(one_sim)(keys)
    first_fstate = jax.tree_util.tree_map(
        lambda x: x[0] if isinstance(x, jax.Array) else x,
        fstates,
    )
    return jnp.mean(losses), first_fstate


def validation_loss(raw_j, istate, key, cfg: ExperimentConfig):
    keys = jax.random.split(key, cfg.validation_batch_size)

    def one_loss(sim_key):
        fstate = simulate_final_state(raw_j, istate, sim_key, cfg)
        return concentric_ring_loss(fstate, cfg.inner_radius, cfg.outer_radius)

    return jnp.mean(jax.vmap(one_loss)(keys))


def _plot_state_on_axis(state, ax, title):
    colors = jnp.argmax(state.celltype, axis=1)
    if state.position.shape[1] == 3:
        ax.scatter(
            state.position[:, 0],
            state.position[:, 1],
            state.position[:, 2],
            c=colors,
            cmap="coolwarm",
            s=140,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.set_zlabel("z")
    else:
        ax.scatter(
            state.position[:, 0],
            state.position[:, 1],
            c=colors,
            cmap="coolwarm",
            s=140,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def _diagnostics_text(metrics):
    return "\n".join(
        [
            f"loss: {metrics['loss']:.3f}",
            f"mean r A: {metrics['mean_radius_A']:.3f}",
            f"mean r B: {metrics['mean_radius_B']:.3f}",
            f"same contacts: {metrics['same_type_contact_fraction']:.3f}",
            f"cross contacts: {metrics['cross_type_contact_fraction']:.3f}",
        ]
    )


def plot_state(state, path, title):
    if state.position.shape[1] == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
    _plot_state_on_axis(state, ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_state_with_diagnostics(state, path, title, metrics, effective_j):
    type_idx = jnp.argmax(state.celltype, axis=1)
    type_a = state.celltype[:, 0]
    center = (state.position * type_a[:, None]).sum(axis=0) / (
        type_a.sum() + 1e-8
    )
    radial_distance = jnp.sqrt(
        jnp.sum((state.position - center) ** 2, axis=-1) + 1e-8
    )

    if state.position.shape[1] == 3:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection="3d")
    else:
        fig, (ax, ax_radial) = plt.subplots(1, 2, figsize=(12, 6))

    _plot_state_on_axis(state, ax, title)
    if state.position.shape[1] == 3:
        ax_radial = fig.add_subplot(122)

    ax_radial.hist(
        radial_distance[type_idx == 0],
        bins=12,
        alpha=0.65,
        label="type A",
        color="#3f5fc4",
    )
    ax_radial.hist(
        radial_distance[type_idx == 1],
        bins=12,
        alpha=0.65,
        label="type B",
        color="#c72946",
    )
    ax_radial.axvline(1.0, color="#3f5fc4", linestyle="--", linewidth=1.5)
    ax_radial.axvline(2.0, color="#c72946", linestyle="--", linewidth=1.5)
    ax_radial.set_xlabel("distance from type-A center")
    ax_radial.set_ylabel("cell count")
    ax_radial.legend(frameon=False, loc="upper left")
    ax_radial.grid(alpha=0.2)

    ax_radial.text(
        0.98,
        0.95,
        "J = "
        f"[{effective_j[0]:.2f}, {effective_j[1]:.2f}, {effective_j[2]:.2f}]\n"
        + _diagnostics_text(metrics),
        transform=ax_radial.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pairwise_distance_diagnostics(state):
    type_idx = jnp.argmax(state.celltype, axis=1)
    pair_dist = jnp.sqrt(
        jnp.sum(
            (state.position[:, None, :] - state.position[None, :, :]) ** 2,
            axis=-1,
        )
        + 1e-8
    )
    non_self = ~jnp.eye(state.position.shape[0], dtype=bool)
    same = type_idx[:, None] == type_idx[None, :]
    unlike = ~same
    contact = (pair_dist < 2.5 * float(state.radius[0, 0])) & non_self

    return {
        "mean_same_distance": float(
            jnp.sum(jnp.where(same & non_self, pair_dist, 0.0))
            / (jnp.sum(same & non_self) + 1e-8)
        ),
        "mean_unlike_distance": float(
            jnp.sum(jnp.where(unlike & non_self, pair_dist, 0.0))
            / (jnp.sum(unlike & non_self) + 1e-8)
        ),
        "same_type_contact_fraction": float(
            jnp.sum(contact & same) / (jnp.sum(contact) + 1e-8)
        ),
        "cross_type_contact_fraction": float(
            jnp.sum(contact & unlike) / (jnp.sum(contact) + 1e-8)
        ),
    }


def plot_pairwise_state_with_diagnostics(state, path, title, metrics, effective_j):
    type_idx = jnp.argmax(state.celltype, axis=1)
    position = state.position
    if position.shape[1] == 3:
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(121, projection="3d")
    else:
        fig, (ax, ax_pairwise) = plt.subplots(1, 2, figsize=(13, 6))

    _plot_state_on_axis(state, ax, title)
    if position.shape[1] == 3:
        ax_pairwise = fig.add_subplot(122)

    same_distances = []
    unlike_distances = []
    for i in range(position.shape[0]):
        for j in range(i + 1, position.shape[0]):
            distance = float(jnp.sqrt(jnp.sum((position[i] - position[j]) ** 2)))
            if int(type_idx[i]) == int(type_idx[j]):
                same_distances.append(distance)
            else:
                unlike_distances.append(distance)

    ax_pairwise.hist(
        same_distances,
        bins=18,
        alpha=0.65,
        label="same-type pairs",
        color="#888888",
    )
    ax_pairwise.hist(
        unlike_distances,
        bins=18,
        alpha=0.65,
        label="unlike-type pairs",
        color="#36a36f",
    )
    ax_pairwise.set_xlabel("pairwise distance")
    ax_pairwise.set_ylabel("pair count")
    ax_pairwise.legend(frameon=False, loc="upper right")
    ax_pairwise.grid(alpha=0.2)
    ax_pairwise.text(
        0.02,
        0.98,
        "J = "
        f"[{effective_j[0]:.2f}, {effective_j[1]:.2f}, {effective_j[2]:.2f}]\n"
        f"loss: {metrics['loss']:.3f}\n"
        f"mean same dist: {metrics['mean_same_distance']:.3f}\n"
        f"mean unlike dist: {metrics['mean_unlike_distance']:.3f}\n"
        f"same contacts: {metrics['same_type_contact_fraction']:.3f}\n"
        f"cross contacts: {metrics['cross_type_contact_fraction']:.3f}",
        transform=ax_pairwise.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_checkpoint(untrained_state, trained_state, path, title):
    if trained_state.position.shape[1] == 3:
        fig = plt.figure(figsize=(12, 6))
        ax_untrained = fig.add_subplot(121, projection="3d")
        ax_trained = fig.add_subplot(122, projection="3d")
    else:
        fig, (ax_untrained, ax_trained) = plt.subplots(1, 2, figsize=(12, 6))

    _plot_state_on_axis(untrained_state, ax_untrained, "untrained final")
    _plot_state_on_axis(trained_state, ax_trained, "current trained final")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sphere_metrics_from_two_type_metrics(metrics):
    mean_radius_a = metrics.get("mean_radius_A", metrics.get("mean_radius_type_1"))
    mean_radius_b = metrics.get("mean_radius_B", metrics.get("mean_radius_type_2"))
    return {
        **metrics,
        "mean_radius_type_1": metrics.get("mean_radius_type_1", mean_radius_a),
        "mean_radius_type_2": metrics.get("mean_radius_type_2", mean_radius_b),
        "mean_radius_type_3": metrics.get("mean_radius_type_3", 0.0),
    }


def run_sanity_checks(istate, key, cfg: ExperimentConfig, outdir):
    cases = {
        "core_shell_refined": [3.0, 2.5, 1.2],
        "lattice_refined": [0.9, 3.8, 1.1],
        "lobe_refined": [3.0, 0.8, 3.8],
    }
    results = {}

    for name, effective_j in cases.items():
        raw_j = raw_from_effective_j(effective_j, cfg.j_min, cfg.j_max)
        fstate = simulate_final_state(raw_j, istate, key, cfg)
        metrics = state_diagnostics(fstate, cfg.inner_radius, cfg.outer_radius)
        plot_translucent_spheres(
            fstate,
            os.path.join(outdir, f"sanity_{name}.png"),
            name,
            sphere_metrics_from_two_type_metrics(metrics),
            effective_j,
        )
        results[name] = {"effective_j": effective_j, **metrics}

    return results


def run_two_type_core_shell_ratio_sanity(outdir, seed):
    from two_type_core_shell_ratio import (
        TwoTypeConfig,
        build_initial_state as build_two_type_initial_state,
        diagnostics as two_type_diagnostics,
        plot_contacts as plot_two_type_contacts,
        plot_radial as plot_two_type_radial,
        simulate_history as simulate_two_type_history,
        time_averaged_paper_loss,
    )

    case_cfg = TwoTypeConfig(
        n_type_1=10,
        n_type_2=30,
        init_spread=0.7,
        target_r1=0.75,
        target_r2=1.5,
        n_steps=50,
        relaxation_steps=200,
        relaxation_dt=1e-4,
        brownian_kT=0.6,
        seed=seed,
    )
    effective_j = jnp.asarray(
        [[3.0, 2.5], [2.5, 1.2]], dtype=jnp.float64
    )

    key = jax.random.PRNGKey(case_cfg.seed)
    key, init_key, sim_key = jax.random.split(key, 3)
    istate = build_two_type_initial_state(init_key, case_cfg)
    initial_metrics = two_type_diagnostics(istate, case_cfg)
    initial_metrics["mean_radius_A"] = initial_metrics["mean_radius_type_1"]
    initial_metrics["mean_radius_B"] = initial_metrics["mean_radius_type_2"]
    plot_translucent_spheres(
        istate,
        os.path.join(outdir, "core_shell_refined_initial_state.png"),
        "core_shell_refined initial state",
        sphere_metrics_from_two_type_metrics(initial_metrics),
        effective_j,
    )

    trajectory = simulate_two_type_history(effective_j, istate, sim_key, case_cfg)
    fstate = jax.tree_util.tree_map(lambda x: x[-1], trajectory)
    metrics = two_type_diagnostics(fstate, case_cfg)
    metrics["paper_time_averaged_loss"] = float(
        time_averaged_paper_loss(trajectory, case_cfg)
    )
    metrics["core_shell_loss"] = metrics["paper_final_loss"]
    metrics["target_loss"] = metrics["paper_final_loss"]
    metrics["mean_radius_A"] = metrics["mean_radius_type_1"]
    metrics["mean_radius_B"] = metrics["mean_radius_type_2"]

    sphere_path = os.path.join(outdir, "sanity_core_shell_refined.png")
    radial_path = os.path.join(outdir, "sanity_core_shell_refined_radial.png")
    contact_path = os.path.join(outdir, "sanity_core_shell_refined_contacts.png")
    plot_translucent_spheres(
        fstate,
        sphere_path,
        "two-type core-shell ratio",
        sphere_metrics_from_two_type_metrics(metrics),
        effective_j,
    )
    plot_two_type_radial(fstate, radial_path, metrics, case_cfg, effective_j)
    plot_two_type_contacts(contact_path, metrics, effective_j)

    return {
        "effective_j": np.asarray(effective_j).tolist(),
        "n_cells": case_cfg.n_cells,
        "n_type_a": case_cfg.n_type_1,
        "n_type_b": case_cfg.n_type_2,
        "plot": sphere_path,
        "radial_plot": radial_path,
        "contact_plot": contact_path,
        "config": asdict(case_cfg),
        **metrics,
    }


def build_ratio_style_initial_state(key, cfg: ExperimentConfig):
    from two_type_core_shell_ratio import (
        TwoTypeConfig,
        build_initial_state as build_two_type_initial_state,
    )

    ratio_cfg = TwoTypeConfig(
        n_type_1=cfg.n_type_a,
        n_type_2=cfg.n_cells - cfg.n_type_a,
        n_dim=cfg.n_dim,
        cell_radius=cfg.cell_radius,
        init_spread=0.7,
        target_r1=cfg.inner_radius,
        target_r2=cfg.outer_radius,
        n_steps=cfg.n_steps,
        relaxation_steps=cfg.relaxation_steps,
        relaxation_dt=cfg.relaxation_dt,
        brownian_kT=cfg.brownian_kT,
        brownian_gamma=cfg.brownian_gamma,
        alpha=cfg.alpha,
        j_min=cfg.j_min,
        j_max=cfg.j_max,
        r_onset=cfg.r_onset,
        r_cutoff=cfg.r_cutoff,
        seed=cfg.seed,
    )
    return build_two_type_initial_state(key, ratio_cfg)


def run_best_sanity_suite(cfg: ExperimentConfig, outdir):
    sanity_seed = cfg.seed
    results = {
        "core_shell_refined": run_two_type_core_shell_ratio_sanity(outdir, sanity_seed),
    }
    cases = {
        "lattice_refined": {
            "effective_j": [2.6, 2.55, 2.5],
            "n_cells": 40,
            "n_type_a": 20,
            "loss_fn": lambda state, case_cfg: lattice_loss(state),
        },
        "lobe_refined": {
            "effective_j": [3.0, 0.8, 3.8],
            "n_cells": 40,
            "n_type_a": 20,
            "loss_fn": lambda state, case_cfg: lobe_loss(state),
        },
    }

    for name, case in cases.items():
        case_cfg = ExperimentConfig(
            n_cells=case.get("n_cells", cfg.n_cells),
            n_type_a=case["n_type_a"],
            n_dim=cfg.n_dim,
            cell_radius=cfg.cell_radius,
            init_spread=case.get("init_spread", cfg.init_spread),
            inner_radius=case.get("inner_radius", cfg.inner_radius),
            outer_radius=case.get("outer_radius", cfg.outer_radius),
            n_steps=cfg.n_steps,
            relaxation_steps=cfg.relaxation_steps,
            relaxation_dt=cfg.relaxation_dt,
            epochs=cfg.epochs,
            checkpoint_interval=cfg.checkpoint_interval,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            validation_batch_size=cfg.validation_batch_size,
            brownian_kT=case.get("brownian_kT", cfg.brownian_kT),
            brownian_gamma=cfg.brownian_gamma,
            gradient_discount=cfg.gradient_discount,
            alpha=cfg.alpha,
            j_min=cfg.j_min,
            j_max=cfg.j_max,
            max_grad_norm=cfg.max_grad_norm,
            r_onset=cfg.r_onset,
            r_cutoff=cfg.r_cutoff,
            seed=sanity_seed,
        )
        key = jax.random.PRNGKey(case_cfg.seed)
        key, init_key, sim_key = jax.random.split(key, 3)
        istate = build_ratio_style_initial_state(init_key, case_cfg)
        effective_j = case["effective_j"]
        initial_metrics = state_diagnostics(
            istate, case_cfg.inner_radius, case_cfg.outer_radius
        )
        initial_metrics.update(pairwise_distance_diagnostics(istate))
        initial_metrics["loss"] = float(case["loss_fn"](istate, case_cfg))
        plot_translucent_spheres(
            istate,
            os.path.join(outdir, f"{name}_initial_state.png"),
            f"{name} initial state",
            sphere_metrics_from_two_type_metrics(initial_metrics),
            effective_j,
        )

        raw_j = raw_from_effective_j(effective_j, case_cfg.j_min, case_cfg.j_max)
        fstate = simulate_final_state(raw_j, istate, sim_key, case_cfg)
        metrics = state_diagnostics(
            fstate, case_cfg.inner_radius, case_cfg.outer_radius
        )
        metrics.update(pairwise_distance_diagnostics(fstate))
        metrics["loss"] = float(case["loss_fn"](fstate, case_cfg))
        metrics["target_loss"] = metrics["loss"]

        plot_path = os.path.join(outdir, f"sanity_{name}.png")
        plot_translucent_spheres(
            fstate,
            plot_path,
            name,
            sphere_metrics_from_two_type_metrics(metrics),
            effective_j,
        )

        results[name] = {
            "effective_j": effective_j,
            "n_cells": case_cfg.n_cells,
            "n_type_a": case_cfg.n_type_a,
            "n_type_b": case_cfg.n_cells - case_cfg.n_type_a,
            "seed": case_cfg.seed,
            "plot": plot_path,
            **metrics,
        }

    return results


def train(cfg: ExperimentConfig, outdir):
    key = jax.random.PRNGKey(cfg.seed)
    key, init_key, sim_key, val_key = jax.random.split(key, 4)
    istate = build_initial_state(init_key, cfg)

    init_effective_j = jnp.array([2.0, 1.0, 2.0])
    raw_j = raw_from_effective_j(init_effective_j, cfg.j_min, cfg.j_max)
    untrained_raw_j = raw_j
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.learning_rate),
    )
    opt_state = optimizer.init(raw_j)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    log = {
        "loss": [],
        "validation_loss": [],
        "effective_j": [],
    }

    checkpoint_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    untrained_final_state = simulate_final_state(untrained_raw_j, istate, sim_key, cfg)

    pbar = trange(cfg.epochs, desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        (loss, fstate), grads = value_and_grad(raw_j, istate, sim_key, cfg)
        if not bool(jnp.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}. Try lowering "
                "--learning-rate or --max-grad-norm."
            )
        updates, opt_state = optimizer.update(grads, opt_state, raw_j)
        raw_j = optax.apply_updates(raw_j, updates)
        if not bool(jnp.all(jnp.isfinite(raw_j))):
            raise FloatingPointError(
                f"Non-finite J parameters after epoch {epoch}. Try lowering "
                "--learning-rate or --max-grad-norm."
            )

        effective_j = effective_j_from_raw(raw_j, cfg)

        log["loss"].append(float(loss))
        log["effective_j"].append([float(x) for x in effective_j])

        postfix = {
            "loss": f"{float(loss):.4f}",
            "J_AA": f"{float(effective_j[0]):.3f}",
            "J_AB": f"{float(effective_j[1]):.3f}",
            "J_BB": f"{float(effective_j[2]):.3f}",
        }

        if epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1:
            key, val_subkey = jax.random.split(val_key)
            val_key = key
            val_loss = validation_loss(raw_j, istate, val_subkey, cfg)
            log["validation_loss"].append(
                {"epoch": epoch, "loss": float(val_loss)}
            )
            postfix["val_loss"] = f"{float(val_loss):.4f}"

        checkpoint_epoch = epoch + 1
        if (
            cfg.checkpoint_interval > 0
            and (
                checkpoint_epoch % cfg.checkpoint_interval == 0
                or checkpoint_epoch == cfg.epochs
            )
        ):
            checkpoint_state = simulate_final_state(raw_j, istate, sim_key, cfg)
            checkpoint_path = os.path.join(
                checkpoint_dir, f"epoch_{checkpoint_epoch:04d}.png"
            )
            plot_training_checkpoint(
                untrained_final_state,
                checkpoint_state,
                checkpoint_path,
                f"epoch {checkpoint_epoch}",
            )
            postfix["checkpoint"] = checkpoint_path

        pbar.set_postfix(postfix)

    final_state = simulate_final_state(raw_j, istate, sim_key, cfg)
    final_j = effective_j_from_raw(raw_j, cfg)

    plot_state(istate, os.path.join(outdir, "initial_state.png"), "initial state")
    plot_state(final_state, os.path.join(outdir, "trained_final_state.png"), "trained")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log["loss"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("ring loss")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_curve.png"), dpi=200)
    plt.close(fig)

    with open(os.path.join(outdir, "training_log.pkl"), "wb") as fh:
        pickle.dump(log, fh, protocol=pickle.HIGHEST_PROTOCOL)

    jnp.savez(
        os.path.join(outdir, "trained_j_params.npz"),
        raw_j=raw_j,
        effective_j=final_j,
        j_matrix=jnp.array([[final_j[0], final_j[1]], [final_j[1], final_j[2]]]),
    )

    sanity = run_sanity_checks(istate, sim_key, cfg, outdir)
    with open(os.path.join(outdir, "sanity_checks.json"), "w") as fh:
        json.dump(sanity, fh, indent=2)

    return final_state, log


def run_forward_sanity_only(cfg: ExperimentConfig, outdir):
    sanity = run_best_sanity_suite(cfg, outdir)
    with open(os.path.join(outdir, "sanity_checks.json"), "w") as fh:
        json.dump(sanity, fh, indent=2)

    return sanity


def run_sweep_sanity(cfg: ExperimentConfig, outdir, grid_values):
    key = jax.random.PRNGKey(cfg.seed)
    key, init_key, sim_key = jax.random.split(key, 3)
    istate = build_initial_state(init_key, cfg)
    plot_state(istate, os.path.join(outdir, "initial_state.png"), "initial state")

    pattern_losses = {
        "lattice_like_mixing": lattice_loss,
        "lobe_like_sorting": lobe_loss,
        "core_shell": lambda state: concentric_ring_loss(
            state, cfg.inner_radius, cfg.outer_radius
        ),
    }
    best = {
        name: {
            "pattern_loss": float("inf"),
            "effective_j": None,
            "diagnostics": None,
            "state": None,
        }
        for name in pattern_losses
    }
    rows = []

    grid = list(itertools.product(grid_values, repeat=3))
    for effective_j in trange(len(grid), desc="Sweeping J", dynamic_ncols=True):
        effective_j = grid[effective_j]
        raw_j = raw_from_effective_j(effective_j, cfg.j_min, cfg.j_max)
        fstate = simulate_final_state(raw_j, istate, sim_key, cfg)
        diagnostics = state_diagnostics(fstate, cfg.inner_radius, cfg.outer_radius)
        row = {"effective_j": list(effective_j), **diagnostics}

        for pattern_name, loss_fn in pattern_losses.items():
            pattern_loss = float(loss_fn(fstate))
            row[f"{pattern_name}_score"] = pattern_loss
            if pattern_loss < best[pattern_name]["pattern_loss"]:
                best[pattern_name] = {
                    "pattern_loss": pattern_loss,
                    "effective_j": list(effective_j),
                    "diagnostics": diagnostics,
                    "state": fstate,
                }

        rows.append(row)

    summary = {}
    for pattern_name, result in best.items():
        path = os.path.join(outdir, f"sweep_best_{pattern_name}.png")
        plot_state_with_diagnostics(
            result["state"],
            path,
            f"best_{pattern_name}",
            result["diagnostics"],
            result["effective_j"],
        )
        summary[pattern_name] = {
            "pattern_loss": result["pattern_loss"],
            "effective_j": result["effective_j"],
            "diagnostics": result["diagnostics"],
            "plot": path,
        }

    with open(os.path.join(outdir, "sweep_results.json"), "w") as fh:
        json.dump({"grid_values": grid_values, "best": summary, "rows": rows}, fh, indent=2)

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=ExperimentConfig.n_cells)
    parser.add_argument("--n-type-a", type=int, default=ExperimentConfig.n_type_a)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=ExperimentConfig.checkpoint_interval,
    )
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument(
        "--max-grad-norm", type=float, default=ExperimentConfig.max_grad_norm
    )
    parser.add_argument("--n-steps", type=int, default=ExperimentConfig.n_steps)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=ExperimentConfig.validation_batch_size,
    )
    parser.add_argument(
        "--relaxation-steps", type=int, default=ExperimentConfig.relaxation_steps
    )
    parser.add_argument(
        "--gradient-discount",
        type=float,
        default=ExperimentConfig.gradient_discount,
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run only the manual forward-model probes and save sanity plots.",
    )
    parser.add_argument(
        "--sweep-sanity",
        action="store_true",
        help="Sweep J values and save the best lattice/lobe/core-shell examples.",
    )
    parser.add_argument(
        "--sweep-grid",
        default="0.8,1.2,1.8,2.6,3.2,3.8",
        help="Comma-separated effective J values to sweep for each of J_AA,J_AB,J_BB.",
    )
    parser.add_argument("--outdir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(
        n_cells=args.n_cells,
        n_type_a=args.n_type_a,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        relaxation_steps=args.relaxation_steps,
        gradient_discount=args.gradient_discount,
        seed=args.seed,
    )

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(
            "results-natcompsci-2025",
            "concentric_ring",
            f"run_{int(time.time())}",
        )
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    if args.sweep_sanity:
        grid_values = [float(x) for x in args.sweep_grid.split(",")]
        run_sweep_sanity(cfg, outdir, grid_values)
    elif args.sanity_only:
        run_forward_sanity_only(cfg, outdir)
    else:
        train(cfg, outdir)
    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
