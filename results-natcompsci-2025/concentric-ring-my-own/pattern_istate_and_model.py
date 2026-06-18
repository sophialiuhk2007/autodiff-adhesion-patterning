import equinox as eqx
import jax
import jax.numpy as np
import jax_md
import jax_md.space  # type: ignore
import numpy as onp

import jax_morph as jxm  # type: ignore

MIN_CELL_FRACTION_PER_TYPE = 0.05


def min_cells_per_type(n_cells):
    return int(onp.ceil(MIN_CELL_FRACTION_PER_TYPE * int(n_cells)))


def counts_from_ratios(n_cells, type_ratios):
    n_cells = int(n_cells)
    type_ratios = onp.asarray(type_ratios, dtype=float)
    if type_ratios.shape != (3,):
        raise ValueError("type_ratios must have three entries: [ratio1, ratio2, ratio3].")
    min_per_type = min_cells_per_type(n_cells)
    min_total = 3 * min_per_type
    if n_cells < min_total:
        min_percent = 100.0 * MIN_CELL_FRACTION_PER_TYPE
        raise ValueError(f"n_cells must be at least {min_total} so each cell type can have at least {min_percent:.0f}% of cells.")
    if onp.any(type_ratios < 0.0) or type_ratios.sum() <= 0.0:
        raise ValueError("type_ratios must be nonnegative and sum to a positive value.")

    fractions = type_ratios / type_ratios.sum()
    extras_float = fractions * (n_cells - min_total)
    extras = onp.floor(extras_float).astype(int)
    remainder = int(n_cells - min_total - extras.sum())
    for idx in onp.argsort(-(extras_float - extras))[:remainder]:
        extras[idx] += 1
    return extras + min_per_type, fractions


def build_istate(init_key, *, n_cells=None, type_ratios=None, n_type_1=20, n_type_2=20, n_type_3=20):
    n_dim = 3
    if n_cells is not None or type_ratios is not None:
        if n_cells is None or type_ratios is None:
            raise ValueError("Pass both n_cells and type_ratios, or pass n_type_1/n_type_2/n_type_3.")
        counts, _ = counts_from_ratios(n_cells, type_ratios)
        n_type_1, n_type_2, n_type_3 = [int(x) for x in counts]
    n_cells = n_type_1 + n_type_2 + n_type_3
    disp, shift = jax_md.space.free()

    class CellState(jxm.BaseCellState):
        division: jax.Array

    istate = CellState(
        displacement=disp,
        shift=shift,
        position=np.zeros(shape=(n_cells, n_dim)),
        celltype=np.zeros(shape=(n_cells, 3)).at[0].set([1.0, 0.0, 0.0]),
        radius=np.zeros(shape=(n_cells, 1)).at[0].set(0.5),
        division=np.zeros(shape=(n_cells, 1)).at[0].set(1.0),
    )

    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=3.0, alpha=2.8)
    init_model = jxm.Sequential(
        [
            jxm.env.CellDivision(),
            jxm.env.CellGrowth(growth_rate=0.03, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(mech_potential, relaxation_steps=10),
        ]
    )
    init_state, _ = jxm.simulate(init_model, istate, init_key, n_cells - 1)

    celltype = np.concatenate(
        [
            np.tile(np.array([[1.0, 0.0, 0.0]]), (n_type_1, 1)),
            np.tile(np.array([[0.0, 1.0, 0.0]]), (n_type_2, 1)),
            np.tile(np.array([[0.0, 0.0, 1.0]]), (n_type_3, 1)),
        ],
        axis=0,
    )
    celltype = celltype[jax.random.permutation(jax.random.fold_in(init_key, 1), n_cells)]
    init_state = eqx.tree_at(lambda s: s.celltype, init_state, celltype)
    init_state = eqx.tree_at(lambda s: s.division, init_state, np.ones((n_cells, 1)))
    return init_state


def _logit(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def raw_j_from_visible_j(visible_j, *, epsilon_min=0.8, epsilon_max=3.8):
    visible_j = np.asarray(visible_j)
    if visible_j.shape != (6,):
        raise ValueError("visible_j must be [J11, J12, J13, J22, J23, J33].")
    scaled = (visible_j - epsilon_min) / (epsilon_max - epsilon_min)
    return _logit(scaled)


def visible_j_from_raw_j(raw_j, *, epsilon_min=0.8, epsilon_max=3.8):
    return jax.nn.sigmoid(raw_j) * (epsilon_max - epsilon_min) + epsilon_min


def raw_fractions_from_visible_fractions(fractions):
    fractions = np.asarray(fractions)
    if fractions.shape != (3,):
        raise ValueError("fractions must be [frac1, frac2, frac3].")
    fractions = fractions / fractions.sum()
    return np.log(np.clip(fractions, 1e-6, 1.0))


def visible_fractions_from_raw_fractions(raw_fractions):
    return jax.nn.softmax(raw_fractions)


def raw_species_matrix_from_raw_j(raw_j):
    return np.array(
        [
            [2.0 * raw_j[0], raw_j[1], raw_j[2]],
            [raw_j[1], 2.0 * raw_j[3], raw_j[4]],
            [raw_j[2], raw_j[4], 2.0 * raw_j[5]],
        ]
    )


def raw_species_matrix_from_target_alpha(target_alpha, shape, *, alpha_min=1.0, alpha_max=3.0):
    scaled = (target_alpha - alpha_min) / alpha_max
    raw = np.ones(shape) * _logit(scaled)
    diag = np.diag(raw)
    return raw.at[np.diag_indices(raw.shape[0])].set(2.0 * diag)


def build_relaxation_model_from_raw_j(raw_j):
    raw_epsilon = raw_species_matrix_from_raw_j(raw_j)
    raw_alpha = raw_species_matrix_from_target_alpha(2.8, raw_epsilon.shape)
    mech_potential = jxm.env.mechanics.MorsePotentialSpecies(epsilon=raw_epsilon, alpha=raw_alpha, epsilon_min=0.8, epsilon_max=3.0)
    return jxm.Sequential([jxm.env.mechanics.BrownianMechanicalRelaxation(mech_potential, kT=0.5, relaxation_steps=100, discount=0.99)])


class TrainablePatternJModel(jxm.SimulationStep):
    raw_j: jax.Array
    raw_fractions: jax.Array

    def return_logprob(self) -> bool:
        return False

    def __call__(self, state, *, key=None, **kwargs):
        return build_relaxation_model_from_raw_j(self.raw_j)(state, key=key)


def build_model(visible_j, *, initial_type_fractions=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)):
    return TrainablePatternJModel(raw_j_from_visible_j(visible_j), raw_fractions_from_visible_fractions(np.asarray(initial_type_fractions)))


def sample_initial_celltypes(model, state, key):
    n_cells = state.celltype.shape[0]
    min_per_type = min_cells_per_type(n_cells)
    min_total = 3 * min_per_type
    fractions = visible_fractions_from_raw_fractions(model.raw_fractions)
    sampled_extra_counts = jax.random.multinomial(key, n_cells - min_total, fractions).astype(np.int32)
    counts = sampled_extra_counts + min_per_type
    key_perm = jax.random.fold_in(key, 1)
    cell_ids = np.arange(n_cells)
    labels = np.where(cell_ids < counts[0], 0, np.where(cell_ids < counts[0] + counts[1], 1, 2))
    labels = labels[jax.random.permutation(key_perm, n_cells)]
    celltype = jax.nn.one_hot(labels, 3)
    logprob = np.sum(sampled_extra_counts * np.log(fractions + 1e-8))
    return eqx.tree_at(lambda s: s.celltype, state, celltype), logprob


def _contact_matrix(state):
    disp = jax.vmap(jax.vmap(state.displacement, in_axes=(None, 0)), in_axes=(0, None))(state.position, state.position)
    dist = np.sqrt(np.sum(disp**2, axis=-1) + 1e-8)
    contact = jax.nn.sigmoid(20.0 * (1.0 - dist))
    return contact * (1.0 - np.eye(state.celltype.shape[0]))


def contact_frequency(state, i, j):
    contact = _contact_matrix(state)
    ti = state.celltype[:, i]
    tj = state.celltype[:, j]
    if i == j:
        mask = ti[:, None] * ti[None, :] * (1.0 - np.eye(state.celltype.shape[0]))
    else:
        mask = (ti[:, None] * tj[None, :] + tj[:, None] * ti[None, :]) * (1.0 - np.eye(state.celltype.shape[0]))
    return np.sum(contact * mask) / (np.sum(mask) + 1e-8)


def media_contact_frequency(state, celltype_idx):
    contact = _contact_matrix(state)
    neighbor_contact = np.sum(contact, axis=1)
    media_exposure = 1.0 / (1.0 + neighbor_contact)
    cells = state.celltype[:, celltype_idx]
    return np.sum(cells * media_exposure) / (np.sum(cells) + 1e-8)


def centroid_distance(state, i, j):
    ti = state.celltype[:, i]
    tj = state.celltype[:, j]
    ci = np.sum(state.position * ti[:, None], axis=0) / (np.sum(ti) + 1e-8)
    cj = np.sum(state.position * tj[:, None], axis=0) / (np.sum(tj) + 1e-8)
    return np.sqrt(np.sum((ci - cj) ** 2) + 1e-8)


def shell_distance_loss(state):
    shell = state.celltype[:, 2]
    core = state.celltype[:, 0] + state.celltype[:, 1]
    center = (state.position * core[:, None]).sum(axis=0) / core.sum()
    dist = np.sqrt(np.sum((state.position - center) ** 2, axis=-1))
    return -np.sum(shell * dist) / shell.sum()


def celltype_compactness_loss(state, celltype_idx):
    cells = state.celltype[:, celltype_idx]
    center = (state.position * cells[:, None]).sum(axis=0) / cells.sum()
    dist = np.sqrt(np.sum((state.position - center) ** 2, axis=-1))
    return np.sum(cells * dist) / (cells.sum() + 1e-8)


def shell_target_radius_loss(state, target_radius):
    shell = state.celltype[:, 2]
    core = state.celltype[:, 0] + state.celltype[:, 1]
    center = (state.position * core[:, None]).sum(axis=0) / (core.sum() + 1e-8)
    dist = np.sqrt(np.sum((state.position - center) ** 2, axis=-1) + 1e-8)
    return np.sum(shell * np.abs(dist - target_radius)) / (shell.sum() + 1e-8)


def pattern_loss(
    state,
    *,
    pattern,
    shell_distance_weight=1.0,
    shell_target_radius=3.0,
    type1_compactness_weight=1.0,
    type2_compactness_weight=1.0,
    type3_compactness_weight=None,
    w11=1,
    w12=1,
    w13=1,
    w22=1,
    w23=1,
    w33=1,
    w_media1=1,
    w_media2=1,
    w_media3=1,
    w_centroid_distance=1,
):
    f11 = contact_frequency(state, 0, 0)
    f12 = contact_frequency(state, 0, 1)
    f13 = contact_frequency(state, 0, 2)
    f22 = contact_frequency(state, 1, 1)
    f23 = contact_frequency(state, 1, 2)
    f33 = contact_frequency(state, 2, 2)
    loss = shell_distance_weight * shell_distance_loss(state)

    if pattern == "salt-pepper-shell":
        return loss - w12 * f12 - w13 * f13 - w23 * f23
    if pattern == "bilobed-shell":
        m1 = media_contact_frequency(state, 0)
        m2 = media_contact_frequency(state, 1)
        m3 = media_contact_frequency(state, 2)
        d12 = centroid_distance(state, 0, 1)

        return (
            type1_compactness_weight * celltype_compactness_loss(state, 0)
            + type2_compactness_weight * celltype_compactness_loss(state, 1)
            + w_media1 * m1
            + w_media2 * m2
            - w_media3 * m3
            - w_centroid_distance * d12
            - w11 * f11
            - w22 * f22
            + w12 * f12
            - w13 * f13
            - w23 * f23
            + w33 * f33
        )
    raise ValueError(f"Unknown pattern: {pattern}")


def pattern_trajectory_loss(**kwargs):
    def _cost(trajectory):
        return jax.vmap(lambda state: pattern_loss(state, **kwargs))(trajectory)

    return _cost
