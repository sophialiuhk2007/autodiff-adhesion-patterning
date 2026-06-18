import jax
import jax.numpy as np
import jax_md
import jax_md.space  # type: ignore
import equinox as eqx

import jax_morph as jxm  # type: ignore


def build_istate(init_key, *, n_type_1=20, n_type_2=40):

    N_DIM = 3  # in 3D
    disp, shift = jax_md.space.free()
    n_cells = n_type_1 + n_type_2

    class CellState(jxm.BaseCellState):
        division: jax.Array

    istate = CellState(
        displacement=disp,
        shift=shift,
        position=np.zeros(shape=(n_cells, N_DIM)),
        celltype=np.zeros(shape=(n_cells, 2)).at[0].set([1.0, 0.0]),
        radius=np.zeros(shape=(n_cells, 1)).at[0].set(0.5),
        division=np.zeros(shape=(n_cells, 1)).at[0].set(1.0),
    )

    mech_potential = jxm.env.mechanics.MorsePotential(epsilon=3.0, alpha=2.8)
    init_model = jxm.Sequential(
        substeps=[
            jxm.env.CellDivision(),
            jxm.env.CellGrowth(growth_rate=0.03, max_radius=0.5, growth_type="linear"),
            jxm.env.mechanics.SGDMechanicalRelaxation(mech_potential, relaxation_steps=10),
        ]
    )

    init_state, _ = jxm.simulate(init_model, istate, init_key, n_cells - 1)

    key_type = jax.random.fold_in(init_key, 1)
    celltype = np.concatenate(
        [
            np.tile(np.array([[1.0, 0.0]]), (n_type_1, 1)),
            np.tile(np.array([[0.0, 1.0]]), (n_type_2, 1)),
        ],
        axis=0,
    )
    perm = jax.random.permutation(key_type, n_cells)
    celltype = celltype[perm]
    division = np.ones((n_cells, 1))

    init_state = eqx.tree_at(lambda s: s.celltype, init_state, celltype)  # replaces init_state.celltype = celltype
    init_state = eqx.tree_at(lambda s: s.division, init_state, division)  # replaces init_state.division = division
    return init_state


def _logit(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


# raw_j is the unconstrained trainable parameter
# #visible_j is the constrained [J_AA, J_AB, J_BB] that we want to optimize over
# We use a logit transform to map between them, with the visible_j values constrained to [epsilon_min, epsilon_max].
def raw_j_from_visible_j(visible_j, *, epsilon_min=0.8, epsilon_max=3.8):
    """Convert visible [J_AA, J_AB, J_BB] to unconstrained trainable params."""
    visible_j = np.asarray(visible_j)
    if visible_j.shape != (3,):
        raise ValueError("visible_j must be [J_AA, J_AB, J_BB].")
    if np.any(visible_j < epsilon_min) or np.any(visible_j > epsilon_max):
        raise ValueError(f"visible_j values must be in [{epsilon_min}, {epsilon_max}].")
    scaled = (visible_j - epsilon_min) / (epsilon_max - epsilon_min)
    return _logit(scaled)


def visible_j_from_raw_j(raw_j, *, epsilon_min=0.8, epsilon_max=3.8):
    """Convert trainable raw_j params to visible [J_AA, J_AB, J_BB]."""
    return jax.nn.sigmoid(raw_j) * (epsilon_max - epsilon_min) + epsilon_min


def raw_fraction_from_visible_fraction(fraction, *, fraction_min=0.02, fraction_max=0.98):
    """Convert a visible type-1 fraction to an unconstrained trainable scalar."""
    fraction = np.asarray(fraction)
    if np.any(fraction < fraction_min) or np.any(fraction > fraction_max):
        raise ValueError(f"fraction must be in [{fraction_min}, {fraction_max}].")
    scaled = (fraction - fraction_min) / (fraction_max - fraction_min)
    return _logit(scaled)


def visible_fraction_from_raw_fraction(raw_fraction, *, fraction_min=0.02, fraction_max=0.98):
    """Convert trainable raw_fraction to a visible type-1 sampling probability."""
    return jax.nn.sigmoid(raw_fraction) * (fraction_max - fraction_min) + fraction_min


# The raw matrix JAX-Morph expects is a 2x2 matrix where the diagonal entries are 2*J_AA and 2*J_BB, and the off-diagonal entries are J_AB.
# reason is just because of how the MorsePotentialSpecies is implemented
def raw_species_matrix_from_raw_j(raw_j):
    """Private adapter from [AA, AB, BB] to the 2x2 raw matrix JAX-Morph expects."""
    return np.array(
        [
            [2.0 * raw_j[0], raw_j[1]],
            [raw_j[1], 2.0 * raw_j[2]],
        ]
    )


def raw_species_matrix_from_target_alpha(target_alpha, shape, *, alpha_min=1.0, alpha_max=3.0):
    """Convert a visible target alpha value to MorsePotentialSpecies raw params."""
    scaled = (target_alpha - alpha_min) / alpha_max
    raw = np.ones(shape) * _logit(scaled)

    # Same diagonal correction as epsilon so the effective alpha is target_alpha.
    diag = np.diag(raw)
    return raw.at[np.diag_indices(raw.shape[0])].set(2.0 * diag)


def visible_species_matrix_from_raw(raw_matrix, *, matrix_min, matrix_max):
    """Apply MorsePotentialSpecies' raw-to-visible matrix transform."""
    raw_matrix = 0.5 * (np.triu(raw_matrix) + np.tril(raw_matrix).T + np.triu(raw_matrix).T + np.tril(raw_matrix))
    raw_matrix = raw_matrix - np.eye(raw_matrix.shape[0]) * 0.5 * np.diagonal(raw_matrix)
    return jax.nn.sigmoid(raw_matrix) * matrix_max + matrix_min


def build_relaxation_model_from_raw_j(raw_j):
    raw_epsilon = raw_species_matrix_from_raw_j(raw_j)
    raw_alpha = raw_species_matrix_from_target_alpha(2.8, raw_epsilon.shape)
    mech_potential = jxm.env.mechanics.MorsePotentialSpecies(
        epsilon=raw_epsilon,
        alpha=raw_alpha,
        epsilon_min=0.8,
        epsilon_max=3.0,  # range is 0.8 to 3.8 as in pg 10 of supplement
    )

    return jxm.Sequential(
        [
            jxm.env.mechanics.BrownianMechanicalRelaxation(
                mech_potential,
                kT=0.5,
                relaxation_steps=100,  # Supplement p. 10: 200 Brownian relaxation steps
                discount=0.99,  # Supplement p. 10: gradient discounting factor for optimization
            )
        ]
    )


class TrainableJModel(jxm.SimulationStep):
    """Adhesion model with trainable [J_AA, J_AB, J_BB] and initial type-1 fraction."""

    raw_j: jax.Array
    raw_fraction: jax.Array

    def return_logprob(self) -> bool:
        return False

    def __call__(self, state, *, key=None, **kwargs):
        return build_relaxation_model_from_raw_j(self.raw_j)(state, key=key)


def build_model(visible_j, *, initial_type_1_fraction=1.0 / 2.0):
    """Build the two-type adhesion model from visible J values and an initial type-1 fraction."""
    return TrainableJModel(
        raw_j_from_visible_j(visible_j),
        raw_fraction_from_visible_fraction(initial_type_1_fraction),
    )


def sample_initial_celltypes(model, state, key):
    """Sample hard cell types from the model's global type-1 fraction and return log p(sample)."""
    n_cells = state.celltype.shape[0]
    fraction = visible_fraction_from_raw_fraction(model.raw_fraction)
    n_type_1 = jax.random.binomial(key, n=n_cells - 2, p=fraction).astype(np.int32) + 1
    key_perm = jax.random.fold_in(key, 1)  # separate key for permutation to avoid correlation with n_type_1 sampling
    type_1 = np.arange(n_cells) < n_type_1  # first n_type_1 cells are type 1, rest are type 2, then we permute to randomize which are which
    type_1 = type_1[jax.random.permutation(key_perm, n_cells)].astype(np.float32)
    celltype = np.stack([type_1, 1.0 - type_1], axis=1)
    # for binomial, probability is given by nCk*p^k * (1-p)^(n-k), and we add a small epsilon to avoid log(0)
    # we use n_cells - 2 and add 1 to n_type_1 to ensure at least one cell of each type for the loss to be well-defined, which is important early in training when the fraction can be close to 0 or 1
    # then taking log gives us k*log(p) + (n-k)*log(1-p) with some constant factor from the combinatorial term
    # but we ignore since it doesn't depend on p
    logprob = (n_type_1 - 1) * np.log(fraction + 1e-8) + (n_cells - n_type_1 - 1) * np.log(1.0 - fraction + 1e-8)
    # returns a new state with cell type replaced by the sampled cell type
    # returns also the log probability of that sample under the model's current fraction parameter
    return eqx.tree_at(lambda s: s.celltype, state, celltype), logprob


def contact_frequencies(state, *, contact_distance=1.0, contact_sharpness=20.0):
    disp = jax.vmap(jax.vmap(state.displacement, in_axes=(None, 0)), in_axes=(0, None))(state.position, state.position)
    dist = np.sqrt(np.sum(disp**2, axis=-1) + 1e-8)
    contact = jax.nn.sigmoid(contact_sharpness * (contact_distance - dist))

    n_cells = state.celltype.shape[0]
    pair_mask = 1.0 - np.eye(n_cells)
    type_1 = state.celltype[:, 0]
    type_2 = state.celltype[:, 1]
    type_1_self_mask = type_1[:, None] * type_1[None, :] * pair_mask
    cross_mask = (type_1[:, None] * type_2[None, :] + type_2[:, None] * type_1[None, :]) * pair_mask

    type_1_self_frequency = np.sum(contact * type_1_self_mask) / (np.sum(type_1_self_mask) + 1e-8)
    cross_frequency = np.sum(contact * cross_mask) / (np.sum(cross_mask) + 1e-8)
    return type_1_self_frequency, cross_frequency


def core_shell_loss(
    state,
    target_r1=1.0,
    target_r2=2.0,
    type_1_self_contact_weight=0.0,
    cross_contact_weight=0.0,
    contact_distance=1.0,
    contact_sharpness=20.0,
):
    # follows from pg 10 of the supplement
    target_radii = np.array([target_r1, target_r2])
    core_type = np.argmin(target_radii)  # core is the type with smaller target radius
    core = state.celltype[:, core_type]
    center = (state.position * core[:, None]).sum(axis=0) / core.sum()
    dist = np.sqrt(np.sum((state.position - center) ** 2, axis=-1))
    target = state.celltype @ target_radii
    radial_loss = np.mean((dist - target) ** 2)

    type_1_self_frequency, cross_frequency = contact_frequencies(
        state,
        contact_distance=contact_distance,
        contact_sharpness=contact_sharpness,
    )
    return radial_loss - type_1_self_contact_weight * type_1_self_frequency - cross_contact_weight * cross_frequency


def core_shell_trajectory_loss(
    target_r1=1.0,
    target_r2=2.0,
    type_1_self_contact_weight=0.0,
    cross_contact_weight=0.0,
    contact_distance=1.0,
    contact_sharpness=20.0,
):
    """Return a trajectory cost function compatible with SimpleLoss."""

    def _cost(trajectory):
        return jax.vmap(
            lambda state: core_shell_loss(
                state,
                target_r1,
                target_r2,
                type_1_self_contact_weight,
                cross_contact_weight,
                contact_distance,
                contact_sharpness,
            )
        )(trajectory)

    return _cost


def simulate_forward(model, istate, key, n_steps=50, *, history=True):
    return jxm.simulate(model, istate, key, n_steps=n_steps, history=history)
