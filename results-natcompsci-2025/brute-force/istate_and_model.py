import jax
import jax.numpy as np
import jax_md
import jax_md.space  # type: ignore
import equinox as eqx

import jax_morph as jxm  # type: ignore


def build_istate(init_key, *, n_type_1=10, n_type_2=30, n_type_3=90):

    N_DIM = 3  # in 3D
    disp, shift = jax_md.space.free()
    n_cells = n_type_1 + n_type_2 + n_type_3

    class CellState(jxm.BaseCellState):
        division: jax.Array

    istate = CellState(
        displacement=disp,
        shift=shift,
        position=np.zeros(shape=(n_cells, N_DIM)),
        celltype=np.zeros(shape=(n_cells, 3)).at[0].set([1.0, 0.0, 0.0]),
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
            np.tile(np.array([[1.0, 0.0, 0.0]]), (n_type_1, 1)),
            np.tile(np.array([[0.0, 1.0, 0.0]]), (n_type_2, 1)),
            np.tile(np.array([[0.0, 0.0, 1.0]]), (n_type_3, 1)),
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


# raw_j is the unconstrained trainable parameter.
# visible_j is the constrained [J_11, J_12, J_13, J_22, J_23, J_33] vector.
# We use a logit transform to map between them, with the visible_j values constrained to [epsilon_min, epsilon_max].
def raw_j_from_visible_j(visible_j, *, epsilon_min=0.8, epsilon_max=3.8):
    """Convert visible [J_11, J_12, J_13, J_22, J_23, J_33] to raw params."""
    visible_j = np.asarray(visible_j)
    if visible_j.shape != (6,):
        raise ValueError("visible_j must be [J_11, J_12, J_13, J_22, J_23, J_33].")
    if np.any(visible_j < epsilon_min) or np.any(visible_j > epsilon_max):
        raise ValueError(f"visible_j values must be in [{epsilon_min}, {epsilon_max}].")
    scaled = (visible_j - epsilon_min) / (epsilon_max - epsilon_min)
    return _logit(scaled)


def visible_j_from_raw_j(raw_j, *, epsilon_min=0.8, epsilon_max=3.8):
    """Convert raw_j params to visible [J_11, J_12, J_13, J_22, J_23, J_33]."""
    return jax.nn.sigmoid(raw_j) * (epsilon_max - epsilon_min) + epsilon_min


def raw_species_matrix_from_raw_j(raw_j):
    """Private adapter from six raw J params to the 3x3 matrix JAX-Morph expects."""
    return np.array(
        [
            [2.0 * raw_j[0], raw_j[1], raw_j[2]],
            [raw_j[1], 2.0 * raw_j[3], raw_j[4]],
            [raw_j[2], raw_j[4], 2.0 * raw_j[5]],
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
                relaxation_steps=500,  # Supplement p. 10: 200 Brownian relaxation steps
                discount=0.99,  # Supplement p. 10: gradient discounting factor for optimization
            )
        ]
    )


class TrainableJModel(jxm.SimulationStep):
    """Adhesion-only model with six visible J params for three cell types."""

    raw_j: jax.Array

    def return_logprob(self) -> bool:
        return False

    def __call__(self, state, *, key=None, **kwargs):
        return build_relaxation_model_from_raw_j(self.raw_j)(state, key=key)


def build_model(visible_j):
    """Build the three-type adhesion-only model from six visible J values."""
    return TrainableJModel(raw_j_from_visible_j(visible_j))


def model_visible_j(model):
    """Return visible [J_11, J_12, J_13, J_22, J_23, J_33] from a model."""
    return visible_j_from_raw_j(model.raw_j)


def core_shell_loss(state, target_r1=1.0, target_r2=2.0, target_r3=3.0):
    # follows from pg 10 of the supplement
    target_radii = np.array([target_r1, target_r2, target_r3])
    core_type = np.argmin(target_radii)  # core is the type with smaller target radius
    core = state.celltype[:, core_type]
    center = (state.position * core[:, None]).sum(axis=0) / core.sum()
    dist = np.sqrt(np.sum((state.position - center) ** 2, axis=-1))
    target = state.celltype @ target_radii
    return np.mean((dist - target) ** 2)


def core_shell_trajectory_loss(target_r1=1.0, target_r2=2.0, target_r3=3.0):
    """Return a trajectory cost function compatible with SimpleLoss."""

    def _cost(trajectory):
        return jax.vmap(lambda state: core_shell_loss(state, target_r1, target_r2, target_r3))(trajectory)

    return _cost


def simulate_forward(model, istate, key, n_steps=50, *, history=True):
    result = jxm.simulate(model, istate, key, n_steps=n_steps, history=history)
    if history:
        trajectory = result
        fstate = jax.tree.map(lambda x: x[-1], trajectory)
        return fstate, trajectory
    return result
