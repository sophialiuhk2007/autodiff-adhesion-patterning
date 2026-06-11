import jax
import jax.numpy as np
import jax_md
import jax_md.space  # type: ignore
import equinox as eqx

import jax_morph as jxm  # type: ignore


def build_istate(init_key, *, n_type_1=10, n_type_2=30):

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


def raw_species_matrix_from_target_epsilon(target_epsilon, *, epsilon_min=0.8, epsilon_max=3.8):
    # Convert a visible target epsilon/J matrix to MorsePotentialSpecies raw params.
    target_epsilon = np.asarray(target_epsilon)
    target_epsilon = 0.5 * (target_epsilon + target_epsilon.T)
    if np.any(target_epsilon < epsilon_min) or np.any(target_epsilon > epsilon_max):
        raise ValueError(f"target_epsilon values must be in [{epsilon_min}, {epsilon_max}] " "for MorsePotentialSpecies inverse parameterization.")
    epsilon_range = epsilon_max - epsilon_min
    scaled = (target_epsilon - epsilon_min) / epsilon_range
    return _logit(scaled)


def raw_species_matrix_from_target_alpha(target_alpha, shape, *, alpha_min=1.0, alpha_max=3.0):
    """Convert a visible target alpha value to MorsePotentialSpecies raw params."""
    scaled = (target_alpha - alpha_min) / alpha_max
    return np.ones(shape) * _logit(scaled)


def build_model(target_epsilon):
    """Build a two-type adhesion-only model from a visible 2x2 epsilon/J matrix."""
    target_epsilon = np.asarray(target_epsilon)
    raw_epsilon = raw_species_matrix_from_target_epsilon(target_epsilon, epsilon_min=0.8, epsilon_max=3.8)
    raw_alpha = raw_species_matrix_from_target_alpha(2.8, raw_epsilon.shape)
    mech_potential = jxm.env.mechanics.MorsePotentialSpecies(
        epsilon=raw_epsilon,  # taget is given by J matrix and inverse parametrize
        alpha=raw_alpha,  # target is 2.8 as in fig 4 and inverse parametrize
        epsilon_min=0.8,
        epsilon_max=3.0,  # range is 0.8 to 3.8 as in pg 10 of supplement
    )

    return jxm.Sequential(
        [
            jxm.env.mechanics.BrownianMechanicalRelaxation(
                mech_potential,
                kT=0.1,
                relaxation_steps=200,  # Supplement p. 10: 200 Brownian relaxation steps
                discount=0.99,  # Supplement p. 10: gradient discounting factor for optimization
            )
        ]
    )


def core_shell_loss(state, target_r1=1.0, target_r2=2.0):
    # follows from pg 10 of the supplement
    target_radii = np.array([target_r1, target_r2])
    core_type = np.argmin(target_radii)  # core is the type with smaller target radius
    core = state.celltype[:, core_type]
    center = (state.position * core[:, None]).sum(axis=0) / core.sum()
    dist = np.sqrt(np.sum((state.position - center) ** 2, axis=-1))
    target = state.celltype @ target_radii
    return np.mean((dist - target) ** 2)


def simulate_forward(model, istate, key, n_steps=50, *, history=True):
    return jxm.simulate(model, istate, key, n_steps=n_steps, history=history)
