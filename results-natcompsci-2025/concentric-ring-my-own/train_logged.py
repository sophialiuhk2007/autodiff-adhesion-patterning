import argparse
import json
import os
import shutil

# JAX Imports
import jax
import jax.numpy as np
import optax

# JAX-Morph Imports
import jax_morph as jxm  # type: ignore

import equinox as eqx

# Local Imports
import istate_and_model
from training_log_helpers import InitialFractionReinforceLoss, train_with_recoverable_log

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"


# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def random_model(key, *, initial_type_1_fraction, j_min=0.8, j_max=3.8):
    visible_j = jax.random.uniform(key, shape=(3,), minval=j_min, maxval=j_max)
    return istate_and_model.build_model(visible_j, initial_type_1_fraction=initial_type_1_fraction)


def dump_j_parameters(path, init_model, trained_model):
    with open(path, "w") as f:
        json.dump(
            {
                "initial_j": np.asarray(istate_and_model.visible_j_from_raw_j(init_model.raw_j)).tolist(),
                "trained_j": np.asarray(istate_and_model.visible_j_from_raw_j(trained_model.raw_j)).tolist(),
                "initial_type_1_fraction": float(istate_and_model.visible_fraction_from_raw_fraction(init_model.raw_fraction)),
                "trained_type_1_fraction": float(istate_and_model.visible_fraction_from_raw_fraction(trained_model.raw_fraction)),
            },
            f,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-opt-runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--n-val-episodes", type=int, default=0)  # 64 in paper changed to 4 to reduce time
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "exponential", "linear"),
        default="exponential",
        help="Learning-rate schedule. Use constant to ignore --final-learning-rate.",
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=1e-3,
        help="Final learning rate for exponential or linear decay.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if training loss does not improve by --early-stop-min-delta for this many epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum training-loss improvement counted by early stopping.",
    )
    parser.add_argument("--n-cells", type=int, default=60)
    parser.add_argument("--initial-type-1-fraction", type=float, default=1.0 / 2.0)
    parser.add_argument("--target-r1", type=float, default=1.0)
    parser.add_argument("--target-r2", type=float, default=2.0)
    parser.add_argument("--type-1-self-contact-weight", type=float, default=0.1)
    parser.add_argument("--cross-contact-weight", type=float, default=0.2)
    parser.add_argument("--contact-distance", type=float, default=1.0)
    parser.add_argument("--contact-sharpness", type=float, default=20.0)
    parser.add_argument("--outdir", type=str, default="./trained_models/")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before training.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    key = jxm.utils.generate_random_key()
    key, init_key = jax.random.split(key)

    n_cells = args.n_cells
    initial_type_1_fraction = args.initial_type_1_fraction

    initial_n_type_1 = int(np.clip(round(n_cells * initial_type_1_fraction), 1, n_cells - 1))
    initial_n_type_2 = n_cells - initial_n_type_1
    initial_type_1_fraction = initial_n_type_1 / n_cells

    istate = istate_and_model.build_istate(init_key, n_type_1=initial_n_type_1, n_type_2=initial_n_type_2)

    # Optimization parameters
    N_OPT_RUNS = args.n_opt_runs  # number of independent training runs with different random initializations
    EPOCHS = args.epochs  # number of Adam gradient descent updates per optimization run
    N_EPISODES = args.n_episodes  # number of trajectories to simulate per epoch for computing the training loss
    N_VAL_EPISODES = args.n_val_episodes  # number of trajectories to simulate for validation (not used for training, but can be used to monitor performance and select the best model)
    N_STEPS = args.n_steps  # number of simulation steps per episode (i.e. trajectory length)
    LEARNING_RATE = args.learning_rate
    FINAL_LEARNING_RATE = args.final_learning_rate
    LR_SCHEDULE = args.lr_schedule

    COST_FN = istate_and_model.core_shell_trajectory_loss(
        target_r1=args.target_r1,
        target_r2=args.target_r2,
        type_1_self_contact_weight=args.type_1_self_contact_weight,
        cross_contact_weight=args.cross_contact_weight,
        contact_distance=args.contact_distance,
        contact_sharpness=args.contact_sharpness,
    )
    LOSS = InitialFractionReinforceLoss(
        COST_FN,
        n_sim_steps=N_STEPS,
        n_episodes=N_EPISODES,
        n_val_episodes=N_VAL_EPISODES,
        lambda_l1=0.0,
    )

    if LR_SCHEDULE == "constant" or FINAL_LEARNING_RATE is None:
        learning_rate_schedule = lambda step: LEARNING_RATE
        OPTIMIZER = optax.adam(LEARNING_RATE)
    elif LR_SCHEDULE == "exponential":
        learning_rate_schedule = optax.exponential_decay(
            init_value=LEARNING_RATE,
            transition_steps=max(EPOCHS, 1),
            decay_rate=FINAL_LEARNING_RATE / LEARNING_RATE,
            staircase=False,
            end_value=FINAL_LEARNING_RATE,
        )
        OPTIMIZER = optax.adam(learning_rate_schedule)
    elif LR_SCHEDULE == "linear":
        learning_rate_schedule = optax.linear_schedule(
            init_value=LEARNING_RATE,
            end_value=FINAL_LEARNING_RATE,
            transition_steps=max(EPOCHS, 1),
        )
        OPTIMIZER = optax.adam(learning_rate_schedule)
    else:
        raise ValueError(f"Unsupported learning-rate schedule: {LR_SCHEDULE}")

    root_dir = args.outdir

    # !!! empty training_runs folder
    if args.clean:
        try:
            shutil.rmtree(root_dir)
        except FileNotFoundError:
            pass

    os.makedirs(root_dir, exist_ok=True)

    # dump json with opt hyperparams
    with open(os.path.join(root_dir, "train-concentric-opt-hyperparams.json"), "w") as f:
        json.dump(
            {
                "N_OPT_RUNS": N_OPT_RUNS,
                "EPOCHS": EPOCHS,
                "N_EPISODES": N_EPISODES,
                "N_VAL_EPISODES": N_VAL_EPISODES,
                "N_STEPS": N_STEPS,
                "LEARNING_RATE": LEARNING_RATE,
                "LR_SCHEDULE": LR_SCHEDULE,
                "FINAL_LEARNING_RATE": FINAL_LEARNING_RATE,
                "EARLY_STOP_PATIENCE": args.early_stop_patience,
                "EARLY_STOP_MIN_DELTA": args.early_stop_min_delta,
                "N_CELLS": n_cells,
                "INITIAL_TYPE_1_FRACTION": initial_type_1_fraction,
                "INITIAL_N_TYPE_1": initial_n_type_1,
                "INITIAL_N_TYPE_2": initial_n_type_2,
                "TARGET_R1": args.target_r1,
                "TARGET_R2": args.target_r2,
                "TYPE_1_SELF_CONTACT_WEIGHT": args.type_1_self_contact_weight,
                "CROSS_CONTACT_WEIGHT": args.cross_contact_weight,
                "CONTACT_DISTANCE": args.contact_distance,
                "CONTACT_SHARPNESS": args.contact_sharpness,
            },
            f,
        )

    for i in range(N_OPT_RUNS):

        key, model_key, train_key = jax.random.split(key, 3)

        model = random_model(model_key, initial_type_1_fraction=initial_type_1_fraction)
        run_dir = os.path.join(root_dir, f"train-concentric-{i}")
        os.makedirs(run_dir, exist_ok=True)

        opt_model, opt_results = train_with_recoverable_log(
            model,
            istate,
            LOSS,
            key=train_key,
            epochs=EPOCHS,
            optimizer=OPTIMIZER,
            run_dir=run_dir,
            learning_rate_fn=learning_rate_schedule,
            n_cells=n_cells,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
        )

        eqx.tree_serialise_leaves(os.path.join(run_dir, f"init-concentric-{i}.eqx"), model)
        eqx.tree_serialise_leaves(
            os.path.join(run_dir, f"trained-concentric-{i}.eqx"),
            opt_model,
        )
        eqx.tree_serialise_leaves(
            os.path.join(run_dir, f"results-concentric-{i}.eqx"),
            opt_results,
        )
        dump_j_parameters(
            os.path.join(run_dir, "j_parameters.json"),
            model,
            opt_model,
        )
