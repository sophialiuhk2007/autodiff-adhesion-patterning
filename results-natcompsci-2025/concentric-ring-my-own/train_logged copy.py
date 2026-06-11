import argparse
import json
import os
import shutil
from collections import namedtuple

# JAX Imports
import jax
import jax.numpy as np
import optax
from tqdm import trange

# JAX-Morph Imports
import jax_morph as jxm  # type: ignore

import equinox as eqx

# Local Imports
import istate_and_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"


# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def random_model(key, *, j_min=0.8, j_max=3.8):
    visible_j = jax.random.uniform(key, shape=(3,), minval=j_min, maxval=j_max)
    return istate_and_model.build_model(visible_j)


def dump_j_parameters(path, init_model, trained_model):
    with open(path, "w") as f:
        json.dump(
            {
                "initial_j": np.asarray(istate_and_model.model_visible_j(init_model)).tolist(),
                "trained_j": np.asarray(istate_and_model.model_visible_j(trained_model)).tolist(),
            },
            f,
            indent=2,
        )


def _json_float(value):
    return None if value is None else float(value)


def write_progress_log(path, entries):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp_path, path)


def train_with_recoverable_log(
    model,
    istate,
    loss,
    *,
    key,
    epochs,
    optimizer,
    run_dir,
    learning_rate_fn,
    early_stop_patience,
    early_stop_min_delta,
):
    loss_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss.loss_fn, has_aux=loss.has_aux))

    progress_path = os.path.join(run_dir, "training_progress.json")

    key, subkey = jax.random.split(key)
    rl, grads = loss_and_grad(model, istate, key=subkey)

    if loss.has_aux:
        rl, loss_aux = rl
        val_losses = [float(loss_aux)]
    else:
        loss_aux = None
        val_losses = None
    losses = [float(rl)]
    best_train_loss = float(rl)
    epochs_without_improvement = 0
    stopped_reason = None

    progress = [
        {
            "epoch": 0,
            "train_loss": float(rl),
            "val_loss": _json_float(loss_aux),
            "learning_rate": float(learning_rate_fn(0)),
            "j": np.asarray(istate_and_model.model_visible_j(model)).tolist(),
        }
    ]
    write_progress_log(progress_path, progress)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    pbar = trange(epochs, dynamic_ncols=True)
    for epoch in pbar:
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        key, subkey = jax.random.split(key)
        rl, grads = loss_and_grad(model, istate, key=subkey)

        if loss.has_aux:
            rl, loss_aux = rl
            val_losses.append(float(loss_aux))
        else:
            loss_aux = None

        losses.append(float(rl))
        current_epoch = epoch + 1
        current_j = istate_and_model.model_visible_j(model)
        current_lr = float(learning_rate_fn(current_epoch))

        if float(rl) < best_train_loss - early_stop_min_delta:
            best_train_loss = float(rl)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress.append(
            {
                "epoch": int(current_epoch),
                "train_loss": float(rl),
                "val_loss": _json_float(loss_aux),
                "learning_rate": current_lr,
                "j": np.asarray(current_j).tolist(),
            }
        )
        write_progress_log(progress_path, progress)

        if loss.has_aux:
            pbar.set_description(f"train={float(rl):.5f}, val={float(loss_aux):.5f}, " f"lr={current_lr:.2e}, J={np.asarray(current_j).tolist()}")
        else:
            pbar.set_description(f"train={float(rl):.5f}, lr={current_lr:.2e}, " f"J={np.asarray(current_j).tolist()}")

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_reason = f"early_stop: train_loss did not improve by at least " f"{early_stop_min_delta} for {early_stop_patience} epochs"
            pbar.write(stopped_reason)
            break

    if progress:
        progress[-1]["stopped_reason"] = stopped_reason
        write_progress_log(progress_path, progress)

    return model, namedtuple("OptimizationResults", ["model", "loss", "loss_aux", "grad"])(None, losses, val_losses, None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-opt-runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--n-val-episodes", type=int, default=4)  # 64 in paper changed to 4 to reduce time
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=None,
        help="If set, exponentially decay learning rate from --learning-rate to this value over --epochs.",
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

    parser.add_argument("--n-type-1", type=int, default=20)
    parser.add_argument("--n-type-2", type=int, default=40)
    parser.add_argument("--target-r1", type=float, default=1.0)
    parser.add_argument("--target-r2", type=float, default=2.0)
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

    istate = istate_and_model.build_istate(init_key, n_type_1=args.n_type_1, n_type_2=args.n_type_2)

    # Optimization parameters
    N_OPT_RUNS = args.n_opt_runs  # number of independent training runs with different random initializations
    EPOCHS = args.epochs  # number of Adam gradient descent updates per optimization run
    N_EPISODES = args.n_episodes  # number of trajectories to simulate per epoch for computing the training loss
    N_VAL_EPISODES = args.n_val_episodes  # number of trajectories to simulate for validation (not used for training, but can be used to monitor performance and select the best model)
    N_STEPS = args.n_steps  # number of simulation steps per episode (i.e. trajectory length)
    LEARNING_RATE = args.learning_rate
    FINAL_LEARNING_RATE = args.final_learning_rate

    COST_FN = istate_and_model.core_shell_trajectory_loss(target_r1=args.target_r1, target_r2=args.target_r2)
    LOSS = jxm.opt._old.losses.SimpleLoss(
        COST_FN,
        n_sim_steps=N_STEPS,
        n_episodes=N_EPISODES,
        n_val_episodes=N_VAL_EPISODES,
        lambda_l1=0.0,
        normalize_cost_returns=False,
    )

    if FINAL_LEARNING_RATE is None:
        learning_rate_schedule = lambda step: LEARNING_RATE
        OPTIMIZER = optax.adam(LEARNING_RATE)
    else:
        learning_rate_schedule = optax.exponential_decay(
            init_value=LEARNING_RATE,
            transition_steps=max(EPOCHS, 1),
            decay_rate=FINAL_LEARNING_RATE / LEARNING_RATE,
            staircase=False,
            end_value=FINAL_LEARNING_RATE,
        )
        OPTIMIZER = optax.adam(learning_rate_schedule)

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
                "FINAL_LEARNING_RATE": FINAL_LEARNING_RATE,
                "EARLY_STOP_PATIENCE": args.early_stop_patience,
                "EARLY_STOP_MIN_DELTA": args.early_stop_min_delta,
                "N_TYPE_1": args.n_type_1,
                "N_TYPE_2": args.n_type_2,
                "TARGET_R1": args.target_r1,
                "TARGET_R2": args.target_r2,
            },
            f,
        )

    for i in range(N_OPT_RUNS):

        key, model_key, train_key = jax.random.split(key, 3)

        model = random_model(model_key)
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
