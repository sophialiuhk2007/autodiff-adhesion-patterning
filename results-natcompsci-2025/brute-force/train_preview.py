import argparse
import json
import os
import shutil
from collections import namedtuple
from pathlib import Path

# JAX Imports
import jax
import jax.numpy as np
import optax
from tqdm import trange

# JAX-Morph Imports
import jax_morph as jxm  # type: ignore

import equinox as eqx

# Local Imports
import forward_visualization_helpers as forward_viz
import istate_and_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"


# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def random_model(key, *, j_min=0.8, j_max=3.8):
    visible_j = jax.random.uniform(key, shape=(6,), minval=j_min, maxval=j_max)
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


def simulate_final_state(model, istate, key, n_steps):
    result = jxm.simulate(model, istate, key, n_steps)
    return result[0] if isinstance(result, tuple) else result


def save_training_previews(model, run_dir, epoch, args, preview_keys):
    preview_dir = Path(run_dir) / "previews" / f"epoch_{epoch:04d}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    target_radii = np.array([args.target_r1, args.target_r2, args.target_r3])
    visible_j = istate_and_model.model_visible_j(model)
    summary = {
        "epoch": int(epoch),
        "j": np.asarray(visible_j).tolist(),
        "examples": [],
    }

    for example_i, key in enumerate(preview_keys):
        init_key, sim_key = jax.random.split(key)
        preview_istate = istate_and_model.build_istate(
            init_key,
            n_type_1=args.n_type_1,
            n_type_2=args.n_type_2,
            n_type_3=args.n_type_3,
        )
        fstate = simulate_final_state(model, preview_istate, sim_key, args.n_steps)
        output = forward_viz.save_state_visualization(
            notebook_dir=Path(script_dir),
            outdir=preview_dir,
            state=fstate,
            title=f"epoch {epoch} example {example_i}",
            filename=f"example_{example_i:02d}_final.png",
            target_j=visible_j,
            target_radii=target_radii,
            loss_fn=istate_and_model.core_shell_loss,
        )
        summary["examples"].append(
            {
                "example": int(example_i),
                "path": output["path"],
                "metrics": output["metrics"],
            }
        )

    with open(preview_dir / "preview_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def train_with_previews(
    model,
    istate,
    loss,
    *,
    key,
    epochs,
    optimizer,
    run_dir,
    args,
):
    loss_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(loss.loss_fn, has_aux=loss.has_aux)
    )

    key, subkey = jax.random.split(key)
    rl, grads = loss_and_grad(model, istate, key=subkey)

    if loss.has_aux:
        rl, loss_aux = rl
        val_losses = [float(loss_aux)]
    else:
        val_losses = None
    losses = [float(rl)]

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    preview_keys = jax.random.split(
        jax.random.PRNGKey(args.preview_seed), args.preview_n_examples
    )

    if args.preview_interval > 0:
        save_training_previews(model, run_dir, 0, args, preview_keys)

    pbar = trange(epochs, dynamic_ncols=True)
    for epoch in pbar:
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        key, subkey = jax.random.split(key)
        rl, grads = loss_and_grad(model, istate, key=subkey)

        if loss.has_aux:
            rl, loss_aux = rl
            val_losses.append(float(loss_aux))
            pbar.set_description(f"Loss: {float(loss_aux):.5f}")
        else:
            pbar.set_description(f"Epoch: {epoch}")

        losses.append(float(rl))

        current_epoch = epoch + 1
        if args.preview_interval > 0 and current_epoch % args.preview_interval == 0:
            save_training_previews(model, run_dir, current_epoch, args, preview_keys)

    if args.preview_interval > 0 and epochs % args.preview_interval != 0:
        save_training_previews(model, run_dir, epochs, args, preview_keys)

    return model, namedtuple(
        "OptimizationResults", ["model", "loss", "loss_aux", "grad"]
    )(None, losses, val_losses, None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-opt-runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--n-val-episodes", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-type-1", type=int, default=10)
    parser.add_argument("--n-type-2", type=int, default=30)
    parser.add_argument("--n-type-3", type=int, default=90)
    parser.add_argument("--target-r1", type=float, default=1.0)
    parser.add_argument("--target-r2", type=float, default=2.0)
    parser.add_argument("--target-r3", type=float, default=3.0)
    parser.add_argument("--preview-interval", type=int, default=50)
    parser.add_argument("--preview-n-examples", type=int, default=4)
    parser.add_argument("--preview-seed", type=int, default=1234)
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

    istate = istate_and_model.build_istate(
        init_key,
        n_type_1=args.n_type_1,
        n_type_2=args.n_type_2,
        n_type_3=args.n_type_3,
    )

    # Optimization parameters
    N_OPT_RUNS = args.n_opt_runs  # number of independent training runs with different random initializations
    EPOCHS = args.epochs  # number of Adam gradient descent updates per optimization run
    N_EPISODES = args.n_episodes  # number of trajectories to simulate per epoch for computing the training loss
    N_VAL_EPISODES = args.n_val_episodes  # number of trajectories to simulate for validation (not used for training, but can be used to monitor performance and select the best model)
    N_STEPS = args.n_steps  # number of simulation steps per episode (i.e. trajectory length)
    LEARNING_RATE = args.learning_rate

    COST_FN = istate_and_model.core_shell_trajectory_loss(
        target_r1=args.target_r1,
        target_r2=args.target_r2,
        target_r3=args.target_r3,
    )
    LOSS = jxm.opt._old.losses.SimpleLoss(
        COST_FN,
        n_sim_steps=N_STEPS,
        n_episodes=N_EPISODES,
        n_val_episodes=N_VAL_EPISODES,
        lambda_l1=0.0,
        normalize_cost_returns=False,
    )

    OPTIMIZER = optax.adam(LEARNING_RATE)

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
                "N_TYPE_1": args.n_type_1,
                "N_TYPE_2": args.n_type_2,
                "N_TYPE_3": args.n_type_3,
                "TARGET_R1": args.target_r1,
                "TARGET_R2": args.target_r2,
                "TARGET_R3": args.target_r3,
                "PREVIEW_INTERVAL": args.preview_interval,
                "PREVIEW_N_EXAMPLES": args.preview_n_examples,
                "PREVIEW_SEED": args.preview_seed,
            },
            f,
        )

    for i in range(N_OPT_RUNS):

        key, model_key, train_key = jax.random.split(key, 3)

        model = random_model(model_key)
        run_dir = os.path.join(root_dir, f"train-concentric-{i}")
        os.makedirs(run_dir, exist_ok=True)

        opt_model, opt_results = train_with_previews(
            model,
            istate,
            LOSS,
            key=train_key,
            epochs=EPOCHS,
            optimizer=OPTIMIZER,
            run_dir=run_dir,
            args=args,
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
