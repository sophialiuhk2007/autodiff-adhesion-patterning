import argparse
import json
import os
import shutil
from collections import namedtuple

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import optax
from tqdm import trange

import jax_morph as jxm  # type: ignore
import pattern_istate_and_model as pattern_model
from jax_morph.simulation import simulate
import jax.tree_util as jtu

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

OptimizationResults = namedtuple("OptimizationResults", ["model", "loss", "loss_aux", "grad"])


Loss = namedtuple("Loss", ["loss_fn", "has_aux"])


def random_model(key, *, initial_type_fractions, j_min=0.8, j_max=3.8):
    visible_j = jax.random.uniform(key, shape=(6,), minval=j_min, maxval=j_max)
    return pattern_model.build_model(visible_j, initial_type_fractions=initial_type_fractions)


def build_starting_model(key, *, initial_type_fractions, initial_j=None):
    if initial_j is None:
        return random_model(key, initial_type_fractions=initial_type_fractions)

    initial_j = np.asarray(initial_j)
    if initial_j.shape != (6,):
        raise ValueError("--initial-j must provide six values: J11 J12 J13 J22 J23 J33.")
    return pattern_model.build_model(initial_j, initial_type_fractions=initial_type_fractions)


def make_lr_schedule(args):
    if args.lr_schedule == "constant" or args.final_learning_rate is None:
        return lambda step: args.learning_rate, optax.adam(args.learning_rate)
    if args.lr_schedule == "exponential":
        schedule = optax.exponential_decay(
            init_value=args.learning_rate,
            transition_steps=max(args.epochs, 1),
            decay_rate=args.final_learning_rate / args.learning_rate,
            staircase=False,
            end_value=args.final_learning_rate,
        )
        return schedule, optax.adam(schedule)
    if args.lr_schedule == "linear":
        schedule = optax.linear_schedule(args.learning_rate, args.final_learning_rate, transition_steps=max(args.epochs, 1))
        return schedule, optax.adam(schedule)
    raise ValueError(f"Unsupported learning-rate schedule: {args.lr_schedule}")


def counts_from_fractions(model, n_cells):
    fractions = onp.asarray(jax.device_get(pattern_model.visible_fractions_from_raw_fractions(model.raw_fractions)))
    min_per_type = pattern_model.min_cells_per_type(n_cells)
    min_total = 3 * min_per_type
    extras_float = fractions * (n_cells - min_total)
    extras = onp.floor(extras_float).astype(int)
    remainder = int(n_cells - min_total - extras.sum())
    for idx in onp.argsort(-(extras_float - extras))[:remainder]:
        extras[idx] += 1
    counts = extras + min_per_type
    return fractions, counts


def progress_entry(epoch, train_loss, val_loss, lr, model, n_cells):
    fractions, counts = counts_from_fractions(model, n_cells)
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "learning_rate": float(lr),
        "j": np.asarray(pattern_model.visible_j_from_raw_j(model.raw_j)).tolist(),
        "type_fractions": np.asarray(fractions).tolist(),
        "type_counts": np.asarray(counts).tolist(),
    }


def write_progress(path, rows):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp_path, path)


def resolve_outdir(outdir):
    outdir = os.path.expanduser(outdir)
    if os.path.isabs(outdir):
        return outdir

    notebook_prefix = os.path.join("results-natcompsci-2025", "concentric-ring-my-own")
    if outdir == notebook_prefix:
        outdir = "."
    elif outdir.startswith(notebook_prefix + os.sep):
        outdir = os.path.relpath(outdir, notebook_prefix)

    return os.path.abspath(outdir)


def InitialPatternFractionsReinforceLoss(
    cost_fn,
    *,
    n_sim_steps,
    n_episodes=1,
    n_val_episodes=0,
    lambda_l1=0.0,
    target_type_fractions=None,
    fraction_penalty_weight=0.0,
):
    n_sim_steps = int(n_sim_steps)
    n_episodes = int(n_episodes)
    n_val_episodes = int(n_val_episodes)
    lambda_l1 = float(lambda_l1)
    fraction_penalty_weight = float(fraction_penalty_weight)
    if target_type_fractions is not None:
        target_type_fractions = np.asarray(target_type_fractions)

    def _sim(model, base_istate, key):
        key_init, key_sim = jax.random.split(key)
        istate, logprob = pattern_model.sample_initial_celltypes(model, base_istate, key_init)
        trajectory = simulate(model, istate, key_sim, n_sim_steps, history=True)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]
        _istate = jtu.tree_map(lambda x: x[None, :, :], istate)
        trajectory = jtu.tree_map(lambda *v: np.concatenate(v), *[_istate, trajectory])
        return trajectory, logprob

    def _loss(model, istate, *, key, **kwargs):
        key, *subkeys = jax.random.split(key, n_episodes + 1)
        trajectory, logprob = jax.vmap(_sim, (None, None, 0))(model, istate, np.asarray(subkeys))
        episode_cost = jax.vmap(cost_fn)(trajectory).mean(-1)
        train_cost = episode_cost.mean()
        baseline = episode_cost.mean() if n_episodes > 1 else 0.0
        advantage = jax.lax.stop_gradient(episode_cost - baseline)
        loss = train_cost + np.mean(advantage * logprob)

        if fraction_penalty_weight > 0.0 and target_type_fractions is not None:
            fractions = pattern_model.visible_fractions_from_raw_fractions(model.raw_fractions)
            loss = loss + fraction_penalty_weight * np.sum((fractions - target_type_fractions) ** 2)

        if lambda_l1 > 0.0:
            reg = jax.tree_util.tree_map(lambda x: np.abs(x).sum(), eqx.filter(model, eqx.is_array))
            loss = loss + lambda_l1 * jax.tree_util.tree_reduce(lambda x, y: x + y, reg)

        if n_val_episodes > 0:
            key, *subkeys = jax.random.split(key, n_val_episodes + 1)
            val_trajectory, _ = jax.vmap(_sim, (None, None, 0))(model, istate, np.asarray(subkeys))
            val_cost = jax.vmap(cost_fn)(val_trajectory).mean(-1).mean()
            return loss, val_cost

        return loss

    return Loss(loss_fn=_loss, has_aux=(n_val_episodes > 0))


def train_with_log(model, istate, loss, *, key, epochs, optimizer, run_dir, learning_rate_fn, n_cells, early_stop_patience, early_stop_min_delta):
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
    progress = [progress_entry(0, rl, loss_aux, learning_rate_fn(0), model, n_cells)]
    write_progress(progress_path, progress)

    best_train_loss = float(rl)
    epochs_without_improvement = 0
    stopped_reason = None
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
        current_lr = float(learning_rate_fn(current_epoch))
        current_j = pattern_model.visible_j_from_raw_j(model.raw_j)
        current_fractions, current_counts = counts_from_fractions(model, n_cells)

        if float(rl) < best_train_loss - early_stop_min_delta:
            best_train_loss = float(rl)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress.append(progress_entry(current_epoch, rl, loss_aux, current_lr, model, n_cells))
        write_progress(progress_path, progress)
        if loss.has_aux:
            pbar.set_description(f"train={float(rl):.5f}, val={float(loss_aux):.5f}, lr={current_lr:.2e}, J={np.asarray(current_j).tolist()}, frac={np.asarray(current_fractions).tolist()}, n={np.asarray(current_counts).tolist()}")
        else:
            pbar.set_description(f"train={float(rl):.5f}, lr={current_lr:.2e}, J={np.asarray(current_j).tolist()}, frac={np.asarray(current_fractions).tolist()}, n={np.asarray(current_counts).tolist()}")

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_reason = f"early_stop: train_loss did not improve by at least {early_stop_min_delta} for {early_stop_patience} epochs"
            pbar.write(stopped_reason)
            break

    if progress:
        progress[-1]["stopped_reason"] = stopped_reason
        write_progress(progress_path, progress)

    return model, OptimizationResults(None, losses, val_losses, None)


def dump_j_parameters(path, init_model, trained_model):
    with open(path, "w") as f:
        json.dump(
            {
                "initial_j": np.asarray(pattern_model.visible_j_from_raw_j(init_model.raw_j)).tolist(),
                "trained_j": np.asarray(pattern_model.visible_j_from_raw_j(trained_model.raw_j)).tolist(),
                "initial_type_fractions": np.asarray(pattern_model.visible_fractions_from_raw_fractions(init_model.raw_fractions)).tolist(),
                "trained_type_fractions": np.asarray(pattern_model.visible_fractions_from_raw_fractions(trained_model.raw_fractions)).tolist(),
                "j_order": ["J11", "J12", "J13", "J22", "J23", "J33"],
            },
            f,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=("salt-pepper-shell", "bilobed-shell"), default="salt-pepper-shell")
    parser.add_argument("--n-opt-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--n-val-episodes", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--lr-schedule", choices=("constant", "exponential", "linear"), default="exponential")
    parser.add_argument("--final-learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--n-cells", type=int, default=60)
    parser.add_argument("--type-ratios", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--shell-distance-weight", type=float, default=1.0)
    parser.add_argument("--shell-target-radius", type=float, default=3.0)
    parser.add_argument("--type1-compactness-weight", type=float, default=1.0)
    parser.add_argument("--type2-compactness-weight", type=float, default=1.0)
    parser.add_argument("--type3-compactness-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--w11", type=float, default=1)
    parser.add_argument("--w12", type=float, default=0)
    parser.add_argument("--w13", type=float, default=0)
    parser.add_argument("--w22", type=float, default=1)
    parser.add_argument("--w23", type=float, default=0.0)
    parser.add_argument("--w33", type=float, default=1.0)
    parser.add_argument("--w-media1", type=float, default=1.0)
    parser.add_argument("--w-media2", type=float, default=1.0)
    parser.add_argument("--w-media3", type=float, default=1.0)
    parser.add_argument("--w-centroid-distance", type=float, default=1.0)
    parser.add_argument("--w-frac", type=float, default=0.0)
    parser.add_argument("--initial-j", type=float, nargs=6, default=None, metavar=("J11", "J12", "J13", "J22", "J23", "J33"))
    parser.add_argument("--outdir", type=str, default="./trained_models_patterns/")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    key = jxm.utils.generate_random_key()
    key, init_key = jax.random.split(key)
    initial_type_counts, initial_type_fractions = pattern_model.counts_from_ratios(args.n_cells, args.type_ratios)
    if args.type3_compactness_weight is not None:
        args.type2_compactness_weight = args.type3_compactness_weight

    istate = pattern_model.build_istate(init_key, n_cells=args.n_cells, type_ratios=args.type_ratios)
    cost_fn = pattern_model.pattern_trajectory_loss(
        pattern=args.pattern,
        shell_distance_weight=args.shell_distance_weight,
        shell_target_radius=args.shell_target_radius,
        type1_compactness_weight=args.type1_compactness_weight,
        type2_compactness_weight=args.type2_compactness_weight,
        w11=args.w11,
        w12=args.w12,
        w13=args.w13,
        w22=args.w22,
        w23=args.w23,
        w33=args.w33,
        w_media1=args.w_media1,
        w_media2=args.w_media2,
        w_media3=args.w_media3,
        w_centroid_distance=args.w_centroid_distance,
    )
    loss = InitialPatternFractionsReinforceLoss(
        cost_fn,
        n_sim_steps=args.n_steps,
        n_episodes=args.n_episodes,
        n_val_episodes=args.n_val_episodes,
        lambda_l1=0.0,
        target_type_fractions=initial_type_fractions,
        fraction_penalty_weight=args.w_frac,
    )
    learning_rate_schedule, optimizer = make_lr_schedule(args)

    root_dir = resolve_outdir(args.outdir)
    if args.clean:
        try:
            shutil.rmtree(root_dir)
        except FileNotFoundError:
            pass
    os.makedirs(root_dir, exist_ok=True)

    hyperparams = vars(args).copy()
    hyperparams["resolved_outdir"] = root_dir
    hyperparams["initial_type_counts"] = onp.asarray(initial_type_counts).tolist()
    hyperparams["initial_type_fractions"] = onp.asarray(initial_type_fractions).tolist()
    with open(os.path.join(root_dir, "train-pattern-opt-hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    for i in range(args.n_opt_runs):
        key, model_key, train_key = jax.random.split(key, 3)
        model = build_starting_model(model_key, initial_type_fractions=initial_type_fractions, initial_j=args.initial_j)
        run_dir = os.path.join(root_dir, f"train-pattern-{i}")
        os.makedirs(run_dir, exist_ok=True)
        opt_model, opt_results = train_with_log(
            model,
            istate,
            loss,
            key=train_key,
            epochs=args.epochs,
            optimizer=optimizer,
            run_dir=run_dir,
            learning_rate_fn=learning_rate_schedule,
            n_cells=args.n_cells,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
        )
        eqx.tree_serialise_leaves(os.path.join(run_dir, f"init-pattern-{i}.eqx"), model)
        eqx.tree_serialise_leaves(os.path.join(run_dir, f"trained-pattern-{i}.eqx"), opt_model)
        eqx.tree_serialise_leaves(os.path.join(run_dir, f"results-pattern-{i}.eqx"), opt_results)
        dump_j_parameters(os.path.join(run_dir, "j_parameters.json"), model, opt_model)
