# template from opt/_old/training.py, modified to include recoverable logging and early stopping
import json
import os
from collections import namedtuple

import equinox as eqx
import jax
import jax.numpy as np
import jax.tree_util as jtu
from tqdm import trange

import istate_and_model
from jax_morph.simulation import simulate

# loss_aux = validation loss
Loss = namedtuple("Loss", ["loss_fn", "has_aux"])
OptimizationResults = namedtuple("OptimizationResults", ["model", "loss", "loss_aux", "grad"])


def _json_float(value):
    return None if value is None else float(value)


def write_progress_log(path, entries):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp_path, path)


def cell_counts_from_fraction(model, n_cells):
    fraction = float(istate_and_model.visible_fraction_from_raw_fraction(model.raw_fraction))
    n_type_1 = int(np.clip(round(n_cells * fraction), 1, n_cells - 1))
    return fraction, n_type_1, n_cells - n_type_1


def progress_entry(epoch, train_loss, val_loss, learning_rate, model, n_cells):
    fraction, n_type_1, n_type_2 = cell_counts_from_fraction(model, n_cells)
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": _json_float(val_loss),
        "learning_rate": float(learning_rate),
        "j": np.asarray(istate_and_model.visible_j_from_raw_j(model.raw_j)).tolist(),
        "type_1_fraction": fraction,
        "n_type_1": n_type_1,
        "n_type_2": n_type_2,
    }


def InitialFractionReinforceLoss(cost_fn, *, n_sim_steps, n_episodes=1, n_val_episodes=0, lambda_l1=0.0):
    n_sim_steps = int(n_sim_steps)
    n_episodes = int(n_episodes)
    n_val_episodes = int(n_val_episodes)
    lambda_l1 = float(lambda_l1)

    def _sim(model, base_istate, key):
        key_init, key_sim = jax.random.split(key)
        # sample initial cell types according to the model's current fraction parameter, and get the log probability of that sample for the REINFORCE loss
        istate, logprob = istate_and_model.sample_initial_celltypes(model, base_istate, key_init)
        # simulate forward and get trajectory
        trajectory = simulate(model, istate, key_sim, n_sim_steps, history=True)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]
        # prepend initial state to trajectory and add batch dimension for episodes, so shape goes from (n_steps, ...) to (1 + n_steps, ...)
        _istate = jtu.tree_map(lambda x: x[None, :, :], istate)
        trajectory = jtu.tree_map(lambda *v: np.concatenate(v), *[_istate, trajectory])
        return trajectory, logprob

    def _loss(model, istate, *, key, n_sim_steps=n_sim_steps, n_val_episodes=n_val_episodes, **kwargs):
        key, *subkeys = jax.random.split(key, n_episodes + 1)
        # simulate forward for n_episodes and get trajectories and log probabilities of the initial cell type samples
        trajectory, logprob = jax.vmap(_sim, (None, None, 0))(model, istate, np.asarray(subkeys))
        # compute cost for each episode and average over episodes for the training loss
        episode_cost = jax.vmap(cost_fn)(trajectory).mean(-1)
        train_cost = episode_cost.mean()

        # subtract baseline to reduce variance
        baseline = episode_cost.mean() if n_episodes > 1 else 0.0
        # avoid doing gradient through baseline by treating it as a constant
        # this is sampling so avoid backprop through the cost as well
        # since it's an estimate of the expected cost under the model's current parameters
        advantage = jax.lax.stop_gradient(episode_cost - baseline)
        loss = train_cost + np.mean(advantage * logprob)

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


def train_with_recoverable_log(model, istate, loss, *, key, epochs, optimizer, run_dir, learning_rate_fn, n_cells, early_stop_patience, early_stop_min_delta):
    # retrieve training loss and gradient
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

    progress = [progress_entry(0, rl, loss_aux, learning_rate_fn(0), model, n_cells)]
    write_progress_log(progress_path, progress)

    # initialize Adam optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    pbar = trange(epochs, dynamic_ncols=True)

    for epoch in pbar:
        # update model parameters using Adam
        updates, opt_state = optimizer.update(grads, opt_state, model)
        # apply updates to model
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
        current_j = istate_and_model.visible_j_from_raw_j(model.raw_j)
        current_frac, current_n_type_1, current_n_type_2 = cell_counts_from_fraction(model, n_cells)
        current_lr = float(learning_rate_fn(current_epoch))

        if float(rl) < best_train_loss - early_stop_min_delta:
            best_train_loss = float(rl)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        progress.append(progress_entry(current_epoch, rl, loss_aux, current_lr, model, n_cells))
        write_progress_log(progress_path, progress)

        if loss.has_aux:
            pbar.set_description(f"train={float(rl):.5f}, val={float(loss_aux):.5f}, lr={current_lr:.2e}, J={np.asarray(current_j).tolist()}, frac={current_frac:.3f}, n1={current_n_type_1}, n2={current_n_type_2}")
        else:
            pbar.set_description(f"train={float(rl):.5f}, lr={current_lr:.2e}, J={np.asarray(current_j).tolist()}, frac={current_frac:.3f}, n1={current_n_type_1}, n2={current_n_type_2}")

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_reason = f"early_stop: train_loss did not improve by at least {early_stop_min_delta} for {early_stop_patience} epochs"
            pbar.write(stopped_reason)
            break

    if progress:
        progress[-1]["stopped_reason"] = stopped_reason
        write_progress_log(progress_path, progress)

    return model, OptimizationResults(None, losses, val_losses, None)
