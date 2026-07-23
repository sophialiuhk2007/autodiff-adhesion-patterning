# Reproducing the Summer Write-up Results

## Table of Contents

1. [Overview](#overview)
2. [Repository and Branch Layout](#repository-and-branch-layout)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Task 0: Fixed-Ratio Concentric Shells](#task-0-fixed-ratio-concentric-shells)
6. [Task 1: Learnable Cell-Type Ratio](#task-1-learnable-cell-type-ratio)
7. [Task 2: Salt-and-Pepper Core With Envelope](#task-2-salt-and-pepper-core-with-envelope)
8. [Task 3: Learn `J` With MMD](#task-3-learn-j-with-mmd)
9. [Task 4: Learn `C` Without MMD](#task-4-learn-c-without-mmd)
10. [Task 5: Learn `C` With MMD](#task-5-learn-c-with-mmd)
11. [Output File Structure](#output-file-structure)
12. [Practical Notes](#practical-notes)

## Overview

This fork builds on `jax-morph` to reproduce adhesion-only multicellular
patterning results. The central objects are:

- `J`: a symmetric cell-type interaction matrix. Larger entries correspond to
  stronger Morse-potential adhesion between the corresponding cell types.
- `C`: a cadherin-composition matrix. When learning composition, the effective
  interaction matrix is `J = C J' C^T`, where `J'` is a cadherin-cadherin
  empirical or synthetic matrix.
- Target `.npz` states: saved point clouds of typed cells used by the MMD loss.

There are two broad result families:

- **Learning `J` directly.** Early tasks use hand-designed losses; later tasks
  use MMD + Kabsch alignment against target point clouds.
- **Learning `C`.** Composition tasks learn cadherin choices that induce a
  useful `J` through `C J' C^T`, either with an older configuration-specific
  loss or with the MMD stack.

Every result follows the same workflow:

1. Check out the correct branch.
2. Create one shared Python environment for the project, with the repo installed
   in editable mode.
3. Run the branch-specific training or forward-simulation command.
4. Inspect the saved output directory.
5. Run the matching visualization notebook or helper script.

## Repository and Branch Layout

Across these branches, most experiment code lives under
`results-natcompsci-2025/concentric-ring-my-own`:

```text
results-natcompsci-2025/
  concentric-ring-my-own/
    *istate_and_model*.py           # Initial states, trainable parameters, models, and losses.
    train*.py                       # Command-line training scripts.
    fig4_visualizations*.ipynb      # Forward simulations and figure-generation notebooks.
    plot_training_progress*.ipynb   # Loss curves, learned parameters, and optimization diagnostics.
    target_state_designer_server.py # Browser tool for designing and exporting target `.npz` files.
    target_state_modules/           # Reusable modules for designing target states.
    mmd_target_states*/             # Saved target point clouds used by MMD-based runs.
    trained_models*/                # Training outputs: parameters, logs, checkpoints, and artifacts.
```

Branch map:

| Branch             | Reproduces                      | Main entry points                                                         |
| ------------------ | ------------------------------- | ------------------------------------------------------------------------- |
| `my-notebook-work` | fixed-ratio concentric shells   | `istate_and_model.py`, `train_logged.py`                                  |
| `task_1`           | learnable cell-type ratio       | `istate_and_model.py`, `training_log_helpers.py`, `train_logged.py`       |
| `task_2_s&p`       | salt-and-pepper core + envelope | `pattern_istate_and_model.py`, `train_logged_patterns.py`                 |
| `mmd_expand`       | MMD learning of `J`             | `pattern_istate_and_model_chunk.py`, `train_logged_patterns_resumable.py` |
| `play-composition` | learning `C` without MMD        | `anneal_istate_and_model.py`, `train_anneal.py`                           |
| `composition-mmd`  | learning `C` with MMD           | `pattern_istate_and_model_chunk.py`, `train_logged_patterns_resumable.py` |

Branch names with special characters should be quoted in shell commands. For
example, use `git switch "task_2_s&p"`.

## Installation

Clone the repository:

```bash
git clone https://github.com/sophialiuhk2007/summer_research_2026.git jax-morph
cd jax-morph
git fetch --all
```

Create one virtual environment in the repo root. All branches listed in this
guide use the same package dependencies, so you only need to do this once.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install jupyterlab
```

The editable install (`-e .`) installs the local `jax_morph` package and the
main training dependencies. JupyterLab is installed separately for the
visualization notebooks.

- JAX
- JAX-MD from GitHub
- Equinox
- Diffrax
- Optax
- Matplotlib
- tqdm
- NetworkX

Check the installation:

```bash
python -c "import jax, jax_md, equinox, diffrax, optax, jax_morph; print('jax-morph environment OK')"
```

Most experiment scripts should be run from the experiment directory:

```bash
cd results-natcompsci-2025/concentric-ring-my-own
```

## Quick Start

This is the shortest path to a full MMD training run for the bilobed + envelope
target:

```bash
git switch mmd_expand
cd results-natcompsci-2025/concentric-ring-my-own

python train_logged_patterns_resumable.py \
  --target-state-path ./mmd_target_states_42cells/mmd-target-selected-000.npz \
  --n-opt-runs 2 --epochs 50 \
  --n-episodes 4 --n-val-episodes 0 --n-steps 1 \
  --relaxation-total-steps 1000 --relaxation-blocks 10 \
  --mmd-sigma 1.0 --mmd-chunk-size 512 --mmd-final-only \
  --dtype float32 \
  --learning-rate 0.1 --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --outdir ./quickstart_mmd_run/ \
  --checkpoint-every 1 --artifact-every 10 --artifact-simulations 2 \
  --clean
```

This quick-start command is intentionally small. It checks environment setup,
target loading, loss evaluation, checkpointing, and artifact saving. Use the
full task commands below for the results in `main.tex`.

To make a new MMD target with the designer:

```bash
git switch mmd_expand
cd results-natcompsci-2025/concentric-ring-my-own
python target_state_designer_server.py
```

Open the printed local URL, design and export the target `.npz`, and pass that
file to `--target-state-path`.

## Task 0: Fixed-Ratio Concentric Shells

Branch: `my-notebook-work`

Write-up result: `fig:trained-concentric-2`.

This is the first hand-designed concentric-shell training run. It learns
`[J11, J12, J22]` for `10` type-1 and `30` type-2 cells.

```bash
git switch my-notebook-work
cd results-natcompsci-2025/concentric-ring-my-own

python train_logged.py \
  --n-opt-runs 10 --epochs 1000 \
  --n-episodes 4 --n-val-episodes 4 --n-steps 20 \
  --n-type-1 10 --n-type-2 30 \
  --target-r1 1.0 --target-r2 2.0 \
  --learning-rate 0.1 \
  --outdir ./trained_models/ --clean
```

Visualization notebooks:

```bash
jupyter lab fig4_visualizations.ipynb
jupyter lab plot_training_progress.ipynb
```

Expected output:

```text
trained_models/
  train-concentric-*/
    training_progress.json
    j_parameters.json
    trained-concentric-*.eqx
```

## Task 1: Learnable Cell-Type Ratio

Branch: `task_1`

Write-up results: `fig:learnable-ratio-concentric`,
`fig:multiple-training-concentric-ratio`.

This learns both `J` and the type-1 fraction using the REINFORCE-style fraction
estimator.

Using `n-episodes = 10`, `n-steps = 20`, `epochs = 200`,
`--learning-rate 0.1`, `--final-learning-rate 0.001`, and
`--lr-schedule exponential`:

```bash
git switch task_1
cd results-natcompsci-2025/concentric-ring-my-own

python train_logged.py \
  --n-opt-runs 10 --epochs 200 \
  --n-episodes 10 --n-val-episodes 0 --n-steps 20 \
  --n-cells 60 --initial-type-1-fraction 0.5 \
  --target-r1 1.0 --target-r2 2.0 \
  --type-1-self-contact-weight 0.1 \
  --cross-contact-weight 0.2 \
  --contact-distance 1.0 --contact-sharpness 20.0 \
  --learning-rate 0.1 --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --outdir ./trained_models/ --clean
```

Visualization notebooks:

```bash
jupyter lab fig4_visualizations.ipynb
jupyter lab plot_training_progress.ipynb
```

Expected output:

```text
trained_models/train-concentric-*/
  training_progress.json
  j_parameters.json   # includes trained_j and trained_type_1_fraction
```

## Task 2: Salt-and-Pepper Core With Envelope

Branch: `task_2_s&p`

Write-up results: `fig:combined-concentric-shell-intermixed`,
`fig:subfigure-j-matrix-concentric-intermixed`, `fig:random-concentric-intermixed`.

This learns a 3-type `J` for a salt-and-pepper/intermixed core of types 1 and 2
with a type-3 envelope. The write-up uses `n-episodes = 10`, `n-steps = 20`,
`epochs = 200`, `--learning-rate 0.1`, `--final-learning-rate 0.001`, and
`--lr-schedule exponential`.

The loss weights and cell-type-ratio settings are `w_frac = 5`, `w12 = 1`, and
`type-ratios = [1 1 4]`.

```bash
git switch "task_2_s&p"
cd results-natcompsci-2025/concentric-ring-my-own

python train_logged_patterns.py \
  --pattern salt-pepper-shell \
  --n-opt-runs 5 --epochs 200 \
  --n-episodes 10 --n-val-episodes 0 --n-steps 20 \
  --n-cells 60 --type-ratios 1 1 4 \
  --shell-distance-weight 1.0 \
  --w12 1.0 --w33 1.0 \
  --w-media1 10.0 --w-media2 10.0 --w-media3 10.0 \
  --w-frac 5.0 \
  --learning-rate 0.1 --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --outdir ./trained_models_patterns_salt_pepper/ --clean
```

Expected training output:

```text
trained_models_patterns_salt_pepper/
  train-pattern-opt-hyperparams.json
  train-pattern-*/
    training_progress.json
    j_parameters.json
```

Visualization notebook for the trained salt-and-pepper/envelope run:

```bash
jupyter lab fig4_visualizations_patterns.ipynb
```

To generate the figure testing whether the type-1/type-2 core is consistent
with random mixing, run the cross-contact null notebook:

```bash
jupyter lab fig4_visualizations_patterns_cross_contact_null.ipynb
```

Before running all cells, set the notebook variable `J` to the learned `trained_j`
from the selected run's `j_parameters.json`, and keep the cell-count setting
consistent with the training run (`N_CELLS = 60`, `TYPE_RATIOS = np.array([1,
1, 4])`). The notebook runs replicate forward simulations, compares the
observed type-1/type-2 core cross-contact fraction against random relabelings
of the same final geometry, and saves:

```text
pattern_cross_contact_null_outputs/
  cross_contact_null_summary.png
  cross_contact_null_summary.json
```

## Task 3: Learn `J` With MMD

Branch: `mmd_expand`

Write-up results: `fig:change-num-cells`, `fig:morse-potential-graph`,
`fig:training-runs-bilobed-envelope`, `fig:recapitulation-three-cell-type`, and
the modular `J`-matrix rule simulations in `sec:rules`.

This learns `J` directly from a saved typed target point cloud.

```bash
git switch mmd_expand
cd results-natcompsci-2025/concentric-ring-my-own

python train_logged_patterns_resumable.py \
  --target-state-path ./mmd_target_states_42cells/mmd-target-selected-000.npz \
  --n-opt-runs 10 --epochs 1000 \
  --n-episodes 10 --n-val-episodes 0 --n-steps 1 \
  --relaxation-total-steps 5000 --relaxation-blocks 20 \
  --mmd-sigma 1.0 --mmd-chunk-size 512 --mmd-final-only \
  --dtype float32 \
  --learning-rate 0.1 --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --outdir ./trained_models_patterns_bilobed_envelope_mmd/ \
  --checkpoint-every 1 --artifact-every 50 --artifact-simulations 4 \
  --clean
```

Resume:

```bash
python train_logged_patterns_resumable.py \
  --target-state-path ./mmd_target_states_42cells/mmd-target-selected-000.npz \
  --outdir ./trained_models_patterns_bilobed_envelope_mmd/ \
  --resume
```

Use other targets by replacing `--target-state-path` with one of these saved
target `.npz` files, or with any target `.npz` exported from
`target_state_designer_server.py`:

```text
mmd_target_states/
mmd_target_states_42cells/
mmd_target_states_54cells/
mmd_target_states_60/
mmd_target_states_trilobed_30cells/
```

Visualization notebooks:

```bash
# Forward simulation notebook for an inspected J and cell-count setting.
# Set J/TYPE_RATIOS, run relaxation, save initial/final state panels,
# hidden-type panels, relaxation snapshots, and MP4 movies in pattern_forward_outputs/.
jupyter lab fig4_visualizations_patterns.ipynb

# Forward simulation notebook with Morse-energy inspection.
# It runs the same kind of relaxation/movie workflow, but also saves
# Morse-energy traces and per-frame .npz files for checking how the potential
# changes during relaxation.
jupyter lab fig4_visualizations_patterns_morse_potential.ipynb

# Training diagnostics: loss curves, learned J values, and optimization progress.
jupyter lab plot_training_progress_patterns.ipynb
```

Expected output:

```text
trained_models_patterns_bilobed_envelope_mmd/
  train-pattern-*/
    training_progress.json
    j_parameters.json
    training_checkpoint.eqx
    artifacts/
```

## Task 4: Learn `C` Without MMD

Branch: `play-composition`

Write-up result: the synthetic-test-matrix composition runs using the older
plain loss in `sec:results-C`.

This learns `C` using the older bilobed/shell trajectory losses rather than MMD.

```bash
git switch play-composition
cd results-natcompsci-2025/concentric-ring-my-own

python train_anneal.py \
  --j-matrix-csv ./test_j_matrices/J_matrix_23x23_Test.csv \
  --cadherins 1 2 3 \
  --pattern bilobed \
  --n-opt-runs 10 --epochs 100 \
  --n-episodes 10 --n-val-episodes 0 --n-steps 1 \
  --n-type-1 10 --n-type-2 10 \
  --learning-rate 0.01 --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --kT-start 1.0 --kT-end 0.1 \
  --relaxation-total-steps 5000 --relaxation-blocks 20 \
  --outdir ./trained_models_anneal/ --clean
```

Shell target variant. This command relies on the script defaults for options
not listed here:

```bash
python train_anneal.py \
  --j-matrix-csv ./test_j_matrices/J_matrix_23x23_Test.csv \
  --cadherins 1 2 3 \
  --pattern shell \
  --n-type-1 10 --n-type-2 30 \
  --type-1-radius 1.0 --type-2-radius 2.0 \
  --outdir ./trained_models_anneal_shell/ --clean
```

Visualization notebooks:

```bash
jupyter lab fig4_visualizations.ipynb
jupyter lab fig4_visualizations_morse_potential.ipynb
jupyter lab plot_training_progress.ipynb
```

Expected output:

```text
trained_models_anneal/
  train-anneal-*/
    training_progress.json
    j_parameters.json   # includes C, top cadherins, reduced J
```

## Task 5: Learn `C` With MMD

Branch: `composition-mmd`

Write-up results: the MMD composition runs in `sec:results-C`, including
synthetic matrix tests and Sean's empirical `23 x 23` matrix.

This learns cadherin composition `C` while evaluating the simulated structure
against an MMD target `.npz`.

Synthetic test matrix:

```bash
git switch composition-mmd
cd results-natcompsci-2025/concentric-ring-my-own

python train_logged_patterns_resumable.py \
  --target-state-path ./mmd_target_states_bilobed_envelope/mmd-target-selected-000.npz \
  --j-matrix-csv ./test_j_matrices/J_matrix_23x23_Test.csv \
  --cadherins 1 2 3 \
  --pattern bilobed-shell \
  --n-opt-runs 10 --epochs 1000 \
  --n-episodes 20 --n-val-episodes 0 --n-steps 1 \
  --relaxation-total-steps 5000 --relaxation-blocks 20 \
  --mmd-sigma 1.0 --mmd-chunk-size 512 --mmd-final-only \
  --composition-temperature 1.0 \
  --final-composition-temperature 0.1 \
  --composition-temperature-schedule linear \
  --reinforce --distinct-cadherins \
  --cadherin-overlap-lambda 0.1 \
  --learning-rate 0.5 --final-learning-rate 0.05 \
  --lr-schedule exponential \
  --outdir ./trained_models_patterns_composition_mmd_test/ \
  --checkpoint-every 1 --artifact-every 50 --artifact-simulations 4 \
  --clean
```

Empirical `23 x 23` matrix:

```bash
python train_logged_patterns_resumable.py \
  --target-state-path ./mmd_target_states/concentric-60.npz \
  --j-matrix-csv ./test_j_matrices/J_matrix_23x23_4C_normal.csv \
  --pattern bilobed-shell \
  --n-opt-runs 10 --epochs 1000 \
  --n-episodes 20 --n-val-episodes 0 --n-steps 1 \
  --relaxation-total-steps 5000 --relaxation-blocks 20 \
  --mmd-sigma 1.0 --mmd-chunk-size 512 --mmd-final-only \
  --composition-temperature 1.0 \
  --final-composition-temperature 0.1 \
  --composition-temperature-schedule linear \
  --reinforce --cadherin-overlap-lambda 0.1 \
  --learning-rate 0.5 --final-learning-rate 0.05 \
  --lr-schedule exponential \
  --outdir ./trained_models_patterns_composition_mmd_empirical/ \
  --checkpoint-every 1 --artifact-every 50 --artifact-simulations 4 \
  --clean
```

Bilobed + envelope restricted to `CDH1`, `CDH2`, `CDH6`:

```bash
python train_logged_patterns_resumable.py \
  --target-state-path ./mmd_target_states_bilobed_envelope/mmd-target-selected-000.npz \
  --j-matrix-csv ./test_j_matrices/J_matrix_23x23_4C_normal.csv \
  --cadherins 1 2 6 \
  --reinforce --distinct-cadherins \
  --outdir ./trained_models_patterns_composition_mmd_cdh126/ \
  --clean
```

Visualization notebooks:

```bash
jupyter lab fig4_visualizations_patterns_morse_potential_from_c.ipynb
jupyter lab fig4_visualizations_patterns.ipynb
jupyter lab plot_training_progress.ipynb
```

To regenerate predictions with a different empirical cadherin-cadherin matrix,
replace `--j-matrix-csv` and rerun.

Expected output:

```text
trained_models_patterns_composition_mmd_*/
  train-pattern-*/
    training_progress.json
    j_parameters.json   # C, top cadherins, reduced J, Morse-rescaled J
    artifacts/
```

## Output File Structure

Training scripts create one output directory per experiment and one subdirectory
per independent optimization run.

Typical direct-`J` output:

```text
trained_models_patterns_bilobed_envelope_mmd/
  train-pattern-opt-hyperparams.json
  train-pattern-0/
    init-pattern-0.eqx
    trained-pattern-0.eqx
    results-pattern-0.eqx
    j_parameters.json
    training_progress.json
    training_checkpoint.eqx
    loss_curve.png
    artifacts/
      epoch_0050-final-state.npz
      epoch_0050-forward-snapshots.npz
```

Typical composition output adds:

```text
j_parameters.json
  initial_c
  trained_c
  initial_reduced_j
  trained_reduced_j
  initial_j / trained_j
  top_cadherins
  cadherin_labels
```

Useful files:

- `training_progress.json`: per-epoch loss, learning rate, current `J`, and,
  on composition branches, current `C` and top cadherins.
- `j_parameters.json`: initial/trained parameters in a compact summary.
- `*-final-state.npz`: final simulated state used for static visualizations.
- `*-forward-snapshots.npz`: trajectory snapshots used for movies and traces.
- `.eqx` files: serialized Equinox models/results.

## Practical Notes

- For larger targets, prefer `float32`, chunked MMD distances, and resumable
  checkpoints.
- If a run is interrupted, rerun the same command with the same `--outdir` plus
  `--resume`.
- Most scripts write outputs relative to
  `results-natcompsci-2025/concentric-ring-my-own`, so keep that as the working
  directory.
