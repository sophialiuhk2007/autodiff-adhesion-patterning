# JAX Morph: Salt-and-Pepper Core With Envelope

This repository gives the training script and model used for learning relative energy differences (**J-matrix**) for a salt-and-pepper/intermixed core of cell
types 1 and 2 surrounded by a type-3 envelope. Here, we define a loss function that hopes to captures factors specific to this configuration. This approach of defining a bespoke loss function is an older approach taken in the early stages of our project, which aims to learn the **J-matrix** for configurations with >2 cell types.

The experiment code and notebook live in:

```text
experiments/intermixed_envelope/
```

Important files:

- `train_logged_patterns.py`: trains the model and writes logs/models.
- `pattern_istate_and_model.py`: defines the initial state, trainable `J`
  model, cell-type sampling, and loss.
- `visualizations.ipynb`: loads the trained run and generates visualizations.
- `trained_models/`: an example trained model.

## Installation

Clone the repo and enter it:

```bash
git clone https://github.com/sophialiuhk2007/summer_research_2026.git jax-morph
cd jax-morph
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the package:

```bash
python -m pip install -e .
```

Install JupyterLab for the visualization notebook:

```bash
python -m pip install jupyterlab
```

Check the environment:

```bash
python -c "import jax, jax_md, equinox, diffrax, optax, jax_morph; print('jax-morph environment OK')"
```

## Quick Start

Go to the experiment directory:

```bash
cd experiments/intermixed_envelope
```

The bundled trained run is already here:

```text
trained_models/
  train-pattern-opt-hyperparams.json
  train-pattern-0/
    training_progress.json
    init-pattern-0.eqx
    trained-pattern-0.eqx
    results-pattern-0.eqx
    j_parameters.json
```

This trained run uses:

```text
n_cells = 60
type_ratios = 1 1 4
w_frac = 5
w12 = 1
w33 = 1
w_media1 = w_media2 = w_media3 = 10
```

The learned `J` order is:

```text
J11, J12, J13, J22, J23, J33
```

## Use the Visualization Notebook

Start JupyterLab from the experiment directory:

```bash
cd experiments/intermixed_envelope
jupyter lab
```

Open:

```text
visualizations.ipynb
```

For the trained run section, use:

```python
RUN_ID = 0
opt_hyper, training_log, istate, initial_model, trained_model = load_run_data_patterns(RUN_ID)
```

The loader should default to:

```python
root="trained_models"
```

To change the directory, modify the root value as shown:

```python
def load_run_data_patterns(run_id, root="trained_models"):
```

Then run the notebook cells from top to bottom. The notebook reads the trained
model from `trained_models/train-pattern-0/` and writes visualization outputs
to folders such as:

```text
pattern_forward_outputs/
trained_pattern_forward_outputs/
```

## Manual Forward Simulation in the Notebook

The notebook also has an early manual simulation cell with variables like `J`,
`N_CELLS`, `TYPE_RATIOS`, `DEFAULT_LOSS_KWARGS`, and `N_STEPS`.

For this repo, use:

```python
N_CELLS = 60
TYPE_RATIOS = np.array([1, 1, 4])
DEFAULT_LOSS_KWARGS = dict(
    shell_distance_weight=1.0,
    w12=1.0,
    w33=1.0,
    w_media1=10.0,
    w_media2=10.0,
    w_media3=10.0,
)
```

To use the bundled trained `J`, read it from:

```text
trained_models/train-pattern-0/j_parameters.json
```

For `RUN_ID = 0`, the trained `J` is:

```python
J = np.array([
    3.2010049413847446,
    3.6702796313897172,
    0.9112921368503344,
    3.4763464893202896,
    0.8043657292249345,
    0.8135481326069253,
])
```

## Rerun Training

To regenerate the trained run used by the notebook:

```bash
cd experiments/intermixed_envelope

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
  --outdir ./trained_models/ --clean
```

`--clean` deletes the existing `trained_models` directory before writing the new
run. Remove `--clean` if you want to keep the bundled run.

The useful output files are:

- `training_progress.json`: loss, learning rate, `J`, and type-count trace.
- `j_parameters.json`: initial and trained `J`, plus trained type fractions.
- `trained-pattern-*.eqx`: serialized trained model.
