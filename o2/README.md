# Running JAX-Morph Training on O2

These commands assume the HMS O2 login host is `o2.hms.harvard.edu` and that O2 runs jobs through Slurm. The concentric-ring job below uses the CPU `short` partition; the older Figure 2 script uses a GPU partition.

## 1. Sync the repo from your Mac

Replace `O2_USER` with your O2 username.

```bash
rsync -avh --progress \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  /Users/mw/Desktop/jax-morph/ \
  O2_USER@o2.hms.harvard.edu:~/jax-morph/
```

For later updates, rerun the same `rsync` command.

## 2. Create the environment on O2

```bash
ssh O2_USER@o2.hms.harvard.edu
cd ~/jax-morph

conda env create -f environment.yml
conda activate jax-morph
python -m pip install -e .
```

If `conda` is not available immediately, run `module avail conda`, `module avail miniconda`, or use the O2-provided conda setup for your account.

## 3. Submit the training job

### Concentric ring training

Run a short CPU sanity job first:

```bash
cd ~/jax-morph
mkdir -p logs
TRAIN_ARGS="--epochs 5 --n-opt-runs 1 --n-episodes 1 --n-val-episodes 1 --n-steps 2 --clean" \
  sbatch o2/concentric_train.sbatch
```

Then run the full default CPU training:

```bash
cd ~/jax-morph
mkdir -p logs
sbatch o2/concentric_train.sbatch
```

The full job runs:

```bash
cd results-natcompsci-2025/concentric-ring-my-own
python train.py --clean
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/jxm_concentric_train_<JOBID>.out
tail -f logs/jxm_concentric_train_<JOBID>.err
```

### Figure 2 training

```bash
cd ~/jax-morph
mkdir -p logs
sbatch o2/fig2_train.sbatch
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/jxm_fig2_train_<JOBID>.out
tail -f logs/jxm_fig2_train_<JOBID>.err
```

Cancel it if needed:

```bash
scancel <JOBID>
```

## 4. Copy results back

For the concentric ring model:

```bash
rsync -avh --progress \
  O2_USER@o2.hms.harvard.edu:~/jax-morph/results-natcompsci-2025/concentric-ring-my-own/trained_models/ \
  /Users/mw/Desktop/jax-morph/results-natcompsci-2025/concentric-ring-my-own/trained_models/
```

The script runs `results-natcompsci-2025/figure_2/fig2_train.py`, which creates a new `run_<timestamp>/` folder inside `results-natcompsci-2025/figure_2`.

```bash
rsync -avh --progress \
  O2_USER@o2.hms.harvard.edu:~/jax-morph/results-natcompsci-2025/figure_2/run_*/ \
  /Users/mw/Desktop/jax-morph/results-natcompsci-2025/figure_2/
```

## Notes

- If O2 says you need an account for jobs, run `sshare -U -u "$USER"` on O2 and add the account to the relevant sbatch script, e.g. `o2/concentric_train.sbatch` or `o2/fig2_train.sbatch`.
- For `o2/concentric_train.sbatch`, `jax.devices()` should print CPU devices because the script sets `JAX_PLATFORMS=cpu`.
- For `o2/fig2_train.sbatch`, if `jax.devices()` prints only CPU devices, the CUDA module or `jax[cuda12]` install is not lined up yet.
- The script writes Matplotlib and JAX caches under `.cache/` in the project to avoid home-directory cache permission issues.
