# Running Figure 2 Training on O2

These commands assume the HMS O2 login host is `o2.hms.harvard.edu` and that O2 runs jobs through Slurm. O2's docs describe login nodes behind that host and Slurm submission with `sbatch`; GPU jobs use GPU partitions and `--gres=gpu:N`.

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

module load gcc/9.2.0 || true
module load cuda/12.4 || module load cuda/12.2 || module load cuda/12.1 || true

conda env create -f environment.yml
conda activate jax-morph
python -m pip install -e .
```

If `conda` is not available immediately, run `module avail conda`, `module avail miniconda`, or use the O2-provided conda setup for your account.

## 3. Submit the training job

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

The script runs `results-natcompsci-2025/figure_2/fig2_train.py`, which creates a new `run_<timestamp>/` folder inside `results-natcompsci-2025/figure_2`.

```bash
rsync -avh --progress \
  O2_USER@o2.hms.harvard.edu:~/jax-morph/results-natcompsci-2025/figure_2/run_*/ \
  /Users/mw/Desktop/jax-morph/results-natcompsci-2025/figure_2/
```

## Notes

- If O2 says you need an account for GPU jobs, run `sshare -U -u "$USER"` on O2 and add the account to `o2/fig2_train.sbatch`.
- If `jax.devices()` prints only CPU devices, the CUDA module or `jax[cuda12]` install is not lined up yet.
- The script writes Matplotlib and JAX caches under `.cache/` in the project to avoid home-directory cache permission issues.
