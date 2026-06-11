# Adhesion-Only Forward Sanity Runs

This folder contains the adhesion-only forward-model sanity runs used for the
Week 2-3 target checks. The current cleaned-up workflow is to run the manual
forward sanity suite, not the exploratory sweeps.

## Model Scope

- Cell types: A and B.
- Target: type A at inner target radius, type B at outer target radius, both measured from the type-A center.
- Active mechanism: mechanical relaxation under a two-species Morse adhesion potential.
- Disabled mechanisms: morphogen signaling, gene networks, and cell division.
- Tuned parameters: exactly three effective adhesion energies:
  - `J_AA`
  - `J_AB = J_BA`
  - `J_BB`

All other simulation parameters are fixed in `train_concentric_ring.py`. The
forward model uses JAX-Morph's `MorsePotentialSpecies`, matching the
NatCompSci branch's species-level Morse adhesion parameterization, with alpha
held fixed.

## Paper-Aligned Defaults

The shared defaults follow the paper/supplemental cadherin core-shell experiment as closely as possible while enforcing the mentor-requested simplification to two cell types and direct tuning of the adhesion matrix:

- `n_cells = 60`
- `n_dim = 3`
- no cell division
- `n_steps = 150` outer simulation steps
- `relaxation_steps = 25` mechanical relaxation substeps per outer step
- Morse well-depth values sigmoid-constrained to the paper's stable range `[0.8, 3.8]`

The current sanity cases are:

- `core_shell_refined`: `J = [[3.0, 2.5], [2.5, 1.2]]`, 40 cells, 10 type-A cells, target radii `0.75/1.5`, `seed = 2`, `n_steps = 50`, `relaxation_steps = 200`, `relaxation_dt = 1e-4`, `brownian_kT = 0.6`.
- `lattice_refined`: `J = [2.6, 2.55, 2.5]`, 40 cells, 20 type-A cells.
- `lobe_refined`: `J = [3.0, 0.8, 3.8]`, 40 cells, 20 type-A cells.

All three sanity cases use the same initial coordinates from `seed = 2`; only the cell-type labels differ.

## Loss

The implemented loss follows the supplied core-shell/ring loss, reduced to two cell types:

```text
C = mean position of type-A cells
L = mean_i (||r_i - C|| - R_type(i))^2
R_A = inner_radius
R_B = outer_radius
```

The default values are `R_A = 1.0` and `R_B = 2.0`.

## Run Forward Sanity Suite

From the repo root:

```bash
python "results-natcompsci-2025/radial shell/train_concentric_ring.py" \
  --sanity-only \
  --outdir "results-natcompsci-2025/radial shell/final_sanity_refined_new"
```

This saves the refined sanity-check figures:

- `sanity_core_shell_refined.png`
- `sanity_lattice_refined.png`
- `sanity_lobe_refined.png`
- one initial-state image for each pattern
- `sanity_checks.json`

The generated outputs are kept in:

```text
results-natcompsci-2025/radial shell/final_sanity_refined_new/
```
