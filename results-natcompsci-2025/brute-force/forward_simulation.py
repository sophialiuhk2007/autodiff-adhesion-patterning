import argparse
import os
from pathlib import Path

import jax
import jax.numpy as np

import forward_visualization_helpers as forward_viz
import istate_and_model


script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--j",
        type=float,
        nargs=6,
        required=True,
        metavar=("J_11", "J_12", "J_13", "J_22", "J_23", "J_33"),
        help="Visible three-type J vector: J_11 J_12 J_13 J_22 J_23 J_33.",
    )
    parser.add_argument("--n-type-1", type=int, default=10)
    parser.add_argument("--n-type-2", type=int, default=30)
    parser.add_argument("--n-type-3", type=int, default=90)
    parser.add_argument("--target-r1", type=float, default=1.0)
    parser.add_argument("--target-r2", type=float, default=2.0)
    parser.add_argument("--target-r3", type=float, default=3.0)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="./brute_force_forward_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    init_key, sim_key = jax.random.split(key)

    target_j = np.array(args.j)
    target_radii = np.array([args.target_r1, args.target_r2, args.target_r3])

    istate = istate_and_model.build_istate(
        init_key,
        n_type_1=args.n_type_1,
        n_type_2=args.n_type_2,
        n_type_3=args.n_type_3,
    )
    model = istate_and_model.build_model(target_j)
    fstate, trajectory = istate_and_model.simulate_forward(
        model, istate, sim_key, n_steps=args.n_steps, history=True
    )

    results = forward_viz.save_forward_visualizations(
        notebook_dir=script_dir,
        outdir=args.outdir,
        istate=istate,
        trajectory=trajectory,
        fstate=fstate,
        target_j=target_j,
        target_radii=target_radii,
        loss_fn=istate_and_model.core_shell_loss,
        n_type_1=args.n_type_1,
        n_type_2=args.n_type_2,
        n_type_3=args.n_type_3,
        n_steps=args.n_steps,
        seed=args.seed,
    )

    print("Initial image:", results["initial_path"])
    print("Final image:", results["final_path"])
    print("Animation HTML:", results["animation_html"])
    print("Final metrics:", results["final_metrics"])


if __name__ == "__main__":
    main()
