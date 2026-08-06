import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib

def set_axes_equal(ax, positions, radius):
    mins = np.asarray(positions).min(axis=0) - radius
    maxs = np.asarray(positions).max(axis=0) + radius
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.5 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

def plot_translucent_spheres(state, path, title, metrics, effective_j, cell_mask=None):
    positions = np.asarray(state.position)
    type_idx = np.asarray(jnp.argmax(state.celltype, axis=1))
    radii = np.asarray(state.radius).reshape(-1)
    colors = ["#3452b8", "#d4ad4f", "#b4002c"]
    alphas = [0.56, 0.44, 0.34]

    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 15)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    visible = np.ones(positions.shape[0], dtype=bool)
    if cell_mask is not None:
        visible = np.asarray(cell_mask, dtype=bool)
    order = np.asarray([i for i in np.argsort(type_idx)[::-1] if visible[i]])
    for i in order:
        color = colors[type_idx[i]]
        radius = float(radii[i])
        ax.plot_surface(
            positions[i, 0] + radius * sphere_x,
            positions[i, 1] + radius * sphere_y,
            positions[i, 2] + radius * sphere_z,
            color=color,
            alpha=alphas[type_idx[i]],
            linewidth=0.0,
            shade=True,
            antialiased=True,
        )

    set_axes_equal(ax, positions[visible], float(np.max(radii[visible])))
    ax.view_init(elev=18, azim=35)
    ax.set_axis_off()
    ax.set_title(title, fontsize=18, pad=14)
    text = (
        "J =\n"
        f"{np.array2string(np.asarray(effective_j), precision=2)}\n"
        f"loss: {metrics['loss']:.3f}\n"
        f"mean r: {metrics['mean_radius_type_1']:.2f}, "
        f"{metrics['mean_radius_type_2']:.2f}, "
        f"{metrics['mean_radius_type_3']:.2f}"
    )
    ax.text2D(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
