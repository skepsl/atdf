"""Paper-style plots using the actual map coordinates (horizontal y, vertical x)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import MaxNLocator


def make_reference_plot(map_display, map_extent, robot_traj, estimate_traj=None,
                        ground_truth=None, particles=None, weights=None,
                        covariances=None, roi_bounds=None, title=None,
                        recovery_note=None, max_particles=1000, particle_alpha=0.35,
                        particle_size=12.0, weight_size_scale=220.0,
                        draw_covariances=True):
    """Return a figure styled after additional/PT_2D_Result_2.pdf.

    Map pixels must have lower-origin orientation, as in ATDF.map_display.
    All inputs retain ROS x/y coordinates; only their display order changes.
    Heading angles are radians. Ellipses show each component's two-sigma
    covariance. Zero-covariance bootstrap particles are weight-sized points,
    not invented uncertainty ellipses. This function consumes no random state.
    """
    extent = np.asarray(map_extent, dtype=float).reshape(4)
    if not np.isfinite(extent).all() or extent[0] >= extent[1] or extent[2] >= extent[3]:
        raise ValueError("Map extent must contain finite ordered x/y bounds")
    robot = np.asarray(robot_traj, dtype=float).reshape(-1, 3)
    estimates = (np.empty((0, 2)) if estimate_traj is None else
                 np.asarray(estimate_traj, dtype=float).reshape(-1, 2))
    truth = None if ground_truth is None else np.asarray(ground_truth, dtype=float).reshape(2)
    means = np.empty((0, 2)) if particles is None else np.asarray(particles, dtype=float).reshape(-1, 2)
    for points in (robot, estimates, means, truth):
        if points is not None and not np.isfinite(points).all():
            raise ValueError("Plot positions and headings must be finite")
    roi = None
    if roi_bounds is not None:
        # Partial prior bounds use the map edge on their unspecified axes.
        roi = np.asarray([extent[i] if v is None or not np.isfinite(v) else v
                          for i, v in enumerate(roi_bounds)], dtype=float).reshape(4)
        if roi[0] >= roi[1] or roi[2] >= roi[3]:
            raise ValueError("RoI bounds must be ordered")

    style = {"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
             "font.size": 15, "axes.labelsize": 18, "xtick.labelsize": 14,
             "ytick.labelsize": 14, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.61, 6.18))
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.15 if recovery_note else 0.11,
                            top=0.98)
        if title:
            fig.set_label(str(title))
        ax.imshow(np.asarray(map_display).T, origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=(extent[2], extent[3], extent[0], extent[1]), interpolation="nearest",
                  zorder=0)

        legend = []
        if len(means):
            mass = np.ones(len(means)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
            if len(mass) != len(means) or not np.isfinite(mass).all() or np.any(mass < 0):
                raise ValueError("Particle weights must be finite, nonnegative and match means")
            count = min(len(means), max(1, int(max_particles)))
            # Stable selection avoids changing the experiment's RNG sequence.
            order = (np.linspace(0, len(means) - 1, count, dtype=int)
                     if np.ptp(mass) < 1e-12 else np.argsort(-mass, kind="stable")[:count])
            relative = mass[order] / max(float(mass.max()), 1e-12)
            ax.scatter(means[order, 1], means[order, 0],
                       s=particle_size + weight_size_scale * relative,
                       c="#ffa500", alpha=particle_alpha, edgecolors="none", zorder=2)
            ellipses = []
            if draw_covariances and covariances is not None:
                sigma = np.asarray(covariances, dtype=float)
                if sigma.shape != (len(means), 2, 2) or not np.isfinite(sigma).all():
                    raise ValueError("Particle covariances must be finite N x 2 x 2 arrays")
                for i in order:
                    covariance = sigma[i][::-1, ::-1]
                    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
                    if values[0] < -1e-8:
                        raise ValueError("Particle covariance must be positive semidefinite")
                    if values[-1] <= 1e-12:
                        continue
                    axes = 4 * np.sqrt(np.maximum(values, 0))
                    angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
                    ellipses.append(Ellipse((means[i, 1], means[i, 0]), axes[-1], axes[0], angle=angle))
                if ellipses:
                    ax.add_collection(PatchCollection(ellipses, facecolor="#ffa500", edgecolor="none",
                                                      alpha=particle_alpha, zorder=2))
            legend.append(Line2D([], [], color="#ffa500", marker="o", linestyle="none",
                                 markersize=10, alpha=0.55,
                                 label="Particle Cov. Ours" if ellipses else "Source particles"))

        if len(estimates):
            ax.plot(estimates[:, 1], estimates[:, 0], color="red", linewidth=0.65,
                    marker="^", markersize=10, markeredgecolor="white", markeredgewidth=0.3,
                    zorder=5)
            legend.append(Line2D([], [], color="red", marker="^", linestyle="none",
                                 markersize=10, label="Source estimate"))
        if len(robot):
            ax.plot(robot[:, 1], robot[:, 0], color="red", linewidth=0.65,
                    marker="o", markersize=7.5, markeredgecolor="white", markeredgewidth=0.3,
                    zorder=6)
            # Swapping x/y also swaps the cosine/sine components of the arrow.
            ax.quiver(robot[:, 1], robot[:, 0], 0.35 * np.sin(robot[:, 2]),
                      0.35 * np.cos(robot[:, 2]), angles="xy", scale_units="xy", scale=1,
                      color="red", width=0.004, headwidth=5, headlength=5, zorder=7)
            legend.append(Line2D([], [], color="red", marker="o", linestyle="-",
                                 linewidth=1.2, markersize=8, label="Robot pose"))
        if truth is not None:
            ax.scatter(truth[1], truth[0], s=250, marker="*", color="magenta",
                       edgecolors="white", linewidths=0.35, zorder=9)
            legend.append(Line2D([], [], color="magenta", marker="*", linestyle="none",
                                 markersize=15, label="True Source pose"))
        if len(robot) or len(estimates):
            for points in (robot, estimates):
                if len(points):
                    ax.scatter(points[0, 1], points[0, 0], s=230, marker="s",
                               facecolors="none", edgecolors="red", linewidths=1.1, zorder=8)
            legend.append(Line2D([], [], color="red", marker="s", markerfacecolor="none",
                                 linestyle="none", markersize=11, label="Init. State"))
        if roi is not None:
            box = Rectangle((roi[2], roi[0]), roi[3] - roi[2], roi[1] - roi[0],
                            fill=False, edgecolor="#00b34a", linewidth=1.5,
                            linestyle=(0, (4, 3)), zorder=4)
            ax.add_patch(box)
            legend.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#00b34a",
                                    linewidth=1.3, linestyle=(0, (4, 3)), label="RoI"))

        # Include all data, including particles omitted only by the display cap.
        points = [p[:, :2] for p in (robot, estimates, means) if len(p)]
        if truth is not None:
            points.append(truth[None])
        if roi is not None:
            points.append(np.array([[roi[0], roi[2]], [roi[1], roi[3]]]))
        if points:
            points = np.vstack(points)
            lo, hi = points.min(axis=0) - 0.65, points.max(axis=0) + 0.65
            center = (lo + hi) / 2
            span = np.maximum(hi - lo, [10.0, 12.0])
            # Match the paper's broad map panel while keeping metre scales equal.
            target_ratio = 1.27
            span[1] = max(span[1], target_ratio * span[0])
            span[0] = max(span[0], span[1] / target_ratio)
            lo, hi = center - span / 2, center + span / 2
        else:
            lo, hi = extent[[0, 2]], extent[[1, 3]]
        ax.set_xlim(lo[1], hi[1])
        ax.set_ylim(lo[0], hi[0])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y-plane [m]")
        ax.set_ylabel("x-plane [m]")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(direction="out", length=6, width=2, top=False, right=True)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        if legend:
            ax.legend(handles=legend, loc="upper right", fontsize=13.5, fancybox=True,
                      facecolor="white", edgecolor="white", framealpha=0.90,
                      handlelength=1.2, handletextpad=0.5, labelspacing=0.25, borderpad=0.4)
        if recovery_note:
            fig.text(0.54, 0.02, str(recovery_note), ha="center", va="bottom", fontsize=8,
                     color="#555555")
        return fig
