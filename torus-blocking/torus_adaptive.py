"""
Adaptive routing (torus_adaptive_sim.py) against the equivalent-load rule
blocking ~= ErlangB(PHI * a, K), and against the fixed intra-then-inter routing.

Reads torus_adaptive_sim.csv, plus the fixed-routing results torus_free_sim.csv
(no continuity) and torus_low_sim.csv (continuity).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torus_equivalent_load import erlang_b, erlang_load, K

SIZES = (5, 7, 9)


def fit_phi(load, blocking):
    keep = blocking < 1e-2
    return np.mean([erlang_load(b) / a for a, b in zip(load[keep], blocking[keep])])


if __name__ == "__main__":
    adaptive = np.loadtxt("torus_adaptive_sim.csv", delimiter=",")   # N, load, cont, util, blocking
    fixed = {0: np.loadtxt("torus_free_sim.csv", delimiter=",")[:, [0, 1, 3]],
             1: np.loadtxt("torus_low_sim.csv", delimiter=",")[:, [0, 1, 4]]}
    titles = {0: "Without wavelength continuity", 1: "With wavelength continuity"}

    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    COLOUR = {5: "#2a78d6", 7: "#eb6834", 9: "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    loads = np.linspace(1.0, 4.2, 65)
    print("continuity  N | PHI fixed  PHI adaptive | rule/sim below 1% | load for 1%: fixed  adaptive")
    for ax, c in zip(axes, (0, 1)):
        handles = []
        for N in SIZES:
            d = adaptive[(adaptive[:, 0] == N) & (adaptive[:, 2] == c)]
            f = fixed[c][fixed[c][:, 0] == N]
            phi, phi_fixed = fit_phi(d[:, 1], d[:, 4]), fit_phi(f[:, 1], f[:, 2])
            ax.plot(loads, [erlang_b(phi_fixed * a) for a in loads], "--", color=COLOUR[N], lw=1.2)
            ax.plot(loads, [erlang_b(phi * a) for a in loads], "-", color=COLOUR[N], lw=1.8)
            ax.plot(d[:, 1], d[:, 4], "+", color=COLOUR[N], ms=11, mew=1.8)
            handles.append(plt.Line2D([], [], color=COLOUR[N], lw=4,
                                      label=f"{N}×{N} grid,  Φ = {phi:.2f}  (fixed: {phi_fixed:.2f})"))
            low = d[d[:, 4] < 1e-2]
            ratio = [erlang_b(phi * a) / b for a, b in zip(low[:, 1], low[:, 4])]
            print(f"{c:10d} {N:2d} | {phi_fixed:9.3f} {phi:13.3f} | {min(ratio):6.2f} to {max(ratio):.2f}    |"
                  f" {erlang_load(1e-2) / phi_fixed:17.2f} {erlang_load(1e-2) / phi:9.2f}")
        ax.plot(loads, [erlang_b(a) for a in loads], ":", color=MUTED, lw=1.5)
        ax.axhline(1e-2, color=MUTED, lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.set_xlim(1.0, 4.2)
        ax.set_xlabel("Offered load per link, a  [Erlang]")
        ax.set_title(titles[c], loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=handles, loc="lower right", frameon=False)
    axes[0].set_ylabel("Connection blocking probability")
    axes[0].text(1.04, 1.15e-2, "1% blocking", color=MUTED, fontsize=9)
    fig.legend(handles=[
        plt.Line2D([], [], color=INK, ls="", marker="+", ms=11, mew=1.8, label="Simulation, adaptive routing"),
        plt.Line2D([], [], color=INK, ls="-", lw=1.8, label="Erlang B at load Φ·a, adaptive routing"),
        plt.Line2D([], [], color=INK, ls="--", lw=1.2, label="Erlang B at load Φ·a, fixed routing"),
        plt.Line2D([], [], color=MUTED, ls=":", lw=1.5, label="Erlang B at load a (single link)")],
        loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Adaptive shortest-path routing, N×N toroidal grid, K = {K} channels per link",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig("torus_adaptive.png", dpi=160, facecolor="white")
