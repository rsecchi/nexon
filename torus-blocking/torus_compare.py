"""
With and without wavelength continuity: simulation against the equivalent-load
rule  blocking ~= ErlangB(PHI * a, K),  a = offered load per link.

Reads torus_free_sim.csv (no continuity) and torus_low_sim.csv / torus_wdm_sim.csv
(continuity, random wavelength assignment). PHI is fitted on points below 1%.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torus_wdm import build
from torus_equivalent_load import erlang_b, erlang_load, K

SIZES = (5, 7, 9)


def fit_phi(load, blocking):
    keep = blocking < 1e-2
    return np.mean([erlang_load(b) / a for a, b in zip(load[keep], blocking[keep])])


if __name__ == "__main__":
    free = np.loadtxt("torus_free_sim.csv", delimiter=",")         # N, load, util, blocking
    low = np.loadtxt("torus_low_sim.csv", delimiter=",")           # N, load, ff, util, blocking
    wide = np.loadtxt("torus_wdm_sim.csv", delimiter=",")
    wide = wide[(wide[:, 2] == 0) & (wide[:, 1] > 3.0) & (wide[:, 1] <= 4.0)]
    cont = np.vstack([low, wide])[:, [0, 1, 3, 4]]
    cases = [("Without wavelength continuity", free), ("With wavelength continuity", cont)]

    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    COLOUR = {5: "#2a78d6", 7: "#eb6834", 9: "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    loads = np.linspace(1.0, 4.2, 65)
    print("  N  mean hops | PHI without   PHI with   ratio | hops^(1/K)")
    phis = {}
    for ax, (title, data) in zip(axes, cases):
        handles = []
        for N in SIZES:
            d = data[data[:, 0] == N]
            phi = phis[title, N] = fit_phi(d[:, 1], d[:, 3])
            ax.plot(loads, [erlang_b(phi * a) for a in loads], "-", color=COLOUR[N], lw=1.8)
            ax.plot(d[:, 1], d[:, 3], "+", color=COLOUR[N], ms=11, mew=1.8)
            handles.append(plt.Line2D([], [], color=COLOUR[N], lw=4, label=f"{N}×{N} grid,  Φ = {phi:.2f}"))
        ax.plot(loads, [erlang_b(a) for a in loads], ":", color=MUTED, lw=1.5)
        ax.axhline(1e-2, color=MUTED, lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.set_xlim(1.0, 4.2)
        ax.set_xlabel("Offered load per link, a  [Erlang]")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=handles, loc="lower right", frameon=False)
    axes[0].set_ylabel("Connection blocking probability")
    axes[0].text(1.04, 1.15e-2, "1% blocking", color=MUTED, fontsize=9)
    fig.legend(handles=[
        plt.Line2D([], [], color=INK, ls="", marker="+", ms=11, mew=1.8, label="Simulation"),
        plt.Line2D([], [], color=INK, ls="-", lw=1.8, label="Erlang B at load Φ·a"),
        plt.Line2D([], [], color=MUTED, ls=":", lw=1.5, label="Erlang B at load a (single link)")],
        loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Equivalent-load rule with and without wavelength continuity, K = {K} channels per link",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig("torus_compare.png", dpi=160, facecolor="white")

    for N in SIZES:
        hops = np.mean([len(r) for r in build(N)])
        p0, p1 = phis[cases[0][0], N], phis[cases[1][0], N]
        print(f"{N:3d} {hops:9.1f} | {p0:11.3f} {p1:10.3f} {p1 / p0:7.2f} | {hops ** (1 / K):8.3f}")
    print("\nrule / simulation, points below 1%:")
    for title, data in cases:
        for N in SIZES:
            d = data[(data[:, 0] == N) & (data[:, 3] < 1e-2)]
            ratio = [erlang_b(phis[title, N] * a) / b for a, b in zip(d[:, 1], d[:, 3])]
            print(f"  {title:32s} {N}x{N}: {min(ratio):.2f} to {max(ratio):.2f}")
    print("\nload per link [Erlang] for 1% and 0.1% blocking:")
    for N in SIZES:
        p0, p1 = phis[cases[0][0], N], phis[cases[1][0], N]
        print(f"  {N}x{N}: without {erlang_load(1e-2) / p0:.2f}, {erlang_load(1e-3) / p0:.2f}"
              f"   with {erlang_load(1e-2) / p1:.2f}, {erlang_load(1e-3) / p1:.2f}")
