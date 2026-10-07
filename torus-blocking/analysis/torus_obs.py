"""
OBS-style continuity (sender picks the wavelength): simulation against

    burst loss ~= ErlangB(PHI1 * a / K, 1) = x / (1 + x),   x = PHI1 * a / K

i.e. an Erlang loss system with ONE channel carrying the per-wavelength load
a / K scaled by PHI1. PHI1 is not fitted: it is the mean, over routes, of the
traffic that joins the path after the first hop,

    PHI1 = mean over routes of  sum over later hops (1 - kappa_hop)

where kappa_hop is the fraction of a link's traffic that came from the previous
link of the route. For the N x N grid with n = (N - 1) / 2:

    PHI1 = N (n-1) / (n+1)^2  +  n / (n+1)  -  2 n^2 / (N (n+1)^2)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import DATA, PLOTS
from torus_wdm import K
from torus_equivalent_load import erlang_b

SIZES = (5, 7, 9)
PHI_PATH_AWARE = {5: 1.56, 7: 1.78, 9: 1.92}   # wavelength chosen knowing the whole path


def phi1(N):
    n = (N - 1) // 2
    return N * (n - 1) / (n + 1) ** 2 + n / (n + 1) - 2 * n * n / (N * (n + 1) ** 2)


def rule(N, a):
    x = phi1(N) * a / K
    return x / (1 + x)


if __name__ == "__main__":
    sim = np.loadtxt(DATA / "torus_obs_sim.csv", delimiter=",")    # N, load, occupy, util, loss
    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    COLOUR = {5: "#2a78d6", 7: "#eb6834", 9: "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    loads = np.geomspace(0.008, 5, 120)
    titles = {0: "Lost bursts use no resources", 1: "Lost bursts hold the hops already crossed"}
    print("occupy  N | PHI1 formula | PHI1 from simulation (min to max) | rule/sim (min to max) | load for 1% loss")
    for ax, occupy in zip(axes, (0, 1)):
        handles = []
        for N in SIZES:
            s = sim[(sim[:, 0] == N) & (sim[:, 2] == occupy)]
            fitted = (s[:, 4] / (1 - s[:, 4])) / (s[:, 1] / K)
            ratio = np.array([rule(N, a) for a in s[:, 1]]) / s[:, 4]
            print(f"{occupy:6d} {N:2d} | {phi1(N):12.3f} | {fitted.min():14.2f} to {fitted.max():.2f}"
                  f"          | {ratio.min():8.2f} to {ratio.max():.2f}     | {K * (0.01 / 0.99) / phi1(N):8.3f}")
            ax.plot(loads, [erlang_b(PHI_PATH_AWARE[N] * a) for a in loads], "--", color=COLOUR[N], lw=1.2)
            ax.plot(loads, [rule(N, a) for a in loads], "-", color=COLOUR[N], lw=1.8)
            ax.plot(s[:, 1], s[:, 4], "+", color=COLOUR[N], ms=11, mew=1.8)
            handles.append(plt.Line2D([], [], color=COLOUR[N], lw=4, label=f"{N}×{N} grid,  Φ₁ = {phi1(N):.2f}"))
        ax.axhline(1e-2, color=MUTED, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.008, 5); ax.set_ylim(1e-5, 1)
        ax.set_xlabel("Offered load per link, a  [Erlang]")
        ax.set_title(titles[occupy], loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=handles, loc="upper left", frameon=False)
    axes[0].set_ylabel("Burst loss probability")
    axes[0].text(0.25, 1.15e-2, "1% loss", color=MUTED, fontsize=9)
    fig.legend(handles=[
        plt.Line2D([], [], color=INK, ls="", marker="+", ms=11, mew=1.8, label="Simulation, sender picks the wavelength"),
        plt.Line2D([], [], color=INK, ls="-", lw=1.8, label="Erlang B with one channel at load Φ₁·a/K"),
        plt.Line2D([], [], color=INK, ls="--", lw=1.2, label="Wavelength chosen knowing the whole path (earlier result)")],
        loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"OBS-style wavelength continuity, N×N toroidal grid, K = {K} wavelengths per link",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(PLOTS / "torus_obs.png", dpi=160, facecolor="white")
