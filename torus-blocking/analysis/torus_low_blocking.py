"""
Low-blocking region (< 1%) of the N x N toroidal grid with wavelength continuity.

At these loads almost nothing is blocked, so the load on a link is not reduced
and its idle-wavelength distribution is simply Erlang. This gives a ONE-PASS
model (no iteration):

  1. q(m) = Erlang distribution of the idle wavelengths on a link at load a
  2. follow each path hop by hop with mix() (correlated links) and read off the
     probability that no wavelength is idle on every hop

It is compared with the full reduced-load fixed point and with the long
simulations of torus_low_sim.py.
"""
from collections import Counter
from math import factorial
import numpy as np
from paths import DATA, PLOTS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torus_wdm import build, K, SIZES
from torus_reduced_load import mix, reduced_load


def one_pass(N, a):
    """Blocking at offered load a [Erlang per link], correlated links, no iteration."""
    half = (N - 1) // 2
    busy = np.array([a**k / factorial(k) for k in range(K + 1)])
    q = (busy / busy.sum())[::-1]                          # q[m], m = idle wavelengths
    Ms = mix(q, (half - 1) / (half + 1))                   # going straight
    Mt = mix(q, 2 * half / (N * (half + 1)))               # intra- to inter-plane turn
    routes = build(N)
    kinds = Counter((sum(l % 4 < 2 for l in r), sum(l % 4 >= 2 for l in r)) for r in routes)
    blocking = 0.0
    for (h1, h2), count in kinds.items():
        T = np.linalg.matrix_power(Ms, max(h1 - 1, 0) + max(h2 - 1, 0))
        if h1 and h2:
            T = T @ Mt
        blocking += count * (q @ T[:, 0]) / len(routes)
    return blocking


def no_continuity(N, a):
    """Erlang B per link, path blocking 1 - (1-B)^h (wavelength conversion)."""
    B = 1.0
    for i in range(1, K + 1):
        B = a * B / (i + a * B)
    return 1 - np.mean([(1 - B) ** len(r) for r in build(N)])


def load_for(model, N, target):
    """Offered load per link giving the target blocking (bisection)."""
    lo, hi = 0.2, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if model(N, mid) < target else (lo, mid)
    return lo


if __name__ == "__main__":
    sim = np.loadtxt(DATA / "torus_low_sim.csv", delimiter=",")   # produced by torus_low_sim.py
    arrivals = 5_000_000

    print("  N  load   util |  simulated (+-)    one-pass  fixed-pt  Birman   | one-pass/sim")
    for N, a, _, util, blk in sim:
        N = int(N)
        err = 1.96 * np.sqrt(blk / arrivals)
        op, fp, bi = one_pass(N, a), reduced_load(N, a, True)[1], reduced_load(N, a, False)[1]
        print(f"{N:3d} {a:5.2f} {util:6.3f} | {blk:9.2e} ({err:7.1e}) {op:9.2e} {fp:9.2e} {bi:9.2e} | {op/blk:6.2f}")

    print("\nOffered load per link [Erlang] at a target blocking (utilisation = load / K):")
    print("  N  target | with continuity (one-pass)   no continuity (Erlang B)")
    for N in SIZES:
        for target in (1e-2, 1e-3, 1e-4):
            print(f"{N:3d}  {target:6.0e} | {load_for(one_pass, N, target):12.2f} "
                  f"{load_for(no_continuity, N, target):28.2f}")

    INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, len(SIZES), figsize=(14, 5.2), sharey=True)
    loads = np.linspace(1.0, 3.2, 23)
    for ax, N in zip(axes, SIZES):
        s = sim[sim[:, 0] == N]
        ax.plot(loads / K, [no_continuity(N, a) for a in loads], ":", color=MUTED, lw=1.5)
        ax.plot(loads / K, [reduced_load(N, a, False)[1] for a in loads], "--", color=BLUE, lw=1.5)
        ax.plot(loads / K, [one_pass(N, a) for a in loads], "-", color=ORANGE, lw=1.8)
        ax.plot(s[:, 3], s[:, 4], "+", color=INK, ms=11, mew=1.6)
        ax.axhline(1e-2, color=MUTED, lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(1e-6, 1e-1); ax.set_xlim(0.1, 0.32)
        ax.set_xlabel("Link utilisation (busy wavelengths / K)")
        ax.set_title(f"{N}×{N} grid", loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Connection blocking probability")
    axes[0].text(0.102, 1.15e-2, "1% blocking", color=MUTED, fontsize=9)
    line = plt.Line2D
    fig.legend(handles=[
        line([], [], color=INK, ls="", marker="+", ms=11, mew=1.6, label="Simulation, random wavelength"),
        line([], [], color=ORANGE, ls="-", lw=1.8, label="One-pass model, correlated links"),
        line([], [], color=BLUE, ls="--", lw=1.5, label="Independent links (Birman)"),
        line([], [], color=MUTED, ls=":", lw=1.5, label="Erlang B, no continuity constraint")],
        loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Low-blocking region with wavelength continuity, N×N toroidal grid, K = {K} wavelengths",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(PLOTS / "torus_low_blocking.png", dpi=160, facecolor="white")
