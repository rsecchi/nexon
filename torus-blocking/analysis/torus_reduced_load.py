"""
Reduced-load (Erlang fixed point) model for an N x N toroidal ISL grid with
wavelength continuity, compared with the simulation in torus_wdm.py.

State of a link = number of idle wavelengths m (0..K). Its distribution q(m) is
that of a birth-death chain, like M/M/K/K but with an arrival rate alpha(m) that
depends on the state:

    q(m)  ~  prod_{i=1..m} (K - i + 1) / alpha(i)

alpha(m) is the rate of connections actually set up through the link when it has
m idle wavelengths, i.e. the offered rate thinned by blocking on the other hops.
q and alpha depend on each other, so they are solved by iteration.

Two versions:
  keep = 0   Birman's model: links are independent
  keep > 0   links correlated: a busy wavelength on the next hop belongs, with
             probability `keep`, to a connection that came from the current hop
"""
from collections import Counter
from math import comb
import numpy as np
from paths import DATA, PLOTS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torus_wdm import build, K, SIZES


def mix(q, keep):
    """M[n, n2] = P(n2 wavelengths idle on the path so far AND on the next link |
    n idle on the path so far). The next link has y idle wavelengths (prob q[y]);
    c of its K - y busy ones carry connections continuing from the path, which
    cannot sit on a path-idle wavelength; the rest fall at random."""
    M = np.zeros((K + 1, K + 1))
    for n in range(K + 1):
        for y in range(K + 1):
            busy = K - y
            w = np.array([comb(busy, c) * keep**c * (1 - keep)**(busy - c)
                          for c in range(min(busy, K - n) + 1)], float)
            w /= w.sum()
            for c, wc in enumerate(w):
                for n2 in range(max(0, n + y - (K - c)), min(n, y) + 1):
                    M[n, n2] += q[y] * wc * comb(n, n2) * comb(K - c - n, y - n2) / comb(K - c, y)
    return M


def reduced_load(N, a, correlated):
    """Return (utilisation, blocking) for offered load a [Erlang per link]."""
    routes = build(N)
    half = (N - 1) // 2
    # fraction of a link's connections that came from the previous link of the path
    keep_straight = (half - 1) / (half + 1) if correlated else 0.0
    keep_turn = 2 * half / (N * (half + 1)) if correlated else 0.0
    # routes grouped by (intra-plane hops, inter-plane hops)
    kinds = Counter((sum(l % 4 < 2 for l in r), sum(l % 4 >= 2 for l in r)) for r in routes)
    nu = a * 4 * N * N / sum(map(len, routes))             # Erlang per route

    q = np.ones(K + 1) / (K + 1)
    for _ in range(500):
        Ms, Mt = mix(q, keep_straight), mix(q, keep_turn)
        alpha, blocking = np.zeros(K + 1), 0.0
        for (h1, h2), count in kinds.items():
            T = np.linalg.matrix_power(Ms, max(h1 - 1, 0) + max(h2 - 1, 0))
            if h1 and h2:
                T = T @ Mt
            fail = T[:, 0]                 # P(no common idle wavelength | m idle on one link)
            alpha += nu * count * (h1 + h2) / (4 * N * N) * (1 - fail)
            blocking += count * (q @ fail) / len(routes)
        new = np.cumprod([1.0] + [(K - m + 1) / alpha[m] for m in range(1, K + 1)])
        new /= new.sum()
        if abs(new - q).max() < 1e-12:
            break
        q = 0.5 * q + 0.5 * new            # damped update
    return 1 - q @ np.arange(K + 1) / K, blocking


if __name__ == "__main__":
    sim = np.loadtxt(DATA / "torus_wdm_sim.csv", delimiter=",")   # produced by torus_wdm.py
    sim = sim[sim[:, 2] == 0]                              # random wavelength assignment
    loads = np.linspace(0.8, 9, 42)

    print("  N  load | utilisation: sim  Birman  correlated | blocking: sim    Birman   correlated")
    INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, len(SIZES), figsize=(14, 5.2), sharey=True)
    for ax, N in zip(axes, SIZES):
        s = sim[sim[:, 0] == N]
        for row in s[1::2]:
            b, c = reduced_load(N, row[1], False), reduced_load(N, row[1], True)
            print(f"{N:3d} {row[1]:5.1f} | {row[3]:15.3f} {b[0]:7.3f} {c[0]:11.3f} |"
                  f" {row[4]:13.2e} {b[1]:9.2e} {c[1]:9.2e}")
        ax.plot(*np.array([reduced_load(N, a, False) for a in loads]).T, "--", color=BLUE, lw=1.5)
        ax.plot(*np.array([reduced_load(N, a, True) for a in loads]).T, "-", color=ORANGE, lw=1.8)
        ax.plot(s[:, 3], s[:, 4], "+", color=INK, ms=11, mew=1.6)
        ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_xlim(0.05, 0.6)
        ax.set_xlabel("Link utilisation (busy wavelengths / K)")
        ax.set_title(f"{N}×{N} grid", loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Connection blocking probability")
    line = plt.Line2D
    fig.legend(handles=[
        line([], [], color=INK, ls="", marker="+", ms=11, mew=1.6, label="Simulation, random wavelength"),
        line([], [], color=BLUE, ls="--", lw=1.5, label="Reduced load, independent links (Birman)"),
        line([], [], color=ORANGE, ls="-", lw=1.8, label="Reduced load, correlated links")],
        loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Reduced-load models with wavelength continuity, N×N toroidal grid, K = {K} wavelengths",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(PLOTS / "torus_reduced_load.png", dpi=160, facecolor="white")
