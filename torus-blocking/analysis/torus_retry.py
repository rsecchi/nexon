"""
Probability that a request dropped at time t is dropped again at t + delay
(same route), from torus_retry_sim.py, against a single-link Erlang model.

Model (no continuity): a dropped request found one link of its route full. That
link is an M/M/K/K system, so the probability that it is still (or again) full
after a delay is the transient p_KK(delay) of the Erlang birth-death chain.
The other h-1 links of the route are taken as independent, each full with the
Erlang B probability. Routes are weighted by how likely they are to be the one
that was dropped (longer routes are dropped more often).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import DATA, PLOTS
from torus_wdm import build, K, MU
from torus_retry_sim import N


def erlang_transient(a, delays):
    """p_KK(t): probability an M/M/K/K link is full at time t given full at 0."""
    Q = np.zeros((K + 1, K + 1))
    for k in range(K + 1):
        if k < K:
            Q[k, k + 1] = a * MU               # arrival
        if k > 0:
            Q[k, k - 1] = k * MU               # departure
        Q[k, k] = -Q[k].sum()
    w, V = np.linalg.eig(Q)
    Vinv = np.linalg.inv(V)
    return np.array([((V * np.exp(w * t)) @ Vinv)[K, K].real for t in delays]), sorted(-w.real)


def model(a, delays):
    p_full, rates = erlang_transient(a, delays)
    B = erlang_transient(a, [1e6])[0][0]       # stationary: Erlang B
    hops = np.array([len(r) for r in build(N)])
    blocked = 1 - (1 - B) ** hops              # blocking of each route
    weight = blocked / blocked.sum()           # which route was the dropped one
    again = np.array([(weight * (1 - (1 - p) * (1 - B) ** (hops - 1))).sum() for p in p_full])
    return again, (weight * blocked).sum(), rates


if __name__ == "__main__":
    sim = np.loadtxt(DATA / "torus_retry_sim.csv", delimiter=",")
    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    COLOURS = ["#2a78d6", "#eb6834"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    t = np.geomspace(0.004, 25, 200)
    for ax, c, title in zip(axes, (0, 1), ("Without wavelength continuity", "With wavelength continuity")):
        handles = []
        for colour, a in zip(COLOURS, sorted(set(sim[sim[:, 0] == c][:, 1]))):
            s = sim[(sim[:, 0] == c) & (sim[:, 1] == a)]
            B, limit = s[0, 2], s[s[:, 3] >= 10][:, 5].mean()
            ax.plot(s[:, 3], s[:, 5], "+", color=colour, ms=11, mew=1.8)
            ax.axhline(limit, color=colour, lw=0.9, ls=(0, (1, 2)))
            if c == 0:
                ax.plot(t, model(a, t)[0], "-", color=colour, lw=1.8)
            handles.append(plt.Line2D([], [], color=colour, lw=4,
                                      label=f"a = {a:g} Erlang per link (blocking {100 * B:.2f}%)"))
            excess = (s[:, 5] - limit) / (1 - limit)
            print(f"continuity {c}, a = {a:g}: blocking {B:.2e}, long-delay limit {limit:.4f}")
            for frac in (0.5, 0.1, 0.01):                  # where the excess falls to this fraction
                k = np.argmax(excess < frac)
                d0, d1, e0, e1 = s[k - 1, 3], s[k, 3], excess[k - 1], excess[k]
                cross = d0 * (d1 / d0) ** (np.log(e0 / frac) / np.log(e0 / max(e1, 1e-9)))
                print(f"    excess over the limit falls to {frac:4.0%} after {cross:.2f} holding times")
            if c == 0:
                m = model(a, s[:, 3])
                print("    model/sim:", np.round(m[0] / s[:, 5], 2)[::3], " model limit", round(m[1], 4))
                print("    Erlang relaxation rates (units of MU):", np.round(m[2][1:4], 2))
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.004, 25); ax.set_ylim(2e-3, 1.3)
        ax.set_xlabel("Delay before retrying  [mean holding times, 1/μ]")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(handles=handles, loc="upper right", frameon=False)
    axes[0].set_ylabel("Probability of being dropped again")
    fig.legend(handles=[
        plt.Line2D([], [], color=INK, ls="", marker="+", ms=11, mew=1.8, label="Simulation"),
        plt.Line2D([], [], color=INK, ls="-", lw=1.8, label="Single-link Erlang transient (no continuity only)"),
        plt.Line2D([], [], color=INK, ls=(0, (1, 2)), lw=0.9, label="Long-delay limit (simulated)")],
        loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Retrying a dropped request: {N}×{N} toroidal grid, K = {K} channels per link",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(PLOTS / "torus_retry.png", dpi=160, facecolor="white")
