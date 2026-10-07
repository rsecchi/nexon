"""
Equivalent-load rule for the N x N toroidal grid with wavelength continuity:

    blocking  ~=  ErlangB(PHI * a, K)

a   = offered load per link [Erlang] = lambda * N / (8 * mu)
PHI = equivalent-load factor, one number per grid size (and per K)

PHI is fitted here to the long simulations of torus_low_sim.py (points below 1%
blocking) and the rule is then compared with every simulated point.
"""
import numpy as np

K = 10


def erlang_b(a, k=K):
    b = 1.0
    for i in range(1, k + 1):
        b = a * b / (i + a * b)
    return b


def erlang_load(blocking, k=K):
    """Load at which Erlang B gives the stated blocking (bisection)."""
    lo, hi = 1e-3, 5.0 * k
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if erlang_b(mid, k) < blocking else (lo, mid)
    return lo


if __name__ == "__main__":
    sim = np.loadtxt("torus_low_sim.csv", delimiter=",")
    for N in (5, 7, 9):
        s = sim[sim[:, 0] == N]
        factors = np.array([erlang_load(b) / a for a, b in zip(s[:, 1], s[:, 4])])
        phi = factors[s[:, 4] < 1e-2].mean()
        print(f"\n{N}x{N} grid: PHI = {phi:.2f}   (point by point: {factors.min():.2f} to {factors.max():.2f})")
        print("   load   simulated   ErlangB(PHI*a)   ratio")
        for a, b in zip(s[:, 1], s[:, 4]):
            rule = erlang_b(phi * a)
            print(f"  {a:5.2f}  {b:10.2e}  {rule:13.2e}  {rule / b:7.2f}")
        for target in (1e-2, 1e-3, 1e-4):
            print(f"   load per link for {target:.0e} blocking: {erlang_load(target) / phi:.2f} Erlang")

    # ---------- plot: simulation (+) against ErlangB(PHI * a, K) (lines) ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wide = np.loadtxt("torus_wdm_sim.csv", delimiter=",")      # earlier, shorter runs
    wide = wide[(wide[:, 2] == 0) & (wide[:, 1] > 3.0) & (wide[:, 1] <= 5.0)]
    points = np.vstack([sim, wide])
    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    COLOUR = {5: "#2a78d6", 7: "#eb6834", 9: "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9, 6))
    loads = np.linspace(1.0, 5.2, 85)
    handles = []
    for N in (5, 7, 9):
        s = sim[sim[:, 0] == N]
        phi = np.mean([erlang_load(b) / a for a, b in zip(s[:, 1], s[:, 4]) if b < 1e-2])
        p = points[points[:, 0] == N]
        ax.plot(loads, [erlang_b(phi * a) for a in loads], "-", color=COLOUR[N], lw=1.8)
        ax.plot(p[:, 1], p[:, 4], "+", color=COLOUR[N], ms=11, mew=1.8)
        handles.append(plt.Line2D([], [], color=COLOUR[N], lw=4, label=f"{N}×{N} grid,  Φ = {phi:.2f}"))
    ax.plot(loads, [erlang_b(a) for a in loads], ":", color=MUTED, lw=1.5)
    ax.axhline(1e-2, color=MUTED, lw=0.8)
    ax.text(1.04, 1.15e-2, "1% blocking", color=MUTED, fontsize=9)
    handles += [plt.Line2D([], [], color=INK, ls="", marker="+", ms=11, mew=1.8, label="Simulation"),
                plt.Line2D([], [], color=INK, ls="-", lw=1.8, label="Erlang B at load Φ·a"),
                plt.Line2D([], [], color=MUTED, ls=":", lw=1.5, label="Erlang B at load a (single link)")]
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.set_xlim(1.0, 5.2)
    ax.set_xlabel("Offered load per link, a  [Erlang]")
    ax.set_ylabel("Connection blocking probability")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_title(f"Equivalent-load rule with wavelength continuity, K = {K} wavelengths",
                 loc="left", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig("torus_equivalent_load.png", dpi=160, facecolor="white")
