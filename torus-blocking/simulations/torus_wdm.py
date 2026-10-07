"""
Blocking vs link utilisation in an N x N toroidal ISL grid WITH wavelength
continuity: a connection needs the same wavelength (1 of K) free on every hop.

  + / x  : event-driven simulation, random / first-fit wavelength assignment
  lines  : independent-link model, link-correlation model, and Erlang B
           (no continuity constraint) as a reference

Utilisation rho = mean fraction of busy wavelengths on a link.
Routing: shortest way round, intra-plane hops first, then inter-plane hops.
"""
import random
from collections import Counter
from multiprocessing import Pool
import numpy as np
from paths import DATA, PLOTS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, MU = 10, 1.0
SIZES = [5, 7, 9]                              # N (odd)
SIM_LOADS = np.arange(1.0, 8.01, 0.5)          # offered load per link [Erlang]
N_ARRIVALS, WARMUP = 1_000_000, 100_000
RHO = np.linspace(0.05, 0.75, 71)              # utilisation axis for the theory


# ---------- topology and routing ----------
def build(N):
    """Routes as lists of directed link ids (4 per node)."""
    def step(a, b):                            # shortest direction round the ring
        return 1 if (b - a) % N <= N // 2 else -1

    def path(i, j, i2, j2):                    # link id = 4*node + direction
        links = []
        while i != i2:                         # intra-plane
            s = step(i, i2); links.append(4 * (i * N + j) + (0 if s > 0 else 1)); i = (i + s) % N
        while j != j2:                         # inter-plane
            s = step(j, j2); links.append(4 * (i * N + j) + (2 if s > 0 else 3)); j = (j + s) % N
        return links

    nodes = [(i, j) for i in range(N) for j in range(N)]
    return [path(*s, *d) for s in nodes for d in nodes if s != d]


# ---------- theory (all as a function of the utilisation rho) ----------
def independent(routes, rho):
    """Links independent: a wavelength is free on an h-hop path w.p. (1-rho)^h,
    and the connection is blocked if none of the K wavelengths is."""
    h = np.array([len(r) for r in routes])
    return np.array([((1 - (1 - x) ** h) ** K).mean() for x in rho])

def correlated(routes, rho):
    """Barry-Humblet style: wavelength occupancy is a Markov chain along the path.
    p_leave = fraction of the connections on a link that do not continue on the
    next link of the path; a wavelength free on one hop is busy on the next only
    if a connection joins there (probability p_new)."""
    on_link = Counter(l for r in routes for l in r)
    on_pair = Counter(p for r in routes for p in zip(r, r[1:]))
    out = []
    for x in rho:
        blocked = 0.0
        for r in routes:
            free = 1 - x                                   # first hop
            for l1, l2 in zip(r, r[1:]):
                p_leave = 1 - on_pair[l1, l2] / on_link[l1]
                p_new = x * p_leave / (1 - x * (1 - p_leave))
                free *= 1 - p_new
            blocked += (1 - free) ** K
        out.append(blocked / len(routes))
    return np.array(out)

def erlang_b_reference(routes, loads):
    """No continuity constraint (wavelength conversion at every node)."""
    h = np.array([len(r) for r in routes])
    util, block = [], []
    for a in loads:
        B = 1.0
        for i in range(1, K + 1):
            B = a * B / (i + a * B)
        util.append(a * (1 - B) / K); block.append(1 - ((1 - B) ** h).mean())
    return np.array(util), np.array(block)


# ---------- simulation ----------
def simulate(args):
    N, a, first_fit = args
    random.seed(1000 * N + int(10 * a))
    routes = build(N)
    R, L = len(routes), 4 * N * N
    arr_rate = a * L / sum(map(len, routes)) * R * MU      # total arrival rate
    free = [(1 << K) - 1] * L                  # bitmask of free wavelengths per link
    active = []
    busy = seen = blocked = 0
    for n in range(N_ARRIVALS + WARMUP):
        # departures until the next arrival (memoryless: any active call may leave)
        while random.random() * (arr_rate + len(active) * MU) >= arr_rate:
            k = random.randrange(len(active))
            active[k], active[-1] = active[-1], active[k]
            r, w = active.pop()
            for l in r:
                free[l] |= w
        r = routes[random.randrange(R)]
        avail = (1 << K) - 1
        for l in r:
            avail &= free[l]                   # wavelengths free on every hop
        if n >= WARMUP:                        # arrivals see time averages (PASTA)
            busy += sum(K - free[l].bit_count() for l in r); seen += len(r)
            blocked += avail == 0
        if avail:
            if first_fit:
                w = avail & -avail             # lowest-numbered free wavelength
            else:
                w = random.choice([1 << b for b in range(K) if avail >> b & 1])
            for l in r:
                free[l] ^= w
            active.append((r, w))
    return N, a, first_fit, busy / (seen * K), blocked / N_ARRIVALS


# ---------- run and plot ----------
if __name__ == "__main__":
    jobs = [(N, a, ff) for N in SIZES for a in SIM_LOADS for ff in (0, 1)]
    with Pool() as pool:
        sim = np.array(pool.map(simulate, jobs, chunksize=1))
    np.savetxt(DATA / "torus_wdm_sim.csv", sim, delimiter=",", fmt="%.6g",
               header="N,offered_load_per_link,first_fit,utilisation,blocking")

    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, len(SIZES), figsize=(14, 5.2), sharey=True)
    for ax, N in zip(axes, SIZES):
        routes = build(N)
        ax.plot(*erlang_b_reference(routes, np.linspace(0.5, 10, 60)), ":", color=MUTED, lw=1.5)
        ax.plot(RHO, independent(routes, RHO), "--", color=BLUE, lw=1.5)
        ax.plot(RHO, correlated(routes, RHO), "-", color=ORANGE, lw=1.8)
        for ff, marker in ((0, "+"), (1, "x")):
            s = sim[(sim[:, 0] == N) & (sim[:, 2] == ff)]
            ax.plot(s[:, 3], s[:, 4], marker, color=INK, ms=9 if ff else 11, mew=1.6)
        ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_xlim(0.05, 0.7)
        ax.set_xlabel("Link utilisation (busy wavelengths / K)")
        ax.set_title(f"{N}×{N} grid", loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Connection blocking probability")

    line = plt.Line2D
    h = [line([], [], color=INK, ls="", marker="+", ms=11, mew=1.6, label="Simulation, random wavelength"),
         line([], [], color=INK, ls="", marker="x", ms=9, mew=1.6, label="Simulation, first-fit wavelength"),
         line([], [], color=BLUE, ls="--", lw=1.5, label="Theory: independent links"),
         line([], [], color=ORANGE, ls="-", lw=1.8, label="Theory: correlated links"),
         line([], [], color=MUTED, ls=":", lw=1.5, label="Erlang B, no continuity constraint")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Blocking vs utilisation with wavelength continuity, N×N toroidal grid, K = {K} wavelengths",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(PLOTS / "torus_wdm_blocking.png", dpi=160, facecolor="white")
