"""
Blocking vs link utilisation in an N x N toroidal ISL grid (K channels per link).

  crosses : event-driven simulation
  lines   : Erlang B (M/M/K/K per link) and the Erlang fixed point

Utilisation = mean fraction of busy channels on a link (carried load / K).
Routing: shortest way round, intra-plane hops first, then inter-plane hops.
"""
import random
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, MU = 10, 1.0
SIZES = [5, 7, 9]                              # N (odd)
SIM_LOADS = np.arange(2.5, 9.01, 0.5)          # offered load per link [Erlang]
LINE_LOADS = np.linspace(1.5, 10, 60)
N_ARRIVALS, WARMUP = 2_000_000, 100_000


# ---------- topology and routing ----------
def build(N):
    """Routes (lists of directed link ids) and link-route incidence matrix."""
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
    routes = [path(*s, *d) for s in nodes for d in nodes if s != d]
    A = np.zeros((4 * N * N, len(routes)))
    for r, links in enumerate(routes):
        A[links, r] = 1
    return routes, A


# ---------- theory ----------
def erlang_b(a, k=K):
    b = 1.0
    for i in range(1, k + 1):
        b = a * b / (i + a * b)
    return b

def theory(N, a):
    """(utilisation, link blocking, path blocking) for Erlang B and fixed point.
    All links are equivalent, so one blocking value B describes the network."""
    hops = build(N)[1].sum(0)                  # path length of every route
    # Erlang B: every link sees the full offered load a
    B = erlang_b(a)
    eb = (a * (1 - B) / K, B, 1 - ((1 - B) ** hops).mean())
    # fixed point: load offered to a link is thinned by blocking on the other hops
    B = 0.0
    for _ in range(2000):                      # damped, or it oscillates at high load
        B = 0.8 * B + 0.2 * erlang_b(a * ((1 - B) ** (hops - 1) * hops).sum() / hops.sum())
    accepted = (1 - B) ** hops
    fp = (a * (accepted * hops).sum() / hops.sum() / K, B, 1 - accepted.mean())
    return eb, fp


# ---------- simulation ----------
def simulate(args):
    N, a = args
    random.seed(1000 * N + int(10 * a))
    routes, A = build(N)
    R, L = len(routes), 4 * N * N
    arr_rate = a * L / A.sum() * R * MU        # total arrival rate
    used, active = [0] * L, []
    busy = full = seen = blocked = 0
    for n in range(N_ARRIVALS + WARMUP):
        # departures until the next arrival (memoryless: any active call may leave)
        while random.random() * (arr_rate + len(active) * MU) >= arr_rate:
            k = random.randrange(len(active))
            active[k], active[-1] = active[-1], active[k]
            for l in active.pop():
                used[l] -= 1
        r = routes[random.randrange(R)]
        nfull = sum(used[l] == K for l in r)
        if n >= WARMUP:                        # arrivals see time averages (PASTA)
            busy += sum(used[l] for l in r); seen += len(r)
            full += nfull; blocked += nfull > 0
        if nfull == 0:
            for l in r:
                used[l] += 1
            active.append(r)
    return N, a, busy / (seen * K), full / seen, blocked / N_ARRIVALS


# ---------- run and plot ----------
if __name__ == "__main__":
    with Pool() as pool:
        sim = pool.map(simulate, [(N, a) for N in SIZES for a in SIM_LOADS], chunksize=1)
    np.savetxt("torus_sim.csv", sim, delimiter=",", fmt="%.6g",
               header="N,offered_load_per_link,utilisation,link_blocking,path_blocking")
    sim = np.array(sim)

    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    COLOUR = dict(zip(SIZES, ["#2a78d6", "#eb6834", "#1baf7a"]))
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    titles = ["Link blocking (probability a link is full)",
              "Connection blocking (probability a request is refused)"]
    for col, (ax, title) in enumerate(zip(axes, titles)):
        for N in SIZES:
            th = np.array([theory(N, a) for a in LINE_LOADS])        # [load, model, quantity]
            c = COLOUR[N]
            if col == 1:                                             # Erlang B depends on N here
                ax.plot(th[:, 0, 0], th[:, 0, 2], "--", color=c, lw=1.3)
            ax.plot(th[:, 1, 0], th[:, 1, col + 1], "-", color=c, lw=1.6)
            s = sim[sim[:, 0] == N]
            ax.plot(s[:, 2], s[:, col + 3], "+", color=c, ms=10, mew=1.8)
        if col == 0:                                                 # one curve for all N
            ax.plot(th[:, 0, 0], th[:, 0, 1], "--", color=INK, lw=1.3)
        ax.set_yscale("log"); ax.set_ylim(1e-5, 1); ax.set_xlim(0.15, 0.75)
        ax.set_xlabel("Link utilisation (carried load per channel)")
        ax.set_ylabel("Blocking probability")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    h = [plt.Line2D([], [], color=COLOUR[N], lw=4, label=f"{N}×{N} grid") for N in SIZES]
    h += [plt.Line2D([], [], color=INK, ls="", marker="+", ms=10, mew=1.8, label="Simulation"),
          plt.Line2D([], [], color=INK, ls="-", lw=1.6, label="Erlang fixed point"),
          plt.Line2D([], [], color=INK, ls="--", lw=1.3, label="Erlang B (M/M/K/K per link)")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(f"Blocking vs utilisation in an N×N toroidal grid, K = {K} channels per link",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig("torus_blocking.png", dpi=160, facecolor="white")
