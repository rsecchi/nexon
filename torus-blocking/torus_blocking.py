"""
Blocking in an M x N toroidal ISL grid, modelled as a Kelly loss network.

  node (i, j): satellite i (0..M-1) in orbital plane j (0..N-1)
  4 directed links per node, K channels each (no wavelength continuity)
  connections: Poisson, uniform destination, exp. holding time (mean 1/MU)
  routing: shortest way round, intra-plane hops first, then inter-plane hops

Three estimates of blocking are compared for each load:
  1. event-driven simulation (blocked calls cleared)
  2. Kelly's product form: independent Poisson route counts, keep only the
     samples that respect the K-channel limit on every link (exact, but the
     acceptance rate collapses at high load)
  3. analysis: plain Erlang B per link, and the Erlang fixed point (reduced load)
"""
import random
import numpy as np

M, N, K, MU = 7, 7, 10, 1.0          # M, N must be odd
LOADS = [2, 3, 4, 5, 6, 7, 8]        # offered load per link [Erlang]
random.seed(1)
rng = np.random.default_rng(1)


# ---------- topology and routing ----------
def step(a, b, n):                   # shortest direction round a ring of odd size n
    return 1 if (b - a) % n <= n // 2 else -1

def path(i, j, i2, j2):
    """Directed links used. Link id = 4*node + (0: i+1, 1: i-1, 2: j+1, 3: j-1)."""
    links = []
    while i != i2:                   # intra-plane
        s = step(i, i2, M); links.append(4 * (i * N + j) + (0 if s > 0 else 1)); i = (i + s) % M
    while j != j2:                   # inter-plane
        s = step(j, j2, N); links.append(4 * (i * N + j) + (2 if s > 0 else 3)); j = (j + s) % N
    return links

nodes = [(i, j) for i in range(M) for j in range(N)]
routes = [path(*s, *d) for s in nodes for d in nodes if s != d]
R, L = len(routes), 4 * M * N
hops = np.array([len(r) for r in routes])
A = np.zeros((L, R))                 # link-route incidence matrix
for r, links in enumerate(routes):
    A[links, r] = 1


# ---------- analysis ----------
def erlang_b(a, k):
    b = 1.0
    for i in range(1, k + 1):
        b = a * b / (i + a * b)
    return b

def erlang_simple(nu):               # links independent, offered load not thinned
    B = np.array([erlang_b(x, K) for x in nu * A.sum(1)])
    return B.mean(), 1 - np.exp(A.T @ np.log1p(-B)).mean()

def erlang_fixed_point(nu):          # reduced-load approximation
    B = np.zeros(L)
    for _ in range(500):
        free = np.exp(A.T @ np.log1p(-B))            # P(all links of a route free)
        B = np.array([erlang_b(x, K) for x in nu * (A @ free) / (1 - B)])
    return B.mean(), 1 - free.mean()


# ---------- 1. event-driven simulation ----------
def simulate(nu, n_arrivals=1_000_000, warmup=50_000):
    used, active = [0] * L, []
    arr_rate = nu * R * MU
    blocked = full = seen = 0
    for n in range(n_arrivals + warmup):
        # departures until the next arrival (memoryless: any active call may leave)
        while random.random() * (arr_rate + len(active) * MU) >= arr_rate:
            k = random.randrange(len(active))
            active[k], active[-1] = active[-1], active[k]
            for l in active.pop():
                used[l] -= 1
        r = routes[random.randrange(R)]
        nfull = sum(used[l] == K for l in r)
        if n >= warmup:
            full += nfull; seen += len(r); blocked += nfull > 0
        if nfull == 0:
            for l in r:
                used[l] += 1
            active.append(r)
    return full / seen, blocked / n_arrivals


# ---------- 2. Kelly product form (rejection sampling) ----------
def kelly(nu, n_draws=200_000, batch=2_000):
    kept = full = blocked = 0
    for _ in range(n_draws // batch):
        y = rng.poisson(nu, (batch, R)) @ A.T        # link occupancies
        y = y[(y <= K).all(1)]                       # keep feasible states only
        f = (y == K).astype(float)
        kept += len(y); full += f.sum(); blocked += ((f @ A) > 0).sum()
    if kept < 200:
        return kept / n_draws, np.nan, np.nan
    return kept / n_draws, full / (kept * L), blocked / (kept * R)


# ---------- run ----------
if __name__ == "__main__":
    d = hops.mean()
    print(f"{M}x{N} torus, K={K}: {R} routes, {L} directed links")
    print(f"mean path length: measured {d:.4f}, (M+N)/4 = {(M+N)/4:.4f}, (M+N+2)/4 = {(M+N+2)/4:.4f}")
    per_link = A.sum(1) / R * (M * N)                # hops carried per link, per unit node load
    print(f"load per link / node load: intra-plane {per_link[0]:.4f}, inter-plane {per_link[2]:.4f}\n")

    print("load |        link blocking                  |        path blocking                  | Kelly")
    print("[Erl]|  sim      Kelly    ErlangB  fixed-pt  |  sim      Kelly    ErlangB  fixed-pt  | accept")
    for a in LOADS:
        nu = a * L / hops.sum()                      # Erlangs per route (mean link load = a)
        s, k, e, f = simulate(nu), kelly(nu), erlang_simple(nu), erlang_fixed_point(nu)
        print(f"{a:4.1f} | {s[0]:.2e} {k[1]:.2e} {e[0]:.2e} {f[0]:.2e} |"
              f" {s[1]:.2e} {k[2]:.2e} {e[1]:.2e} {f[1]:.2e} | {k[0]:.1e}")
