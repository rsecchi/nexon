"""Long simulations WITHOUT wavelength continuity (any free channel on each hop)."""
import random
from multiprocessing import Pool
import numpy as np
from paths import DATA
from torus_wdm import build, K, MU, SIZES

N_ARRIVALS, WARMUP = 3_000_000, 100_000
LOADS = np.arange(2.0, 4.01, 0.25)             # offered load per link [Erlang]


def simulate(args):
    N, a = args
    random.seed(1000 * N + int(100 * a))
    routes = build(N)
    R, L = len(routes), 4 * N * N
    arr_rate = a * L / sum(map(len, routes)) * R * MU
    used, active = [0] * L, []
    busy = seen = blocked = 0
    for n in range(N_ARRIVALS + WARMUP):
        while random.random() * (arr_rate + len(active) * MU) >= arr_rate:
            k = random.randrange(len(active))
            active[k], active[-1] = active[-1], active[k]
            for l in active.pop():
                used[l] -= 1
        r = routes[random.randrange(R)]
        full = any(used[l] == K for l in r)
        if n >= WARMUP:
            busy += sum(used[l] for l in r); seen += len(r); blocked += full
        if not full:
            for l in r:
                used[l] += 1
            active.append(r)
    return N, a, busy / (seen * K), blocked / N_ARRIVALS


if __name__ == "__main__":
    with Pool() as pool:
        sim = pool.map(simulate, [(N, a) for N in SIZES for a in LOADS], chunksize=1)
    np.savetxt(DATA / "torus_free_sim.csv", sim, delimiter=",", fmt="%.6g",
               header="N,offered_load_per_link,utilisation,blocking")
