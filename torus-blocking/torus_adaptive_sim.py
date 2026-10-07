"""
Adaptive shortest-path routing on the N x N toroidal grid.

At every node the connection may continue intra-plane or inter-plane, whichever
still brings it closer to the destination. One of the two is picked at random;
if that link is blocked the other is tried; if both are blocked (or the only
remaining one is) the connection is dropped. Paths always have minimum length.

  continuity = 0 : a link is blocked when all its K channels are busy
  continuity = 1 : a link is blocked when none of the wavelengths still free on
                   the path so far is free on it; the wavelength is finally
                   picked at random among those free on the whole path
"""
import random
import sys
from multiprocessing import Pool
import numpy as np

K, MU = 10, 1.0
SIZES = [5, 7, 9]
N_ARRIVALS, WARMUP = 3_000_000, 100_000
LOADS = {0: np.arange(2.0, 4.01, 0.25),        # offered load per link [Erlang]
         1: np.arange(1.25, 3.26, 0.25)}
ALL = (1 << K) - 1


def simulate(args):
    N, a, continuity = args
    random.seed(1000 * N + int(100 * a) + continuity)
    half, L = (N - 1) // 2, 4 * N * N
    arr_rate = a * L / (N / 2) * MU            # mean path length is N/2 hops
    free = [ALL] * L                           # bitmask of free channels per link
    active = []                                # per connection: [(link, channel mask), ...]
    in_use = busy = blocked = 0                # in_use = channels busy network-wide
    rnd, rrange = random.random, random.randrange
    for n in range(N_ARRIVALS + WARMUP):
        # departures until the next arrival (memoryless: any active call may leave)
        while rnd() * (arr_rate + len(active) * MU) >= arr_rate:
            k = rrange(len(active))
            active[k], active[-1] = active[-1], active[k]
            held = active.pop()
            for l, w in held:
                free[l] |= w
            in_use -= len(held)
        if n >= WARMUP:
            busy += in_use

        # source and destination (uniform over the other N*N - 1 nodes)
        i, j = rrange(N), rrange(N)
        di = dj = 0
        while di == 0 and dj == 0:
            di, dj = rrange(-half, half + 1), rrange(-half, half + 1)

        links, avail = [], ALL                 # avail = wavelengths free so far
        while di or dj:
            go_intra = di != 0 and (dj == 0 or rnd() < 0.5)       # first choice
            for attempt in range(2):
                if go_intra:
                    l = 4 * (i * N + j) + (0 if di > 0 else 1)
                else:
                    l = 4 * (i * N + j) + (2 if dj > 0 else 3)
                ok = (avail & free[l]) if continuity else free[l]
                if ok or attempt == 1 or di == 0 or dj == 0:
                    break
                go_intra = not go_intra        # first choice blocked: try the other
            if not ok:
                links = None                   # dropped
                break
            if continuity:
                avail = ok
            links.append(l)
            if go_intra:
                step = 1 if di > 0 else -1
                i, di = (i + step) % N, di - step
            else:
                step = 1 if dj > 0 else -1
                j, dj = (j + step) % N, dj - step

        if links is None:
            blocked += n >= WARMUP
            continue
        if continuity:                         # one wavelength for the whole path
            w = random.choice([1 << b for b in range(K) if avail >> b & 1])
            held = [(l, w) for l in links]
        else:                                  # any free channel on each link
            held = [(l, free[l] & -free[l]) for l in links]
        for l, w in held:
            free[l] ^= w
        active.append(held)
        in_use += len(held)
    return N, a, continuity, busy / (N_ARRIVALS * L * K), blocked / N_ARRIVALS


if __name__ == "__main__":
    jobs = [(N, a, c) for c in (0, 1) for N in SIZES for a in LOADS[c]]
    with Pool() as pool:
        sim = pool.map(simulate, jobs, chunksize=1)
    np.savetxt("torus_adaptive_sim.csv", sim, delimiter=",", fmt="%.6g",
               header="N,offered_load_per_link,continuity,utilisation,blocking")
