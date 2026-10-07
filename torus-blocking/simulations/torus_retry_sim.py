"""
Blocking correlation in time: if a request is dropped at time t, how likely is
the SAME request (same source, destination and route) to be dropped at t + delay?

Unlike the other simulators this one keeps a clock. Time is in units of the mean
holding time (1/MU). Whenever a request is dropped, a set of "probes" is
scheduled at t + delay for each delay in DELAYS. A probe only looks at the
network (would this route be blocked now?) and never occupies a channel, so the
probes do not disturb the traffic.

Fixed routing (intra-plane first, then inter-plane).
  continuity = 0 : blocked when some link of the route has all K channels busy
  continuity = 1 : blocked when no wavelength is free on every link of the route
"""
import heapq
import random
from multiprocessing import Pool
import numpy as np
from paths import DATA
from torus_wdm import build, K, MU

N = 7
N_ARRIVALS, WARMUP = 10_000_000, 100_000
DELAYS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
          1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
CASES = [(0, 3.0), (0, 3.75), (1, 2.0), (1, 2.5)]          # (continuity, load per link)
ALL = (1 << K) - 1


def simulate(args):
    continuity, a = args
    random.seed(100 * continuity + int(100 * a))
    routes = build(N)
    R, L = len(routes), 4 * N * N
    arr_rate = a * L / sum(map(len, routes)) * R * MU
    free = [ALL] * L                           # bitmask of free channels per link
    active, probes = [], []                    # probes: heap of (time, delay index, route index)
    hits, tries = [0] * len(DELAYS), [0] * len(DELAYS)
    now, blocked = 0.0, 0

    def is_blocked(r):
        if continuity:
            avail = ALL
            for l in r:
                avail &= free[l]
            return avail == 0
        return any(free[l] == 0 for l in r)

    n = 0
    while n < N_ARRIVALS + WARMUP:
        rate = arr_rate + len(active) * MU
        now += random.expovariate(rate)        # time of the next event
        while probes and probes[0][0] <= now:  # probes falling before it see the current state
            _, d, ri = heapq.heappop(probes)
            tries[d] += 1
            hits[d] += is_blocked(routes[ri])
        if random.random() * rate >= arr_rate:                 # departure
            k = random.randrange(len(active))
            active[k], active[-1] = active[-1], active[k]
            for l, w in active.pop():
                free[l] |= w
            continue
        n += 1                                                 # arrival
        ri = random.randrange(R)
        r = routes[ri]
        if is_blocked(r):
            if n > WARMUP:
                blocked += 1
                for d, delay in enumerate(DELAYS):
                    heapq.heappush(probes, (now + delay, d, ri))
            continue
        if continuity:                         # one random wavelength free on the whole route
            avail = ALL
            for l in r:
                avail &= free[l]
            w = random.choice([1 << b for b in range(K) if avail >> b & 1])
            held = [(l, w) for l in r]
        else:                                  # any free channel on each link
            held = [(l, free[l] & -free[l]) for l in r]
        for l, w in held:
            free[l] ^= w
        active.append(held)
    return [(continuity, a, blocked / N_ARRIVALS, delay, t, h / t if t else np.nan)
            for delay, t, h in zip(DELAYS, tries, hits)]


if __name__ == "__main__":
    with Pool() as pool:
        rows = [row for result in pool.map(simulate, CASES, chunksize=1) for row in result]
    np.savetxt(DATA / "torus_retry_sim.csv", rows, delimiter=",", fmt="%.6g",
               header="continuity,offered_load_per_link,blocking,delay,probes,blocked_again")
