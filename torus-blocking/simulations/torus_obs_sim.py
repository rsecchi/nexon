"""
OBS-style wavelength continuity: the SENDER picks the wavelength.

The source node knows only its own outgoing link. It picks, at random, one of
the wavelengths free on that link, and every node along the route must use the
same one. The burst is lost at the first hop where that wavelength is busy
(or at the source if its link has no free wavelength at all).

  occupy = 0 : a lost burst uses no resources
  occupy = 1 : a lost burst still holds its wavelength, for its whole duration,
               on the hops it crossed before being dropped (one-way reservation)

Fixed routing (intra-plane first, then inter-plane). The load a is the offered
load per link that the traffic would produce if every burst completed its route.
"""
import random
from multiprocessing import Pool
import numpy as np
from paths import DATA
from torus_wdm import build, K, MU, SIZES

N_ARRIVALS, WARMUP = 2_000_000, 100_000
LOADS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0]    # Erlang per link
ALL = (1 << K) - 1


def simulate(args):
    N, a, occupy = args
    random.seed(1000 * N + int(1000 * a) + occupy)
    routes = build(N)
    R, L = len(routes), 4 * N * N
    arr_rate = a * L / sum(map(len, routes)) * R * MU
    free = [ALL] * L                           # bitmask of free wavelengths per link
    active = []                                # per burst: (links held, wavelength mask)
    in_use = busy = lost = 0
    for n in range(N_ARRIVALS + WARMUP):
        # departures until the next arrival (memoryless: any active burst may end)
        while random.random() * (arr_rate + len(active) * MU) >= arr_rate:
            k = random.randrange(len(active))
            active[k], active[-1] = active[-1], active[k]
            held, w = active.pop()
            for l in held:
                free[l] |= w
            in_use -= len(held)
        if n >= WARMUP:
            busy += in_use
        r = routes[random.randrange(R)]
        local = free[r[0]]                     # all the sender can see
        if local == 0:
            lost += n >= WARMUP
            continue
        w = random.choice([1 << b for b in range(K) if local >> b & 1])
        crossed = 0
        for l in r:                            # same wavelength on every hop
            if not free[l] & w:
                break
            crossed += 1
        if crossed < len(r):
            lost += n >= WARMUP
            if not occupy:
                continue
        held = r[:crossed]
        for l in held:
            free[l] ^= w
        active.append((held, w))
        in_use += crossed
    return N, a, occupy, busy / (N_ARRIVALS * L * K), lost / N_ARRIVALS


if __name__ == "__main__":
    jobs = [(N, a, occupy) for occupy in (0, 1) for N in SIZES for a in LOADS]
    with Pool() as pool:
        sim = pool.map(simulate, jobs, chunksize=1)
    np.savetxt(DATA / "torus_obs_sim.csv", sim, delimiter=",", fmt="%.6g",
               header="N,offered_load_per_link,occupy,utilisation,burst_loss")
