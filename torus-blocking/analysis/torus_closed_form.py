"""
Closed-form blocking for the N x N toroidal grid with wavelength continuity,
valid at low blocking (no load reduction).

All K wavelengths are equivalent, so everything can be written in terms of
"a given set of j wavelengths is idle":

  g[j]  = P(the set is idle on one link)                      (Erlang occupancy)
  r[j]  = P(the set is idle on the next link | idle on this one)

  B(path) = sum_j (-1)^j C(K,j) g[j] r_straight[j]^s r_turn[j]^t

with s straight transitions and t (0 or 1) turns on the path.

r[j] comes from a two-link model: connections on both links (load kappa*a, same
wavelength on both), connections on one link only (load (1-kappa)*a each), all
placed on wavelengths uniformly at random.
"""
from collections import Counter
from math import comb, factorial
import numpy as np
from paths import DATA
from torus_wdm import build, K, SIZES


def g_single(a):
    p = np.array([a**k / factorial(k) for k in range(K + 1)])
    p /= p.sum()                                           # Erlang: p[k], k busy
    return np.array([sum(p[k] * comb(K - j, k) / comb(K, k) for k in range(K - j + 1))
                     for j in range(K + 1)])


def r_pair(a, kappa):
    both, one = np.zeros(K + 1), np.zeros(K + 1)
    for c in range(K + 1):                                 # connections on both links
        for x1 in range(K - c + 1):                        # on link 1 only
            for x2 in range(K - c + 1):                    # on link 2 only
                w = ((kappa * a)**c / factorial(c) * ((1 - kappa) * a)**(x1 + x2)
                     / factorial(x1) / factorial(x2))
                for j in range(K - c + 1):
                    free = comb(K - c, j) / comb(K, j)     # set avoids the shared ones
                    f1 = comb(K - c - j, x1) / comb(K - c, x1)
                    f2 = comb(K - c - j, x2) / comb(K - c, x2)
                    both[j] += w * free * f1 * f2
                    one[j] += w * free * f1
    return both / one


def blocking(N, a):
    """Mean connection blocking at offered load a [Erlang per link]."""
    half = (N - 1) // 2
    g = g_single(a)
    rs = r_pair(a, (half - 1) / (half + 1))                # going straight
    rt = r_pair(a, 2 * half / (N * (half + 1)))            # intra- to inter-plane turn
    sign = np.array([(-1)**j * comb(K, j) for j in range(K + 1)])
    kinds = Counter((sum(l % 4 < 2 for l in r), sum(l % 4 >= 2 for l in r)) for r in build(N))
    total = 0.0
    for (h1, h2), count in kinds.items():
        s, t = max(h1 - 1, 0) + max(h2 - 1, 0), int(h1 > 0 and h2 > 0)
        total += count * (sign * g * rs**s * rt**t).sum()
    return total / sum(kinds.values())


if __name__ == "__main__":
    sim = np.loadtxt(DATA / "torus_low_sim.csv", delimiter=",")   # produced by torus_low_sim.py
    print("  N  load | simulated  closed form  ratio")
    for N, a, _, _, blk in sim:
        b = blocking(int(N), a)
        print(f"{int(N):3d} {a:5.2f} | {blk:9.2e} {b:11.2e} {b / blk:6.2f}")
