# Blocking in a toroidal ISL grid

Simulations and analytical models for connection blocking in an N × N toroidal
satellite grid (N orbital planes, N satellites per plane, 4 directed
inter-satellite links per node, K channels per link). Connections arrive as a
Poisson process, have uniformly chosen destinations and exponential holding
times, and follow a shortest path.

All scripts are plain Python 3 with `numpy` and `matplotlib`, and can be run from
any directory. Defaults are K = 10 and grids 5×5, 7×7, 9×9; the constants are at
the top of each script.

## Layout

    simulations/     event-driven simulators (write to results/)
    analysis/        analytical models, fitting and plotting (read results/data)
    results/data/    simulated points, one CSV per simulator
    results/plots/   figures

`paths.py` in each script folder defines where results are read and written.

## Main result

Below about 1% blocking, the whole network behaves like one Erlang link carrying
the per-link load scaled by a constant factor Φ:

    blocking ≈ ErlangB(Φ · a, K),    a = λ · N / (8 μ)  (offered load per link)

Φ for K = 10, fitted to simulation below 1% blocking:

| Grid | No continuity, fixed routing | No continuity, adaptive | Continuity, fixed routing | Continuity, adaptive |
|------|------|------|------|------|
| 5×5  | 1.14 | 1.05 | 1.56 | 1.59 |
| 7×7  | 1.19 | 1.08 | 1.78 | 1.93 |
| 9×9  | 1.22 | 1.11 | 1.92 | 2.23 |

"Continuity" means a connection needs the same wavelength on every hop (random
wavelength assignment). "Fixed routing" is intra-plane hops first, then
inter-plane. "Adaptive" picks intra- or inter-plane at random at each node and
tries the other if the first is blocked.

### OBS: the sender picks the wavelength

In all the results above the wavelength is chosen knowing the state of the whole
path. If instead the source picks a wavelength free on its own outgoing link and
every node must reuse it (one-way reservation, no conversion), the loss is far
higher and follows an Erlang formula with ONE channel:

    burst loss ≈ x / (1 + x),    x = Φ₁ · a / K

    Φ₁ = N(n−1)/(n+1)² + n/(n+1) − 2n²/(N(n+1)²),   n = (N−1)/2

Φ₁ = 1.04, 1.46, 1.74 for 5×5, 7×7, 9×9. It is derived from the routing, not
fitted. With K = 10, 1% loss is reached at 0.10, 0.07 and 0.06 Erlang per link.

## Simulators (`simulations/`)

| Script | What it simulates | Output |
|---|---|---|
| `torus_blocking.py` | First version: no continuity, fixed routing, 7×7. Compares the event simulation, Kelly's rejection sampling, Erlang B and the Erlang fixed point (prints a table) | – |
| `torus_plots.py` | No continuity, fixed routing, loads 2.5–9 Erlang; also draws its own plot against Erlang B and the fixed point | `torus_sim.csv`, `torus_blocking.png` |
| `torus_wdm.py` | Wavelength continuity, fixed routing, random and first-fit assignment, loads 1–8 Erlang; also draws its own plot. Defines `build()` (routes) used by the other scripts | `torus_wdm_sim.csv`, `torus_wdm_blocking.png` |
| `torus_low_sim.py` | Continuity, fixed routing, long runs in the low-blocking region | `torus_low_sim.csv` |
| `torus_free_sim.py` | No continuity, fixed routing, long runs in the low-blocking region | `torus_free_sim.csv` |
| `torus_adaptive_sim.py` | Adaptive routing, with and without continuity | `torus_adaptive_sim.csv` |
| `torus_retry_sim.py` | Time correlation of blocking: keeps a clock and, after each dropped request, probes whether the same route is still blocked after a set of delays (7×7, with and without continuity) | `torus_retry_sim.csv` |
| `torus_obs_sim.py` | OBS-style continuity: the sender picks a wavelength free on its own link and the burst is lost at the first hop where it is busy; with and without lost bursts holding the hops already crossed | `torus_obs_sim.csv` |

## Models and plots (`analysis/`)

| Script | Content | Needs | Plot |
|---|---|---|---|
| `torus_reduced_load.py` | Reduced-load fixed point with continuity: Birman's independent-link model and a version with correlation between adjacent links | `torus_wdm_sim.csv` | `torus_reduced_load.png` |
| `torus_low_blocking.py` | One-pass (no iteration) version of the correlated model for blocking < 1% | `torus_low_sim.csv` | `torus_low_blocking.png` |
| `torus_closed_form.py` | Closed-form version based on "a given set of j wavelengths is idle" and inclusion–exclusion (prints a table) | `torus_low_sim.csv` | – |
| `torus_equivalent_load.py` | Fits Φ and checks ErlangB(Φ·a, K), continuity, fixed routing | `torus_low_sim.csv`, `torus_wdm_sim.csv` | `torus_equivalent_load.png` |
| `torus_compare.py` | Φ with and without continuity, fixed routing | `torus_free_sim.csv`, `torus_low_sim.csv`, `torus_wdm_sim.csv` | `torus_compare.png` |
| `torus_adaptive.py` | Φ for adaptive routing against fixed routing | `torus_adaptive_sim.csv`, `torus_free_sim.csv`, `torus_low_sim.csv` | `torus_adaptive.png` |
| `torus_retry.py` | Probability of being dropped again against the retry delay, with the single-link Erlang transient for the no-continuity case | `torus_retry_sim.csv` | `torus_retry.png` |
| `torus_obs.py` | OBS burst loss against Erlang B with one channel at load Φ₁·a/K, with Φ₁ from a closed formula | `torus_obs_sim.csv` | `torus_obs.png` |
| `torus_draw_grid.py` | Picture of the grid with node coordinates and link IDs as the simulators number them (`python3 analysis/torus_draw_grid.py 7`) | – | `torus_grid_3x3.png`, `torus_grid_7x7.png` |
| `draw_wavelength_grid.py` | Explanatory picture: wavelengths against links on a line of nodes, showing what blocks a connection with and without continuity | – | `wavelength_grid.png` |

## Reproducing

    python3 simulations/torus_plots.py           # ~2.5 min on 2 cores
    python3 simulations/torus_wdm.py             # ~3 min
    python3 simulations/torus_low_sim.py         # ~7 min
    python3 simulations/torus_free_sim.py        # ~2 min
    python3 simulations/torus_adaptive_sim.py    # ~10 min
    python3 simulations/torus_retry_sim.py       # ~2 min
    python3 simulations/torus_obs_sim.py         # ~4 min
    python3 analysis/torus_reduced_load.py
    python3 analysis/torus_low_blocking.py
    python3 analysis/torus_closed_form.py
    python3 analysis/torus_equivalent_load.py
    python3 analysis/torus_compare.py
    python3 analysis/torus_adaptive.py
    python3 analysis/torus_retry.py
    python3 analysis/torus_obs.py

The CSV files in `results/data` are the results of these runs, so the model and
plot scripts can be run directly without repeating the simulations.

## Caveats

- Φ is fitted to simulation, not derived. It depends on the grid size and on K
  (for 7×7 with continuity: about 1.9, 1.78 and 1.68 for K = 5, 10, 20).
- The correlated-link models are derived here in the spirit of Barry–Humblet and
  Sridharan–Sivarajan; they are not transcriptions of published formulas.
- Everything is for N × N grids with odd N and uniform traffic. With continuity,
  the models assume random wavelength assignment; first-fit blocks less.
