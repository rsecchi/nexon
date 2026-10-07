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

## Simulators (`simulations/`)

| Script | What it simulates | Output |
|---|---|---|
| `torus_blocking.py` | First version: no continuity, fixed routing, 7×7. Compares the event simulation, Kelly's rejection sampling, Erlang B and the Erlang fixed point (prints a table) | – |
| `torus_plots.py` | No continuity, fixed routing, loads 2.5–9 Erlang; also draws its own plot against Erlang B and the fixed point | `torus_sim.csv`, `torus_blocking.png` |
| `torus_wdm.py` | Wavelength continuity, fixed routing, random and first-fit assignment, loads 1–8 Erlang; also draws its own plot. Defines `build()` (routes) used by the other scripts | `torus_wdm_sim.csv`, `torus_wdm_blocking.png` |
| `torus_low_sim.py` | Continuity, fixed routing, long runs in the low-blocking region | `torus_low_sim.csv` |
| `torus_free_sim.py` | No continuity, fixed routing, long runs in the low-blocking region | `torus_free_sim.csv` |
| `torus_adaptive_sim.py` | Adaptive routing, with and without continuity | `torus_adaptive_sim.csv` |

## Models and plots (`analysis/`)

| Script | Content | Needs | Plot |
|---|---|---|---|
| `torus_reduced_load.py` | Reduced-load fixed point with continuity: Birman's independent-link model and a version with correlation between adjacent links | `torus_wdm_sim.csv` | `torus_reduced_load.png` |
| `torus_low_blocking.py` | One-pass (no iteration) version of the correlated model for blocking < 1% | `torus_low_sim.csv` | `torus_low_blocking.png` |
| `torus_closed_form.py` | Closed-form version based on "a given set of j wavelengths is idle" and inclusion–exclusion (prints a table) | `torus_low_sim.csv` | – |
| `torus_equivalent_load.py` | Fits Φ and checks ErlangB(Φ·a, K), continuity, fixed routing | `torus_low_sim.csv`, `torus_wdm_sim.csv` | `torus_equivalent_load.png` |
| `torus_compare.py` | Φ with and without continuity, fixed routing | `torus_free_sim.csv`, `torus_low_sim.csv`, `torus_wdm_sim.csv` | `torus_compare.png` |
| `torus_adaptive.py` | Φ for adaptive routing against fixed routing | `torus_adaptive_sim.csv`, `torus_free_sim.csv`, `torus_low_sim.csv` | `torus_adaptive.png` |
| `torus_draw_grid.py` | Picture of the grid with node coordinates and link IDs as the simulators number them (`python3 analysis/torus_draw_grid.py 7`) | – | `torus_grid_3x3.png`, `torus_grid_7x7.png` |

## Reproducing

    python3 simulations/torus_plots.py           # ~2.5 min on 2 cores
    python3 simulations/torus_wdm.py             # ~3 min
    python3 simulations/torus_low_sim.py         # ~7 min
    python3 simulations/torus_free_sim.py        # ~2 min
    python3 simulations/torus_adaptive_sim.py    # ~10 min
    python3 analysis/torus_reduced_load.py
    python3 analysis/torus_low_blocking.py
    python3 analysis/torus_closed_form.py
    python3 analysis/torus_equivalent_load.py
    python3 analysis/torus_compare.py
    python3 analysis/torus_adaptive.py

The CSV files in `results/data` are the results of these runs, so the model and
plot scripts can be run directly without repeating the simulations.

## Caveats

- Φ is fitted to simulation, not derived. It depends on the grid size and on K
  (for 7×7 with continuity: about 1.9, 1.78 and 1.68 for K = 5, 10, 20).
- The correlated-link models are derived here in the spirit of Barry–Humblet and
  Sridharan–Sivarajan; they are not transcriptions of published formulas.
- Everything is for N × N grids with odd N and uniform traffic. With continuity,
  the models assume random wavelength assignment; first-fit blocks less.
