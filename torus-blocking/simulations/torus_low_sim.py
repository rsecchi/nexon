"""Long simulations in the low-blocking region (random wavelength assignment)."""
from multiprocessing import Pool
import numpy as np
from paths import DATA
import torus_wdm

torus_wdm.N_ARRIVALS = 5_000_000
LOADS = np.arange(1.25, 3.01, 0.25)            # offered load per link [Erlang]

if __name__ == "__main__":
    jobs = [(N, a, 0) for N in torus_wdm.SIZES for a in LOADS]
    with Pool() as pool:
        sim = np.array(pool.map(torus_wdm.simulate, jobs, chunksize=1))
    np.savetxt(DATA / "torus_low_sim.csv", sim, delimiter=",", fmt="%.6g",
               header="N,offered_load_per_link,first_fit,utilisation,blocking")
