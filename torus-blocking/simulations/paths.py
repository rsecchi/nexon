"""Where the simulation scripts write their results."""
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA, PLOTS = RESULTS / "data", RESULTS / "plots"
