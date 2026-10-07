"""Result folders, and access to the simulation scripts (for build(), K, ...)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, PLOTS = ROOT / "results" / "data", ROOT / "results" / "plots"
sys.path.insert(0, str(ROOT / "simulations"))
