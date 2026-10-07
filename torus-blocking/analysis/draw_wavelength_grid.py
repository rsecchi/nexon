"""
Picture of blocking on a line of nodes, as a grid of wavelengths (rows) against
links (columns). Each existing connection is a bar on one wavelength, covering
the links it uses. A new end-to-end connection needs

  with continuity    : one ROW with no bar in it
  without continuity : no COLUMN completely covered
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PLOTS

K, LINKS = 4, 4                                # wavelengths, links (nodes 0..4)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#d9d8d2"
BAR, FREE, FULL = "#2a78d6", "#1baf7a", "#eb6834"

# each case: list of (wavelength, first link, last link) for the connections in progress
CASES = [
    ("Accepted", "wavelength 3 is free on every link",
     [(1, 0, 1), (2, 2, 2), (4, 1, 3)]),
    ("Blocked with continuity only", "every wavelength is hit; each link still has one free",
     [(1, 0, 0), (2, 2, 3), (3, 1, 1), (4, 3, 3)]),
    ("Blocked in both cases", "link 1–2 has all its wavelengths busy",
     [(1, 0, 1), (2, 1, 1), (3, 1, 2), (4, 1, 3)]),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, (title, note, bars) in zip(axes, CASES):
    busy = {(w, l) for w, a, b in bars for l in range(a, b + 1)}
    free_rows = [w for w in range(1, K + 1) if not any((w, l) in busy for l in range(LINKS))]
    full_cols = [l for l in range(LINKS) if all((w, l) in busy for w in range(1, K + 1))]
    for w in free_rows:                        # a usable wavelength
        ax.add_patch(plt.Rectangle((0, K - w), LINKS, 1, fc=FREE, alpha=0.18, ec="none"))
    for l in full_cols:                        # a full link
        ax.add_patch(plt.Rectangle((l, 0), 1, K, fc=FULL, alpha=0.18, ec="none"))
    for x in range(LINKS + 1):
        ax.plot([x, x], [0, K], color=GRID, lw=1)
    for y in range(K + 1):
        ax.plot([0, LINKS], [y, y], color=GRID, lw=1)
    for w, a, b in bars:                       # connections in progress
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (a + 0.1, K - w + 0.28), b - a + 0.8, 0.44, boxstyle="round,pad=0,rounding_size=0.12",
            fc=BAR, ec="white", lw=1.5, zorder=3))
    for w in range(1, K + 1):
        ax.text(-0.15, K - w + 0.5, f"wavelength {w}", ha="right", va="center", fontsize=9.5,
                color=INK, fontweight="bold" if w in free_rows else "normal")
    for l in range(LINKS):
        ax.text(l + 0.5, -0.22, f"link {l}–{l + 1}", ha="center", va="top", fontsize=9,
                color=INK, fontweight="bold" if l in full_cols else "normal")
    for n in range(LINKS + 1):                 # the line of nodes underneath
        ax.add_patch(plt.Circle((n, -0.95), 0.14, fc="white", ec=INK, lw=1.3, zorder=3))
        ax.text(n, -0.95, str(n), ha="center", va="center", fontsize=8, zorder=4)
    ax.plot([0, LINKS], [-0.95, -0.95], color=INK, lw=1.3, zorder=1)
    ax.text(0, K + 0.62, title, fontsize=12, fontweight="bold", color=INK, va="bottom")
    ax.text(0, K + 0.12, note, fontsize=9.5, color=MUTED, va="bottom", linespacing=1.25)
    ax.set_xlim(-1.35, LINKS + 0.15); ax.set_ylim(-1.3, K + 1.5)
    ax.set_aspect("equal"); ax.axis("off")

fig.legend(handles=[
    matplotlib.patches.Patch(fc=BAR, label="connection in progress (one wavelength, the links it uses)"),
    matplotlib.patches.Patch(fc=FREE, alpha=0.3, label="wavelength free end to end"),
    matplotlib.patches.Patch(fc=FULL, alpha=0.3, label="link with every wavelength busy")],
    loc="lower center", ncol=3, frameon=False, fontsize=9.5)
fig.suptitle("A new connection from node 0 to node 4: what blocks it", x=0.012, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(PLOTS / "wavelength_grid.png", dpi=150, facecolor="white")
