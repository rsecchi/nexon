"""
Picture of the toroidal grid as the simulators number it.

  node (i, j)  : satellite i (row) in orbital plane j (column)
  node index   : i * N + j
  link id      : 4 * (node index) + direction, for the link LEAVING that node
                 direction 0: to satellite i+1 (intra-plane, drawn downwards)
                           1: to satellite i-1 (intra-plane, upwards)
                           2: to plane j+1     (inter-plane, to the right)
                           3: to plane j-1     (inter-plane, to the left)

Usage:  python3 torus_draw_grid.py [N]      (default N = 3)
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import PLOTS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
INK, MUTED = "#0b0b0b", "#898781"
INTRA, INTER = "#2a78d6", "#eb6834"
STEP = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}     # direction -> (di, dj)
R, OFF = 0.17, 0.07                                        # node radius, lane offset

fig, ax = plt.subplots(figsize=(2.6 * N + 1.5, 2.6 * N + 1.5))
for i in range(N):
    for j in range(N):
        x, y = j, -i                                       # rows go down the page
        ax.add_patch(plt.Circle((x, y), R, fc="white", ec=INK, lw=1.4, zorder=3))
        ax.text(x, y + 0.035, f"({i},{j})", ha="center", va="center", fontsize=9, zorder=4)
        ax.text(x, y - 0.075, f"n{i * N + j}", ha="center", va="center", fontsize=7, color=MUTED, zorder=4)
        for d, (di, dj) in STEP.items():
            link = 4 * (i * N + j) + d
            colour = INTRA if d < 2 else INTER
            ux, uy = dj, -di                               # unit vector of travel on the page
            px, py = uy, -ux                               # right-hand side of travel
            wraps = not (0 <= i + di < N and 0 <= j + dj < N)
            x0, y0 = x + ux * R + px * OFF, y + uy * R + py * OFF
            length = (0.46 if wraps else 1.0) - 2 * R + (R if wraps else 0)
            ax.annotate("", xy=(x0 + ux * length, y0 + uy * length), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.3,
                                        ls=(0, (3, 2)) if wraps else "-", shrinkA=0, shrinkB=0))
            lx, ly = x0 + ux * 0.16 + px * 0.075, y0 + uy * 0.16 + py * 0.075
            ax.text(lx, ly, str(link), ha="center", va="center", fontsize=8.5, color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
            if wraps:                                      # say where the wrap-around link lands
                ti, tj = (i + di) % N, (j + dj) % N
                ex, ey = x0 + ux * (length + 0.09) + px * 0.02, y0 + uy * (length + 0.09) + py * 0.02
                ax.text(ex, ey, f"to ({ti},{tj})", ha="center", va="center", fontsize=7, color=MUTED,
                        rotation=90 if d < 2 else 0)

for j in range(N):
    ax.text(j, 0.95, f"plane j = {j}", ha="center", fontsize=10, color=INK)
for i in range(N):
    ax.text(-0.95, -i, f"satellite\ni = {i}", ha="center", va="center", fontsize=10, color=INK)
ax.plot([], [], color=INTRA, lw=2, label="intra-plane link (directions 0 and 1)")
ax.plot([], [], color=INTER, lw=2, label="inter-plane link (directions 2 and 3)")
ax.plot([], [], color=MUTED, lw=1.3, ls=(0, (3, 2)), label="dashed: wraps round to the opposite edge")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=9)
ax.set_title(f"{N}×{N} toroidal grid: link id = 4 × (i·N + j) + direction, written next to the node the link leaves",
             loc="left", fontsize=11, fontweight="bold")
ax.set_xlim(-1.25, N - 0.25); ax.set_ylim(-(N - 1) - 0.8, 1.1)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(PLOTS / f"torus_grid_{N}x{N}.png", dpi=150, facecolor="white")
