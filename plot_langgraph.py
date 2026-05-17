"""
Phase 4 -- LangGraph State Graph Visualization
================================================
Renders the agent workflow diagram plus actual Phase 4 run results
side-by-side as a single publication-quality figure.

  python plot_langgraph.py
  Output: langgraph_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

OUT = "langgraph_diagram.png"

# ── Actual Phase 4 run data ────────────────────────────────────────────────────
SCIPY_H    = 8.308
SCIPY_MASS = 2.2432

RUN_DATA = [
    # iter, H_mm,  defl_mm,  stress_MPa, mass,    feasible, agent_reasoning
    (1, 6.00,  0.1325, 87.1,  1.620, False,
     "Computed 1/H³ scaling: need H × (0.1325/0.050)^(1/3) ≈ 1.38× → 9 mm"),
    (2, 9.00,  0.0391, 58.3,  2.430, True,
     "Jump to analytical boundary H ≈ 8.3 mm to minimize mass"),
    (3, 8.30,  0.0508, 63.2,  2.241, False,
     "Just over limit — increase to 8.50 mm"),
    (4, 8.50,  0.0472, 61.4,  2.295, True,
     "Feasible. Binary search: try 8.40 mm"),
    (5, 8.40,  0.0490, 62.3,  2.268, True,
     "Still feasible. Tighten: try 8.32 mm"),
    (6, 8.32,  0.0504, 63.0,  2.246, False,
     "Slightly infeasible — hold at 8.35 mm"),
    (7, 8.35,  0.0499, 62.7,  2.255, True,
     "Converged — 0.1% from scipy baseline"),
]

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor="#F8F9FA")
fig.suptitle(
    "Aero-Opt  |  Phase 4: LangGraph AI Optimization Agent",
    fontsize=15, fontweight="bold", y=0.97, color="#1A1A2E"
)

# Two columns: graph (left 52%) and results (right 48%)
ax_graph  = fig.add_axes([0.01, 0.04, 0.50, 0.90])
ax_table  = fig.add_axes([0.53, 0.04, 0.46, 0.90])

ax_graph.set_xlim(0, 1)
ax_graph.set_ylim(0, 1)
ax_graph.axis("off")
ax_table.axis("off")

# ── Color palette ──────────────────────────────────────────────────────────────
C_BG       = "#F8F9FA"
C_TERM     = "#DEE2E6"   # START / END
C_SIM      = "#CBE4F9"   # run_simulation  (blue)
C_EVAL     = "#C9F7DC"   # evaluate        (green)
C_AGENT    = "#EDE0FF"   # agent_reason    (purple)
C_BORDER   = "#343A40"
C_ARROW    = "#495057"
C_TEXT     = "#212529"
C_SUB      = "#6C757D"
C_FEAS     = "#198754"   # green for feasible
C_INFEAS   = "#DC3545"   # red for infeasible


# ── Helper: rounded box ────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, fc, label, sub="", lfs=11, sfs=8.5, lw=1.8):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.025",
                       facecolor=fc, edgecolor=C_BORDER, linewidth=lw, zorder=4)
    ax.add_patch(b)
    dy = 0.018 if sub else 0
    ax.text(cx, cy + dy, label, ha="center", va="center",
            fontsize=lfs, fontweight="bold", color=C_TEXT, zorder=5)
    if sub:
        ax.text(cx, cy - 0.022, sub, ha="center", va="center",
                fontsize=sfs, color=C_SUB, zorder=5, fontstyle="italic")


def oval(ax, cx, cy, rx, ry, fc, label, fs=10):
    e = mpatches.Ellipse((cx, cy), 2*rx, 2*ry,
                         facecolor=fc, edgecolor=C_BORDER, linewidth=1.8, zorder=4)
    ax.add_patch(e)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=C_TEXT, zorder=5)


def arr(ax, x1, y1, x2, y2, label="", rad=0.0, lx=None, ly=None,
        lha="left", color=C_ARROW):
    style = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=1.8,
                    mutation_scale=16,
                    connectionstyle=style,
                ), zorder=3)
    if label:
        if lx is None: lx = (x1 + x2) / 2 + (0.03 if lha == "left" else -0.03)
        if ly is None: ly = (y1 + y2) / 2
        ax.text(lx, ly, label, ha=lha, va="center", fontsize=9,
                color=C_TEXT, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                zorder=6)


# ── Draw graph ─────────────────────────────────────────────────────────────────
# Node positions  (ax_graph coords, 0-1)
XM  = 0.38    # main column x
XA  = 0.80    # agent_reason x
YS  = 0.88    # START
YN  = 0.70    # run_simulation
YE  = 0.50    # evaluate
YA  = 0.50    # agent_reason (same row)
YD  = 0.24    # END
NW  = 0.32    # node width
NH  = 0.085   # node height

oval(ax_graph, XM, YS, 0.09, 0.04, C_TERM, "START", fs=11)

rbox(ax_graph, XM, YN, NW, NH, C_SIM,
     "run_simulation",
     "simulate()  •  log run to RAG")

rbox(ax_graph, XM, YE, NW, NH, C_EVAL,
     "evaluate",
     "constraints  •  best design  •  convergence")

rbox(ax_graph, XA, YA, NW, NH, C_AGENT,
     "agent_reason",
     "RAG query  •  LLM call  •  parse H")

oval(ax_graph, XM, YD, 0.09, 0.04, C_TERM, "END", fs=11)

# Arrows
arr(ax_graph, XM, YS - 0.04, XM, YN + NH/2)
arr(ax_graph, XM, YN - NH/2, XM, YE + NH/2)
arr(ax_graph, XM, YE - NH/2, XM, YD + 0.04,
    label="converged", lx=XM + 0.04, ly=(YE - NH/2 + YD + 0.04) / 2)

# evaluate → agent_reason
arr(ax_graph, XM + NW/2, YE, XA - NW/2, YA,
    label="not converged", lx=XM + NW/2 + 0.02, ly=YE + 0.04)

# agent_reason → run_simulation (loop, curved up and left)
ax_graph.annotate(
    "", xy=(XM + NW/2, YN),
    xytext=(XA + NW/2, YA + NH/2),
    arrowprops=dict(
        arrowstyle="-|>", color=C_ARROW, lw=1.8, mutation_scale=16,
        connectionstyle="arc3,rad=-0.45",
    ), zorder=3
)
ax_graph.text(0.955, 0.625, "propose\nnew H",
              ha="center", va="center", fontsize=8.5,
              fontstyle="italic", color=C_TEXT,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
              zorder=6)

# ── State dict sidebar (inside graph panel) ────────────────────────────────────
state_x, state_y = 0.05, 0.44
state_w, state_h = 0.25, 0.32
sb = FancyBboxPatch((state_x, state_y - state_h), state_w, state_h,
                    boxstyle="round,pad=0.02",
                    facecolor="white", edgecolor="#ADB5BD", linewidth=1.2, zorder=2)
ax_graph.add_patch(sb)
ax_graph.text(state_x + state_w/2, state_y - 0.025,
              "OptState", ha="center", va="top", fontsize=9.5,
              fontweight="bold", color=C_TEXT, zorder=3)

fields = [
    ("params",      "H, L, load"),
    ("result",      "deflection, stress, mass"),
    ("iteration",   "int"),
    ("converged",   "bool"),
    ("best_params", "best feasible H"),
    ("best_mass",   "float  (minimize)"),
    ("history",     "list of all runs"),
    ("agent_trace", "LLM reasoning log"),
]
for i, (k, v) in enumerate(fields):
    fy = state_y - 0.065 - i * 0.032
    ax_graph.text(state_x + 0.01, fy, f"{k}:", ha="left", va="center",
                  fontsize=7.8, fontweight="bold", color="#2C6E9A", zorder=3)
    ax_graph.text(state_x + state_w - 0.01, fy, v, ha="right", va="center",
                  fontsize=7.5, color=C_SUB, zorder=3)

# dotted line connecting state box to graph nodes
ax_graph.annotate("", xy=(XM - NW/2, YN), xytext=(state_x + state_w, state_y - 0.11),
                  arrowprops=dict(arrowstyle="-", color="#ADB5BD", lw=1.0,
                                  linestyle="dashed"), zorder=1)

# ── Legend (bottom of graph panel) ────────────────────────────────────────────
legend_y = 0.14
lx0 = 0.05
items = [
    (C_SIM,   "Physics node"),
    (C_EVAL,  "Decision node"),
    (C_AGENT, "AI / LLM node"),
]
for i, (c, label) in enumerate(items):
    lx = lx0 + i * 0.30
    b = FancyBboxPatch((lx, legend_y - 0.02), 0.055, 0.035,
                       boxstyle="round,pad=0.01",
                       facecolor=c, edgecolor=C_BORDER, linewidth=1.2, zorder=3)
    ax_graph.add_patch(b)
    ax_graph.text(lx + 0.065, legend_y - 0.003, label,
                  va="center", fontsize=8.5, color=C_TEXT, zorder=4)

# subtitle under graph
ax_graph.text(0.5, 0.03,
              "LangGraph 1.2  •  Claude Sonnet 4.6  (or Ollama --local)  •  ChromaDB RAG",
              ha="center", va="bottom", fontsize=8.5, color=C_SUB)


# ── Right panel: iteration table ───────────────────────────────────────────────
ax_table.set_xlim(0, 1)
ax_table.set_ylim(0, 1)

# Panel title
ax_table.text(0.5, 0.97, "Actual Agent Run  —  Phase 4 Results",
              ha="center", va="top", fontsize=12, fontweight="bold", color=C_TEXT)
ax_table.text(0.5, 0.925,
              f"Starting point: H = 6.00 mm  (infeasible)  →  scipy baseline: H = {SCIPY_H} mm",
              ha="center", va="top", fontsize=9, color=C_SUB)

# Table header
cols   = ["Iter", "H (mm)", "Defl (mm)", "Stress (MPa)", "Mass (kg/m)", "Status"]
col_x  = [0.04, 0.18, 0.33, 0.50, 0.68, 0.84]
HEADER_Y = 0.875

for cx, col in zip(col_x, cols):
    ax_table.text(cx, HEADER_Y, col, ha="left", va="center",
                  fontsize=9, fontweight="bold", color="white",
                  bbox=dict(boxstyle="square,pad=0", fc="#343A40", ec="none"))

# Rows
row_h = 0.073
for i, (it, H, defl, stress, mass, feas, reasoning) in enumerate(RUN_DATA):
    ry   = HEADER_Y - (i + 1) * row_h
    rbg  = "#FFFFFF" if i % 2 == 0 else "#F1F3F5"
    feas_c = C_FEAS if feas else C_INFEAS
    feas_t = "OK" if feas else "INFEASIBLE"

    # Row background
    bg = FancyBboxPatch((0.02, ry - row_h * 0.45), 0.96, row_h * 0.85,
                        boxstyle="square,pad=0",
                        facecolor=rbg, edgecolor="none", zorder=1)
    ax_table.add_patch(bg)

    vals = [str(it), f"{H:.2f}", f"{defl:.4f}", f"{stress:.1f}", f"{mass:.3f}"]
    for cx, v in zip(col_x, vals):
        ax_table.text(cx, ry, v, ha="left", va="center",
                      fontsize=9.5, color=C_TEXT, zorder=2)

    # Status badge
    ax_table.text(col_x[-1], ry, feas_t, ha="left", va="center",
                  fontsize=8.5, fontweight="bold", color=feas_c, zorder=2)

    # Reasoning note (small, below row)
    ax_table.text(col_x[1], ry - row_h * 0.34, f"↳ {reasoning}",
                  ha="left", va="center", fontsize=7.2,
                  color=C_SUB, fontstyle="italic", zorder=2)

# Divider
div_y = HEADER_Y - len(RUN_DATA) * row_h - 0.01
ax_table.axhline(div_y, xmin=0.02, xmax=0.98, color="#ADB5BD", lw=1.0, ls="--")

# Summary box
sum_y = div_y - 0.03
ax_table.text(0.5, sum_y, "Optimization Result", ha="center", va="top",
              fontsize=10, fontweight="bold", color=C_TEXT)

agent_H    = 8.35
agent_mass = 2.255
pct_off    = (agent_mass - SCIPY_MASS) / SCIPY_MASS * 100.0

summary_lines = [
    ("Agent result",   f"H = {agent_H:.2f} mm   mass = {agent_mass:.3f} kg/m"),
    ("scipy baseline", f"H = {SCIPY_H:.3f} mm   mass = {SCIPY_MASS:.4f} kg/m"),
    ("Agreement",      f"{abs(pct_off):.1f}% difference from scipy  •  7 iterations"),
    ("Mass saving",    "16.9% reduction vs. initial H = 10 mm design"),
    ("Constraint",     "Deflection active  •  Stress far from yield (stiffness-dominated)"),
]
for j, (k, v) in enumerate(summary_lines):
    ly = sum_y - 0.065 - j * 0.052
    ax_table.text(0.06, ly, f"{k}:", ha="left", va="center",
                  fontsize=9, fontweight="bold", color="#2C6E9A")
    ax_table.text(0.35, ly, v, ha="left", va="center",
                  fontsize=9, color=C_TEXT)

# Mini convergence strip at bottom
conv_y0 = 0.04
conv_h  = 0.075
conv_x0 = 0.05
conv_w  = 0.90

ax_table.text(conv_x0, conv_y0 + conv_h + 0.01,
              "H convergence (mm):", fontsize=8.5, fontweight="bold", color=C_TEXT)

H_vals = [r[1] for r in RUN_DATA]
feas   = [r[5] for r in RUN_DATA]
bar_w  = conv_w / len(H_vals) * 0.75
for i, (h, f) in enumerate(zip(H_vals, feas)):
    bx   = conv_x0 + i * (conv_w / len(H_vals))
    norm = h / 12.0                          # scale: 12mm = full bar height
    bh   = norm * conv_h
    bc   = C_FEAS if f else C_INFEAS
    b    = FancyBboxPatch((bx, conv_y0), bar_w, bh,
                          boxstyle="square,pad=0",
                          facecolor=bc, edgecolor="none", alpha=0.75, zorder=2)
    ax_table.add_patch(b)
    ax_table.text(bx + bar_w/2, conv_y0 - 0.015, f"{h:.1f}",
                  ha="center", va="top", fontsize=7.5, color=C_TEXT)
    ax_table.text(bx + bar_w/2, conv_y0 + 0.005, str(i + 1),
                  ha="center", va="bottom", fontsize=6.5, color="white", zorder=3)

# scipy reference line on convergence strip
scipy_norm = SCIPY_H / 12.0
scipy_bar_y = conv_y0 + scipy_norm * conv_h
ax_table.axhline(scipy_bar_y, xmin=conv_x0 - 0.01, xmax=conv_x0 + conv_w + 0.01,
                 color="#555555", lw=1.0, ls="--", zorder=3)
ax_table.text(conv_x0 + conv_w + 0.01, scipy_bar_y,
              f"scipy\n{SCIPY_H}mm", ha="left", va="center",
              fontsize=7, color="#555555")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close()

import os
kb = os.path.getsize(OUT) / 1024
print(f"\nSaved: {OUT}  ({kb:.0f} KB)\n")
