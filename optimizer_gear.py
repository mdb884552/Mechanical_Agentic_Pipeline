"""
Gear Optimizer -- scipy SLSQP baseline
========================================
Finds minimum-mass spur gear (N=15 teeth) subject to AGMA
bending and contact stress constraints.

Design variables: m (module, mm), b (face width, mm)
Objective       : minimize mass (kg) = rho * pi/4 * (m*N)^2 * b
Constraints     : sigma_b <= 180 MPa, sigma_c <= 550 MPa

Establishes deterministic ground truth before the AI agent runs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import minimize

from simulate_gear import (simulate, _feasible, nearest_std_module,
                           SIGMA_B_ALLOW, SIGMA_C_ALLOW, N_TEETH,
                           STD_MODULES, RHO)

TORQUE  = 50.0                          # N*m
M_BOUNDS = (1.5, 6.0)
B_BOUNDS = (10.0, 80.0)

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict = {}

def _sim(m: float, b: float) -> dict:
    key = (m, b)
    if key not in _cache:
        _cache[key] = simulate({'m': m, 'b': b, 'torque': TORQUE})
    return _cache[key]


# ── Objective and constraints ──────────────────────────────────────────────────
def objective(x):
    return _sim(x[0], x[1])['mass']

def c_bending(x):
    return SIGMA_B_ALLOW - _sim(x[0], x[1])['bending_stress']   # >= 0

def c_contact(x):
    return SIGMA_C_ALLOW - _sim(x[0], x[1])['contact_stress']   # >= 0

CONSTRAINTS = [
    {'type': 'ineq', 'fun': c_bending},
    {'type': 'ineq', 'fun': c_contact},
]

# ── Run optimizer ──────────────────────────────────────────────────────────────
history = []

def callback(x):
    r = _sim(x[0], x[1])
    history.append({
        'm': x[0], 'b': x[1],
        'mass': r['mass'],
        'sigma_b': r['bending_stress'],
        'sigma_c': r['contact_stress'],
        'feasible': _feasible(r),
    })

x0  = np.array([2.5, 30.0])    # infeasible starting point (sigma_c >> limit)
res = minimize(
    objective, x0,
    method='SLSQP',
    bounds=[M_BOUNDS, B_BOUNDS],
    constraints=CONSTRAINTS,
    callback=callback,
    options={'ftol': 1e-8, 'maxiter': 200, 'disp': False},
)

m_opt  = res.x[0]
b_opt  = res.x[1]
r_opt  = _sim(m_opt, b_opt)
m_std  = nearest_std_module(m_opt)
r_std  = simulate({'m': m_std, 'b': b_opt, 'torque': TORQUE})

# ── Print summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Gear Optimizer -- scipy SLSQP")
print(f"{'='*55}")
print(f"  Problem : N={N_TEETH} teeth, T={TORQUE} N*m, AISI 4140 (250 HB)")
print(f"  Start   : m={x0[0]:.1f} mm  b={x0[1]:.1f} mm")
print()
print(f"  --- Continuous optimum ---")
print(f"  m     = {m_opt:.3f} mm      b     = {b_opt:.3f} mm")
print(f"  d     = {r_opt['pitch_diam_mm']:.1f} mm")
print(f"  sigma_b = {r_opt['bending_stress']:.1f} MPa  (limit {SIGMA_B_ALLOW} MPa)")
print(f"  sigma_c = {r_opt['contact_stress']:.1f} MPa  (limit {SIGMA_C_ALLOW} MPa)")
print(f"  mass    = {r_opt['mass']*1000:.1f} g")
print(f"  Iterations: {len(history)}")
print()
print(f"  --- Snapped to standard module m={m_std} mm ---")
print(f"  b     = {b_opt:.1f} mm   (face width unchanged)")
print(f"  sigma_b = {r_std['bending_stress']:.1f} MPa  "
      f"{'[OK]' if r_std['bending_stress'] <= SIGMA_B_ALLOW else '[VIOLATED]'}")
print(f"  sigma_c = {r_std['contact_stress']:.1f} MPa  "
      f"{'[OK]' if r_std['contact_stress'] <= SIGMA_C_ALLOW else '[VIOLATED]'}")
print(f"  mass    = {r_std['mass']*1000:.1f} g")
print(f"{'='*55}\n")

# ── Visualization ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6), facecolor="#F8F9FA")
fig.suptitle(
    f"Gear Optimization — scipy SLSQP Baseline  "
    f"(N={N_TEETH} teeth, T={TORQUE} N·m, AISI 4140)",
    fontsize=13, fontweight="bold", y=1.01
)

axes = fig.subplots(1, 3)

# ── Panel 1: Constraint map in (m, b) space ────────────────────────────────────
ax = axes[0]
m_grid = np.linspace(1.5, 6.0, 120)
b_grid = np.linspace(10,  80,  120)
MM, BB = np.meshgrid(m_grid, b_grid)

SB = np.vectorize(lambda m, b: simulate({'m': m, 'b': b})['bending_stress'])(MM, BB)
SC = np.vectorize(lambda m, b: simulate({'m': m, 'b': b})['contact_stress'])(MM, BB)
MASS_G = np.vectorize(lambda m, b: simulate({'m': m, 'b': b})['mass'] * 1000)(MM, BB)

infeasible = (SB > SIGMA_B_ALLOW) | (SC > SIGMA_C_ALLOW)
ax.contourf(MM, BB, np.where(infeasible, 1, 0),
            levels=[0.5, 1.5], colors=["#FFCCCC"], alpha=0.5)

mass_levels = [500, 600, 700, 800, 1000, 1300, 1700]
cs_mass = ax.contour(MM, BB, MASS_G, levels=mass_levels,
                     colors=["#888888"], linewidths=0.8, linestyles="dashed")
ax.clabel(cs_mass, fmt="%d g", fontsize=7.5, inline=True)

cs_b = ax.contour(MM, BB, SB, levels=[SIGMA_B_ALLOW],
                  colors=["#D62828"], linewidths=2.0)
cs_c = ax.contour(MM, BB, SC, levels=[SIGMA_C_ALLOW],
                  colors=["#1D6FA4"], linewidths=2.0)

ax.clabel(cs_b, fmt=f"σ_b = {SIGMA_B_ALLOW:.0f} MPa", fontsize=8, inline=True)
ax.clabel(cs_c, fmt=f"σ_c = {SIGMA_C_ALLOW:.0f} MPa", fontsize=8, inline=True)

# Optimizer trajectory
if history:
    hm = [x0[0]] + [h['m'] for h in history]
    hb = [x0[1]] + [h['b'] for h in history]
    ax.plot(hm, hb, "o-", color="#2CA02C", ms=5, lw=1.5,
            label="SLSQP path", zorder=5)
    ax.plot(hm[0], hb[0], "s", color="#FF7F0E", ms=9, zorder=6, label="Start")

ax.plot(m_opt, b_opt, "*", color="#2CA02C", ms=16, zorder=7, label=f"Optimum")
ax.plot(m_std, b_opt, "D", color="#9467BD", ms=10, zorder=7,
        label=f"m={m_std} mm (std)")

ax.set_xlabel("Module  m  (mm)", fontsize=10)
ax.set_ylabel("Face width  b  (mm)", fontsize=10)
ax.set_title("Design Space — Constraint Map", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")

infeasible_patch = mpatches.Patch(color="#FFCCCC", alpha=0.5, label="Infeasible")
ax.legend(handles=ax.get_legend_handles_labels()[0] + [infeasible_patch],
          labels=ax.get_legend_handles_labels()[1] + ["Infeasible"], fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 2: Gear schematic ────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_aspect("equal")
ax2.axis("off")

m_s  = m_std
d_p  = m_s * N_TEETH
d_a  = m_s * (N_TEETH + 2)
d_r  = m_s * (N_TEETH - 2.5)
r_p  = d_p / 2; r_a = d_a / 2; r_r = d_r / 2

# Draw gear circles
theta = np.linspace(0, 2*np.pi, 400)
ax2.plot(r_a*np.cos(theta), r_a*np.sin(theta), 'k-', lw=2.0, label="Outside")
ax2.plot(r_p*np.cos(theta), r_p*np.sin(theta), 'b--', lw=1.2, label="Pitch")
ax2.plot(r_r*np.cos(theta), r_r*np.sin(theta), 'gray', lw=0.8, ls=":", label="Root")

# Draw simplified teeth as rectangular spokes
for i in range(N_TEETH):
    angle = i * 2*np.pi / N_TEETH
    half  = np.pi / N_TEETH * 0.5
    pts = np.array([
        (r_r * np.cos(angle - half*0.85), r_r * np.sin(angle - half*0.85)),
        (r_a * np.cos(angle - half*0.55), r_a * np.sin(angle - half*0.55)),
        (r_a * np.cos(angle + half*0.55), r_a * np.sin(angle + half*0.55)),
        (r_r * np.cos(angle + half*0.85), r_r * np.sin(angle + half*0.85)),
    ])
    poly = plt.Polygon(pts, facecolor="#DDEEFC", edgecolor="#555555", lw=0.7, zorder=3)
    ax2.add_patch(poly)
ax2.fill_between(r_r*np.cos(theta), r_r*np.sin(theta), color="#DDEEFC", zorder=2)

# Annotations
ax2.annotate("", xy=(r_a, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
ax2.text(r_a/2, 1.5, f"d_a = {d_a:.0f} mm", ha="center", va="bottom", fontsize=8)

ax2.annotate("", xy=(r_p, -5), xytext=(0, -5),
             arrowprops=dict(arrowstyle="<->", color="blue", lw=1.2))
ax2.text(r_p/2, -8, f"d = {d_p:.0f} mm", ha="center", va="top", fontsize=8, color="blue")

ax2.annotate("", xy=(r_r, 10), xytext=(0, 10),
             arrowprops=dict(arrowstyle="<->", color="gray", lw=1.0))
ax2.text(r_r/2, 12, f"d_r = {d_r:.0f} mm", ha="center", va="bottom", fontsize=7.5, color="gray")

ax2.set_xlim(-r_a*1.3, r_a*1.5)
ax2.set_ylim(-r_a*1.3, r_a*1.3)

ax2.set_title(
    f"Optimal Gear Geometry\n"
    f"m = {m_s} mm  |  N = {N_TEETH}  |  b = {b_opt:.1f} mm\n"
    f"σ_b = {r_std['bending_stress']:.0f} MPa  |  σ_c = {r_std['contact_stress']:.0f} MPa  "
    f"|  mass = {r_std['mass']*1000:.0f} g",
    fontsize=10, fontweight="bold"
)
ax2.legend(loc="lower right", fontsize=8)

# Crosshatch pattern for face width in corner
bx, by, bw, bh = r_a*1.05, -r_a*0.3, r_a*0.35, b_opt*0.6
rect = plt.Rectangle((bx, by), bw, bh,
                      facecolor="#DDEEFC", edgecolor="#555555", lw=1.0, hatch="//")
ax2.add_patch(rect)
ax2.annotate("", xy=(bx, by+bh), xytext=(bx, by),
             arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0))
ax2.text(bx + bw + 1, by + bh/2, f"b = {b_opt:.1f} mm\n(face width)",
         va="center", fontsize=8)

# ── Panel 3: Convergence history ───────────────────────────────────────────────
ax3 = axes[2]
if history:
    iters = list(range(1, len(history)+1))
    masses = [h['mass']*1000 for h in history]
    sigs_b = [h['sigma_b'] for h in history]
    sigs_c = [h['sigma_c'] for h in history]
    feas   = [h['feasible'] for h in history]
    colors = ['#2CA02C' if f else '#D62828' for f in feas]

    ax3_twin = ax3.twinx()
    ax3.plot(iters, masses, 'k-o', ms=5, lw=1.5, zorder=4, label="Mass (g)")
    ax3_twin.plot(iters, sigs_b, 's--', color='#D62828', ms=5, lw=1.2, label=f"σ_b (MPa)")
    ax3_twin.plot(iters, sigs_c, '^--', color='#1D6FA4', ms=5, lw=1.2, label=f"σ_c (MPa)")
    ax3_twin.axhline(SIGMA_B_ALLOW, color='#D62828', lw=0.8, ls=':', alpha=0.7)
    ax3_twin.axhline(SIGMA_C_ALLOW, color='#1D6FA4', lw=0.8, ls=':', alpha=0.7)

    ax3.scatter(iters, masses, c=colors, zorder=5, s=40)
    ax3.set_xlabel("Iteration", fontsize=10)
    ax3.set_ylabel("Mass (g)", fontsize=10)
    ax3_twin.set_ylabel("Stress (MPa)", fontsize=10)
    ax3.set_title("Convergence History", fontsize=11, fontweight="bold")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper right")

    feas_patch = mpatches.Patch(color='#2CA02C', label='Feasible')
    infeas_patch = mpatches.Patch(color='#D62828', label='Infeasible')
    ax3.legend(lines1+lines2+[feas_patch, infeas_patch],
               labels1+labels2+['Feasible', 'Infeasible'],
               fontsize=7.5, loc="upper right")

ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimizer_gear.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print("  Plot saved: optimizer_gear.png")
