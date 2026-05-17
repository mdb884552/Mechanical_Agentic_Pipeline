"""
Phase 3 -- Constrained Optimization
=====================================
Finds the minimum-mass beam height that satisfies deflection and stress
constraints using scipy.optimize (SLSQP).

Run without AI first -- this establishes the ground-truth optimum that the
LangGraph agent (Phase 4) will be compared against.

Problem
-------
  Minimize    mass_per_depth = rho * L * H          (kg/m)
  over        H in [3 mm, 30 mm],  L = 100 mm fixed
  subject to  max_deflection  <= DELTA_MAX           (stiffness constraint)
              max_von_mises   <= 0.6 * yield_strength (strength constraint)

Analytical check
----------------
  For a rectangular cantilever, deflection = 4PL^3 / (E H^3).
  Setting delta = DELTA_MAX and solving:  H_min = (4PL^3 / (E * delta_max))^(1/3)
  FEA optimizer should agree to within mesh error.

Outputs
-------
  optimizer_results.png  -- 3-panel: constraint landscape + convergence history
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from simulate import simulate, _YIELD_PA

# ── Problem constants ──────────────────────────────────────────────────────────
L           = 0.100          # fixed beam length (m)
P           = 500.0          # tip load (N)
RHO         = 2700.0         # Al-6061 density (kg/m^3)
E_AL        = 70.0e9         # Young's modulus (Pa)
DELTA_MAX   = 0.050e-3       # max allowable deflection: 0.05 mm
STRESS_MAX  = 0.6 * _YIELD_PA   # 60% of yield = 165.6 MPa (safety factor of 1.67)
H_BOUNDS    = (0.003, 0.030)    # 3 mm to 30 mm

# ── Memoised simulation ────────────────────────────────────────────────────────
_cache: dict = {}

def _sim(H: float) -> dict:
    """Cache by exact float value. Repeated calls at the same x hit the cache;
    SLSQP's finite-difference perturbations (different float) run their own FEA.
    Rounding the key would make the constraint gradient appear zero and break the solver."""
    if H not in _cache:
        _cache[H] = simulate({'L': L, 'H': H, 'load': P})
    return _cache[H]

# ── Objective and constraints ──────────────────────────────────────────────────
def objective(x):
    return RHO * L * x[0]      # mass per metre of depth; L is fixed so min mass = min H

def c_deflection(x):
    return DELTA_MAX - _sim(x[0])['max_deflection']   # >= 0 when feasible

def c_stress(x):
    return STRESS_MAX - _sim(x[0])['max_von_mises']   # >= 0 when feasible

# ── Convergence history ────────────────────────────────────────────────────────
history: list[dict] = []

def _callback(x):
    r = _sim(x[0])
    history.append({
        'H':          x[0] * 1e3,                          # mm
        'mass':       RHO * L * x[0],                      # kg/m
        'deflection': r['max_deflection'] * 1e3,           # mm
        'stress':     r['max_von_mises'] / 1e6,            # MPa
    })

# ── Run optimizer ──────────────────────────────────────────────────────────────
H0 = 0.010    # start from Phase 1 / Phase 2 baseline (10 mm)

print(f"\n{'='*55}")
print(f"  Phase 3 -- Constrained Optimization")
print(f"  L={L*1e3:.0f}mm  P={P:.0f}N  Al-6061")
print(f"  delta_max={DELTA_MAX*1e3:.3f}mm  stress_max={STRESS_MAX/1e6:.1f}MPa")
print(f"  Optimizing H in [{H_BOUNDS[0]*1e3:.0f}mm, {H_BOUNDS[1]*1e3:.0f}mm]")
print(f"{'='*55}")

result = minimize(
    objective,
    x0=[H0],
    method='SLSQP',
    bounds=[H_BOUNDS],
    constraints=[
        {'type': 'ineq', 'fun': c_deflection},
        {'type': 'ineq', 'fun': c_stress},
    ],
    callback=_callback,
    options={'ftol': 1e-10, 'maxiter': 200},
)

H_opt   = result.x[0]
r_opt   = _sim(H_opt)
r_init  = _sim(H0)
mass_i  = RHO * L * H0
mass_o  = r_opt['mass_per_depth']

# ── Analytical verification ────────────────────────────────────────────────────
# delta = 4PL^3 / (E H^3)  =>  H_min = (4PL^3 / (E * delta_max))^(1/3)
H_analytical = (4 * P * L**3 / (E_AL * DELTA_MAX)) ** (1 / 3)

# ── Print summary ──────────────────────────────────────────────────────────────
print(f"\n  Initial design  (H = {H0*1e3:.1f} mm)")
print(f"    Mass/depth    : {mass_i:.4f} kg/m")
print(f"    Deflection    : {r_init['max_deflection']*1e3:.4f} mm"
      f"  (limit {DELTA_MAX*1e3:.3f} mm)")
print(f"    Stress        : {r_init['max_von_mises']/1e6:.2f} MPa"
      f"  (limit {STRESS_MAX/1e6:.1f} MPa)")

print(f"\n  Optimal design  (H = {H_opt*1e3:.3f} mm)")
print(f"    Mass/depth    : {mass_o:.4f} kg/m")
print(f"    Deflection    : {r_opt['max_deflection']*1e3:.4f} mm"
      f"  (limit {DELTA_MAX*1e3:.3f} mm)")
print(f"    Stress        : {r_opt['max_von_mises']/1e6:.2f} MPa"
      f"  (limit {STRESS_MAX/1e6:.1f} MPa)")
print(f"    Mass saving   : {(1 - mass_o/mass_i)*100:.1f}%")
print(f"    Iterations    : {len(history)}")

print(f"\n  Analytical H_min (beam theory)   : {H_analytical*1e3:.3f} mm")
print(f"  FEA optimizer H_opt              : {H_opt*1e3:.3f} mm")
print(f"  Agreement                        : "
      f"{abs(H_opt - H_analytical) / H_analytical * 100:.2f}%")

active = []
if abs(r_opt['max_deflection'] - DELTA_MAX) / DELTA_MAX < 0.01:
    active.append('deflection')
if abs(r_opt['max_von_mises'] - STRESS_MAX) / STRESS_MAX < 0.01:
    active.append('stress')
print(f"  Active constraint(s)             : "
      f"{', '.join(active) if active else 'none (check bounds)'}")
print(f"{'='*55}\n")

# ── Constraint landscape sweep ─────────────────────────────────────────────────
H_sweep = np.linspace(H_BOUNDS[0], H_BOUNDS[1], 60)
defl_sweep, stress_sweep, mass_sweep = [], [], []
for H in H_sweep:
    r = _sim(H)
    defl_sweep.append(r['max_deflection'] * 1e3)
    stress_sweep.append(r['max_von_mises'] / 1e6)
    mass_sweep.append(RHO * L * H)

H_mm = H_sweep * 1e3

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Phase 3 -- Constrained Optimization: Minimum-Mass Cantilever Beam  (Al-6061, P=500N)",
    fontsize=12, fontweight='bold'
)

# Panel 1: deflection vs H with constraint boundary
ax = axes[0]
ax.plot(H_mm, defl_sweep, 'steelblue', lw=2, label='FEA deflection')
ax.axhline(DELTA_MAX * 1e3, color='red', ls='--', lw=1.5,
           label=f'Limit = {DELTA_MAX*1e3:.3f} mm')
ax.axvline(H_opt * 1e3, color='green', ls=':', lw=2,
           label=f'Optimal H = {H_opt*1e3:.2f} mm')
ax.axvline(H0 * 1e3, color='grey', ls=':', lw=1.5,
           label=f'Initial H = {H0*1e3:.1f} mm')
ax.fill_between(H_mm,
                [DELTA_MAX * 1e3] * len(H_mm), defl_sweep,
                where=[d > DELTA_MAX * 1e3 for d in defl_sweep],
                alpha=0.15, color='red', label='Infeasible region')
ax.set_xlabel('Beam Height H (mm)')
ax.set_ylabel('Max Deflection (mm)')
ax.set_title('Deflection Constraint')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

# Panel 2: stress vs H with constraint boundary
ax = axes[1]
ax.plot(H_mm, stress_sweep, 'tomato', lw=2, label='FEA von Mises stress')
ax.axhline(STRESS_MAX / 1e6, color='red', ls='--', lw=1.5,
           label=f'Limit = {STRESS_MAX/1e6:.1f} MPa')
ax.axvline(H_opt * 1e3, color='green', ls=':', lw=2,
           label=f'Optimal H = {H_opt*1e3:.2f} mm')
ax.axvline(H0 * 1e3, color='grey', ls=':', lw=1.5,
           label=f'Initial H = {H0*1e3:.1f} mm')
ax.set_xlabel('Beam Height H (mm)')
ax.set_ylabel('Max von Mises Stress (MPa)')
ax.set_title('Stress Constraint')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

# Panel 3: convergence history
ax = axes[2]
if history:
    iters = list(range(1, len(history) + 1))
    h_hist = [h['H'] for h in history]
    ax.plot(iters, h_hist, 'o-', color='purple', lw=2, ms=6)
    ax.axhline(H_opt * 1e3, color='green', ls='--', lw=1.5,
               label=f'Converged H = {H_opt*1e3:.2f} mm')
    ax.axhline(H_analytical * 1e3, color='orange', ls=':', lw=1.5,
               label=f'Analytical H = {H_analytical*1e3:.2f} mm')
    ax.set_xlabel('Optimizer Iteration')
    ax.set_ylabel('Design Variable H (mm)')
    ax.set_title(f'Convergence History  ({len(history)} iterations)')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_results.png', dpi=150, bbox_inches='tight')
print("  Plot saved -> optimizer_results.png")
