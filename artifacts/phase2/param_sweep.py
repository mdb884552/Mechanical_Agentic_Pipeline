"""
Phase 2 -- Parameter Sweep
===========================
Sweeps beam height H and length L to map the design space before
connecting the optimizer (Phase 3).  Validates that simulate() gives
the same result as the Phase 1 script for the base-case geometry.

Outputs
-------
param_sweep.png   -- 2x3 subplot: deflection / stress / mass-deflection tradeoff
                     for H-sweep (top row) and L-sweep (bottom row)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulate import simulate, _YIELD_PA

# ── 0. Sanity check: reproduce Phase 1 result ─────────────────────────────────
BASE = {'L': 0.100, 'H': 0.010, 'load': 500.0}
r0   = simulate(BASE)

analytical_deflection = (500.0 * 0.100**3) / (3 * 70.0e9 * (0.010**3 / 12))
err_pct = abs(r0['max_deflection'] - analytical_deflection) / analytical_deflection * 100

print(f"\n{'='*55}")
print(f"  Phase 2 sanity check (matches Phase 1 base case)")
print(f"    FEA deflection : {r0['max_deflection']*1e3:.4f} mm")
print(f"    Analytical     : {analytical_deflection*1e3:.4f} mm")
print(f"    Error          : {err_pct:.2f} %")
print(f"    {'PASS' if err_pct < 2.0 else 'FAIL'} (threshold 2%)")
print(f"{'='*55}\n")

# ── 1. Height sweep: H from 5 mm to 25 mm, L = 100 mm fixed ──────────────────
H_vals = np.linspace(0.005, 0.025, 12)
defl_H, stress_H, mass_H = [], [], []

print("Sweeping H (beam height) ...")
for H in H_vals:
    r = simulate({**BASE, 'H': H})
    defl_H.append(r['max_deflection'] * 1e3)     # mm
    stress_H.append(r['max_von_mises'] / 1e6)    # MPa
    mass_H.append(r['mass_per_depth'])            # kg/m
    marker = ' <-- baseline' if abs(H - 0.010) < 1e-6 else ''
    print(f"  H={H*1e3:5.1f} mm  "
          f"defl={defl_H[-1]:7.3f} mm  "
          f"stress={stress_H[-1]:6.1f} MPa  "
          f"mass={mass_H[-1]:.4f} kg/m{marker}")

# ── 2. Length sweep: L from 50 mm to 200 mm, H = 10 mm fixed ─────────────────
L_vals = np.linspace(0.050, 0.200, 12)
defl_L, stress_L, mass_L = [], [], []

print("\nSweeping L (beam length) ...")
for L in L_vals:
    r = simulate({**BASE, 'L': L})
    defl_L.append(r['max_deflection'] * 1e3)
    stress_L.append(r['max_von_mises'] / 1e6)
    mass_L.append(r['mass_per_depth'])
    marker = ' <-- baseline' if abs(L - 0.100) < 1e-6 else ''
    print(f"  L={L*1e3:5.0f} mm  "
          f"defl={defl_L[-1]:7.3f} mm  "
          f"stress={stress_L[-1]:6.1f} MPa  "
          f"mass={mass_L[-1]:.4f} kg/m{marker}")

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    "Phase 2 -- Parametric Design Space: Al-6061 Cantilever Beam  (P = 500 N)",
    fontsize=13, fontweight='bold'
)

H_mm = H_vals * 1e3
L_mm = L_vals * 1e3
yield_MPa = _YIELD_PA / 1e6

# --- Row 0: Height sweep ---
ax = axes[0, 0]
ax.plot(H_mm, defl_H, 'o-', color='steelblue', lw=2)
ax.axvline(10, color='grey', ls='--', alpha=0.6, label='Baseline H=10 mm')
ax.set_xlabel('Beam Height H (mm)')
ax.set_ylabel('Max Deflection (mm)')
ax.set_title('Height vs. Deflection')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(H_mm, stress_H, 's-', color='tomato', lw=2)
ax.axhline(yield_MPa, color='red', ls='--', alpha=0.7, label=f'Yield = {yield_MPa:.0f} MPa')
ax.axvline(10, color='grey', ls='--', alpha=0.6, label='Baseline H=10 mm')
ax.set_xlabel('Beam Height H (mm)')
ax.set_ylabel('Max von Mises Stress (MPa)')
ax.set_title('Height vs. Stress')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.plot(mass_H, defl_H, 'D-', color='purple', lw=2)
# Annotate a few H values
for i, H in enumerate(H_vals):
    if i % 3 == 0:
        ax.annotate(f'{H*1e3:.0f}mm',
                    (mass_H[i], defl_H[i]),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)
ax.set_xlabel('Mass per Depth (kg/m)')
ax.set_ylabel('Max Deflection (mm)')
ax.set_title('Mass vs. Deflection  (H sweep)')
ax.grid(alpha=0.3)

# --- Row 1: Length sweep ---
ax = axes[1, 0]
ax.plot(L_mm, defl_L, 'o-', color='steelblue', lw=2)
ax.axvline(100, color='grey', ls='--', alpha=0.6, label='Baseline L=100 mm')
ax.set_xlabel('Beam Length L (mm)')
ax.set_ylabel('Max Deflection (mm)')
ax.set_title('Length vs. Deflection')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(L_mm, stress_L, 's-', color='tomato', lw=2)
ax.axhline(yield_MPa, color='red', ls='--', alpha=0.7, label=f'Yield = {yield_MPa:.0f} MPa')
ax.axvline(100, color='grey', ls='--', alpha=0.6, label='Baseline L=100 mm')
ax.set_xlabel('Beam Length L (mm)')
ax.set_ylabel('Max von Mises Stress (MPa)')
ax.set_title('Length vs. Stress')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.plot(mass_L, defl_L, 'D-', color='purple', lw=2)
for i, L in enumerate(L_vals):
    if i % 3 == 0:
        ax.annotate(f'{L*1e3:.0f}mm',
                    (mass_L[i], defl_L[i]),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)
ax.set_xlabel('Mass per Depth (kg/m)')
ax.set_ylabel('Max Deflection (mm)')
ax.set_title('Mass vs. Deflection  (L sweep)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('param_sweep.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved -> param_sweep.png")
