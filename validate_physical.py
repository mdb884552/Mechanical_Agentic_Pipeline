"""
Phase 5 -- Physical Validation
================================
Compares measured load-deflection data against FEA predictions
to quantify the simulation-to-reality gap.

The printed part is PLA/PETG -- not Al-6061. This script runs
simulate() with the actual print material's properties so the
comparison is fair.

Usage
-----
  python validate_physical.py                    # prompts for CSV or uses demo data
  python validate_physical.py --csv measured_data.csv
  python validate_physical.py --csv measured_data.csv --material petg
  python validate_physical.py --csv measured_data.csv --H 8.31 --depth 19.8

  # Measure the actual printed part dimensions with calipers before running.
  # Pass --H and --depth to match what you physically measured (not the STL target).

measured_data.csv format (no header required, or with header load_N,deflection_mm):
  5,0.498
  10,0.997
  20,1.993
  50,4.981

Output
------
  validation_results.png  --  3-panel comparison plot
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import simulate

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Phase 5 -- Physical validation")
_parser.add_argument("--csv",      type=str,   default=None,
                     help="CSV file with measured (load_N, deflection_mm) pairs")
_parser.add_argument("--material", type=str,   default="pla",
                     choices=["pla", "petg", "al"],
                     help="Print material for FEA comparison (default: pla)")
_parser.add_argument("--H",        type=float, default=8.308,
                     help="Measured beam height mm (caliper reading, default: 8.308)")
_parser.add_argument("--L",        type=float, default=100.0,
                     help="Measured beam length mm (default: 100)")
_parser.add_argument("--depth",    type=float, default=20.0,
                     help="Measured print depth mm (default: 20)")
_parser.add_argument("--out",      type=str,   default="validation_results.png",
                     help="Output plot filename")
ARGS = _parser.parse_args()

# ── Material properties ───────────────────────────────────────────────────────
MATERIALS = {
    "pla":  {"E": 3.5e9,  "nu": 0.35, "label": "PLA  (E=3.5 GPa)"},
    "petg": {"E": 2.1e9,  "nu": 0.40, "label": "PETG (E=2.1 GPa)"},
    "al":   {"E": 70.0e9, "nu": 0.33, "label": "Al-6061 (E=70 GPa)"},
}
mat      = MATERIALS[ARGS.material]
E_mat    = mat["E"]
nu_mat   = mat["nu"]
mat_label = mat["label"]

H_mm    = ARGS.H
L_mm    = ARGS.L
depth   = ARGS.depth
depth_m = depth * 1e-3
out     = ARGS.out


# ── Prediction functions ──────────────────────────────────────────────────────

def fea_deflection(P_N):
    """FEA tip deflection (mm) at the actual print depth."""
    r = simulate({
        'L':    L_mm * 1e-3,
        'H':    H_mm * 1e-3,
        'load': P_N,
        'E':    E_mat,
        'nu':   nu_mat,
    })
    # simulate() returns deflection for b=1m depth; scale to actual depth
    return r['max_deflection'] / depth_m * 1e3   # → mm


def eb_deflection(P_N):
    """Euler-Bernoulli analytical tip deflection (mm)."""
    H = H_mm * 1e-3
    L = L_mm * 1e-3
    b = depth_m
    I = b * H**3 / 12.0
    return P_N * L**3 / (3.0 * E_mat * I) * 1e3  # → mm


# ── Load measured data ────────────────────────────────────────────────────────

def _load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() in ("load_n", "load", "#"):
                continue
            rows.append((float(row[0]), float(row[1])))
    return rows


DEMO_DATA = [
    (1.0,  0.102),
    (2.0,  0.201),
    (5.0,  0.503),
    (10.0, 1.009),
    (20.0, 2.018),
    (50.0, 5.031),
]

if ARGS.csv and os.path.exists(ARGS.csv):
    measured = _load_csv(ARGS.csv)
    data_source = ARGS.csv
else:
    measured = DEMO_DATA
    data_source = "demo data (±1% synthetic noise on Euler-Bernoulli)"
    if ARGS.csv:
        print(f"  WARNING: {ARGS.csv} not found -- using demo data")

loads_meas  = np.array([r[0] for r in measured])
deflec_meas = np.array([r[1] for r in measured])

# ── Compute predictions ───────────────────────────────────────────────────────
P_curve = np.linspace(0, max(loads_meas) * 1.1, 60)
d_fea   = np.array([fea_deflection(P) for P in P_curve])
d_eb    = np.array([eb_deflection(P)  for P in P_curve])

# FEA and analytical predictions AT the measured loads (for residual)
d_fea_at_meas = np.array([fea_deflection(P) for P in loads_meas])
d_eb_at_meas  = np.array([eb_deflection(P)  for P in loads_meas])

residual_fea_pct = (deflec_meas - d_fea_at_meas) / d_fea_at_meas * 100.0
residual_eb_pct  = (deflec_meas - d_eb_at_meas)  / d_eb_at_meas  * 100.0

mean_err_fea = float(np.mean(np.abs(residual_fea_pct)))
mean_err_eb  = float(np.mean(np.abs(residual_eb_pct)))
max_err_fea  = float(np.max(np.abs(residual_fea_pct)))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 11))
fig.suptitle(
    f"Phase 5 -- Physical Validation\n"
    f"Material: {mat_label}   "
    f"H={H_mm:.2f} mm   L={L_mm:.0f} mm   depth={depth:.0f} mm",
    fontsize=12, fontweight="bold"
)

# Panel 1: Load-deflection
ax1 = axes[0]
ax1.plot(P_curve, d_fea, "b-",  lw=2, label="FEA prediction (scikit-fem)")
ax1.plot(P_curve, d_eb,  "g--", lw=1.5, label="Euler-Bernoulli (analytical)")
ax1.scatter(loads_meas, deflec_meas, color="red", zorder=5,
            s=60, label="Measured")
ax1.set_xlabel("Tip load (N)")
ax1.set_ylabel("Tip deflection (mm)")
ax1.set_title("Load-Deflection Curve")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# Panel 2: Absolute residual
ax2 = axes[1]
ax2.axhline(0, color="k", lw=0.8, ls="--")
ax2.plot(loads_meas, residual_fea_pct, "bo-", lw=1.5, ms=6,
         label=f"vs FEA  (mean abs {mean_err_fea:.1f}%)")
ax2.plot(loads_meas, residual_eb_pct,  "g^-", lw=1.5, ms=6,
         label=f"vs E-B  (mean abs {mean_err_eb:.1f}%)")
ax2.set_xlabel("Tip load (N)")
ax2.set_ylabel("(Measured - Predicted) / Predicted  (%)")
ax2.set_title("Residual  --  Measured vs Predicted")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Parity plot (measured vs predicted)
ax3 = axes[2]
d_max = max(deflec_meas.max(), d_fea_at_meas.max()) * 1.05
ax3.plot([0, d_max], [0, d_max], "k--", lw=1, label="Perfect agreement")
ax3.scatter(d_fea_at_meas, deflec_meas, color="blue", zorder=5, s=60,
            label="FEA vs Measured")
ax3.scatter(d_eb_at_meas,  deflec_meas, color="green", marker="^", zorder=5,
            s=60, label="E-B vs Measured")
ax3.set_xlabel("Predicted deflection (mm)")
ax3.set_ylabel("Measured deflection (mm)")
ax3.set_title("Parity Plot")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, d_max)
ax3.set_ylim(0, d_max)

plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Phase 5 -- Physical Validation Results")
print(f"{'='*55}")
print(f"  Data source : {data_source}")
print(f"  Material    : {mat_label}")
print(f"  Geometry    : L={L_mm:.1f} mm  H={H_mm:.3f} mm  depth={depth:.1f} mm")
print(f"  Data points : {len(measured)}")
print()
print(f"  {'Load (N)':>9}  {'Measured':>10}  {'FEA':>10}  {'E-B':>10}  {'err-FEA':>9}")
for P, d_m, d_f, d_a in zip(loads_meas, deflec_meas, d_fea_at_meas, d_eb_at_meas):
    err = (d_m - d_f) / d_f * 100.0
    print(f"  {P:>9.1f}  {d_m:>10.3f}  {d_f:>10.3f}  {d_a:>10.3f}  {err:>+8.1f}%")
print()
print(f"  FEA mean absolute error : {mean_err_fea:.2f}%")
print(f"  FEA max absolute error  : {max_err_fea:.2f}%")
print(f"  E-B mean absolute error : {mean_err_eb:.2f}%")
print()

gap_note = (
    "< 5%  -- FEA and reality in excellent agreement"
    if mean_err_fea < 5 else
    "5-15% -- typical range (material E uncertainty, geometry tolerances)"
    if mean_err_fea < 15 else
    "> 15% -- check: infill=100%? E value correct? Clamp slip? Measured H/depth?"
)
print(f"  Gap assessment: {gap_note}")
print(f"\n  Plot saved: {out}")
print(f"{'='*55}\n")
