"""
Phase 1 Validation: Cantilever Beam FEA
========================================
Problem  : Rectangular beam, clamped at x=0, point load P at x=L (downward)
Reference: Euler-Bernoulli beam theory
   tip deflection  δ = P·L³ / (3·E·I)
   root stress  σ_max = P·L·(H/2) / I
Pass criterion: FEA tip deflection within 2% of analytical.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive — saves to file without needing a display window
import matplotlib.pyplot as plt
from skfem import *
from skfem.models.elasticity import linear_elasticity, plane_stress, lame_parameters

# ── 0. PARAMETERS ─────────────────────────────────────────────────────────────
L   = 0.100     # beam length  (m)
H   = 0.010     # beam height  (m)   → aspect ratio 10:1, beam theory valid
E   = 70.0e9    # Young's modulus, aluminium 6061 (Pa)
nu  = 0.33      # Poisson's ratio
P   = 500.0     # tip load, downward (N)

# ── 1. ANALYTICAL SOLUTION ────────────────────────────────────────────────────
I_beam    = H**3 / 12                        # 2nd moment of area / unit width (m⁴)
d_exact   = P * L**3 / (3 * E * I_beam)     # tip deflection (m)
sig_exact = P * L * (H / 2) / I_beam        # max bending stress at root (Pa)

print(f"\n{'-'*50}")
print(f"  Analytical solution (Euler-Bernoulli)")
print(f"    Tip deflection : {d_exact*1e3:>10.4f} mm")
print(f"    Root stress    : {sig_exact/1e6:>10.2f} MPa")
print(f"{'-'*50}\n")

# ── 2. MESH ───────────────────────────────────────────────────────────────────
# Structured triangular mesh over a 2D rectangle.
# MeshTri.init_tensor creates four named boundary groups automatically:
#   'left'  (x = 0)  — where we clamp
#   'right' (x = L)  — where we apply the load
#   'top'   (y = H)  — free surface
#   'bottom'(y = 0)  — free surface
nx, ny = 40, 10       # elements along length / through height
mesh = MeshTri.init_tensor(
    np.linspace(0, L, nx + 1),
    np.linspace(0, H, ny + 1),
).with_defaults()

# ── 3. FUNCTION SPACE & STIFFNESS MATRIX ──────────────────────────────────────
# ElementVector wraps a scalar P2 element to give a 2-component (ux, uy) field.
# P2 (quadratic) is more accurate than P1 for bending problems.
elem  = ElementVector(ElementTriP2())
basis = Basis(mesh, elem)

# Plane-stress Lame parameters (sigma_zz = 0 assumption, correct for thin 2D sections).
# Derived by matching the 2D constitutive matrix C = E/(1-nu^2)*[[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]].
lam = E * nu / (1.0 - nu**2)   # lambda* (plane-stress)
mu  = E / (2.0 * (1.0 + nu))   # mu unchanged (= shear modulus G)
K = asm(linear_elasticity(lam, mu), basis)

# ── 4. LOAD VECTOR ────────────────────────────────────────────────────────────
# Uniform downward traction on the right face:  t_y = -P/H  (N/m per unit width)
# This is equivalent to a total force P acting at the centroid of the right edge.
fb = FacetBasis(mesh, elem, facets=mesh.boundaries['right'])

@LinearForm
def traction(v, w):
    return -(P / H) * v[1]          # v[1] = y-component of test function

f = traction.assemble(fb)

# ── 5. BOUNDARY CONDITIONS ────────────────────────────────────────────────────
# Clamp the left edge: zero displacement in both x and y.
fixed = basis.get_dofs(mesh.boundaries['left']).all()
u     = solve(*condense(K, f, D=fixed))

# ── 6. POST-PROCESS ───────────────────────────────────────────────────────────
# Split the solution vector into scalar nodal fields (ux, uy).
# basis.split returns [(values, sub_basis), ...] — one entry per component.
(ux, _ux_basis), (uy, _uy_basis) = basis.split(u)

d_fea   = np.max(np.abs(uy))
err_pct = abs(d_fea - d_exact) / d_exact * 100

print(f"  FEA result  ({basis.N:,} DOFs,  {mesh.t.shape[1]:,} elements)")
print(f"    Tip deflection : {d_fea*1e3:>10.4f} mm")
print(f"    Error vs exact : {err_pct:>10.2f} %")

if err_pct < 2.0:
    print(f"\n  PASS -- within 2% threshold\n")
else:
    print(f"\n  FAIL -- refine mesh or check BCs\n")
print(f"{'-'*50}\n")

# For plotting we only need values at mesh vertices (first n_nodes entries of the P2 field).
n_nodes = mesh.p.shape[1]
ux_plot = ux[:n_nodes]
uy_plot = uy[:n_nodes]

# ── 7. PLOTS ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Phase 1 — Cantilever Beam FEA Validation", fontsize=13, fontweight='bold')

# — Left panel: deformed shape coloured by vertical displacement uy —
ax = axes[0]

# Scale factor so the visual deflection is readable (3 mm on screen)
disp_scale = 0.003 / (d_fea + 1e-15)
p_def      = mesh.p.copy()
p_def[0]  += ux_plot * disp_scale
p_def[1]  += uy_plot * disp_scale

ax.triplot(*mesh.p, mesh.t.T, color='lightgrey', lw=0.3, zorder=0, label='Undeformed')
face_vals = uy_plot[mesh.t].mean(axis=0)     # average uy per element for colouring
tpc = ax.tripcolor(*p_def, mesh.t.T, facecolors=face_vals, cmap='RdYlBu_r', shading='flat')
fig.colorbar(tpc, ax=ax, label='uy  (m)')
ax.set_title(f"Deformed (×{disp_scale:.0f})   δ_tip = {d_fea*1e3:.3f} mm")
ax.set_xlabel('x  (m)')
ax.set_ylabel('y  (m)')
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='lower left')

# — Right panel: deflection profile along neutral axis vs analytical —
ax = axes[1]

# Gather nodes near y = H/2 (neutral axis)
neutral_mask = np.abs(mesh.p[1] - H / 2) < H * 0.07
neutral_x    = mesh.p[0][neutral_mask]
neutral_uy   = uy_plot[neutral_mask]
order        = np.argsort(neutral_x)

# Analytical deflection curve: uy(x) = -P/(6EI) * x² * (3L - x)
x_ref  = np.linspace(0, L, 300)
uy_ref = -(P / (6 * E * I_beam)) * x_ref**2 * (3*L - x_ref)

ax.plot(x_ref*1e3, uy_ref*1e3, 'k--', lw=2, label='Analytical')
ax.scatter(neutral_x[order]*1e3, neutral_uy[order]*1e3,
           s=10, c='steelblue', zorder=5, label='FEA (neutral axis)')
ax.set_xlabel('x  (mm)')
ax.set_ylabel('uy  (mm)')
ax.set_title('Deflection Profile — FEA vs. Analytical')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fea_results.png', dpi=150, bbox_inches='tight')
print("  Plot saved -> fea_results.png")
