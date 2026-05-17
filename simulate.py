"""
Phase 2 -- Parametric Simulation Module
========================================
Exposes simulate(params) -> dict: the black-box callable that the
optimizer (Phase 3) and AI agent (Phase 4) will drive.

Usage
-----
    from simulate import simulate
    result = simulate({'L': 0.10, 'H': 0.015})
    print(result['max_deflection'], result['max_von_mises'], result['mass_per_depth'])

Notes
-----
- 2-D plane-stress FEA, P2 (quadratic) triangular elements, Al-6061 defaults.
- Mesh: structured triangular via MeshTri.init_tensor (same approach as Phase 1).
  Phase 3 will replace _make_mesh with gmsh when geometry becomes non-rectangular.
- von Mises stress: vectorized P1 constant-strain approximation per element,
  using vertex displacements extracted from the P2 solution.
"""

import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity


# ── Material defaults: aluminium 6061-T6 ──────────────────────────────────────
_AL6061 = {'E': 70.0e9, 'nu': 0.33, 'rho': 2700.0}
_YIELD_PA = 276.0e6   # yield strength (Pa) -- used externally for constraint checks


def simulate(params: dict) -> dict:
    """
    Run a 2-D plane-stress FEA on a parametric rectangular cantilever beam.

    Required keys
    -------------
    L   : float  beam length (m)
    H   : float  beam height (m)

    Optional keys            default
    --------------------------------
    E         Young's modulus (Pa)      70e9   (Al-6061)
    nu        Poisson's ratio           0.33
    rho       density (kg/m^3)          2700
    load      tip load, downward (N)    500
    nx        elements along length     auto (scales with L/H)
    ny        elements through height   10

    Returns
    -------
    dict
        max_deflection  float  maximum |uy| at any node (m)
        max_von_mises   float  maximum von Mises stress (Pa)
        mass_per_depth  float  rho * L * H  (kg per metre of depth)
        n_dofs          int    degrees of freedom (mesh quality indicator)
    """
    L   = float(params['L'])
    H   = float(params['H'])
    E   = float(params.get('E',   _AL6061['E']))
    nu  = float(params.get('nu',  _AL6061['nu']))
    rho = float(params.get('rho', _AL6061['rho']))
    P   = float(params.get('load', 500.0))
    ny  = int(params.get('ny', 10))
    nx  = int(params.get('nx', max(int(L / H * ny * 0.5), 20)))

    mesh  = _make_mesh(L, H, nx, ny)
    elem  = ElementVector(ElementTriP2())
    basis = Basis(mesh, elem)

    lam = E * nu / (1.0 - nu**2)   # plane-stress Lame parameters
    mu  = E / (2.0 * (1.0 + nu))
    K   = asm(linear_elasticity(lam, mu), basis)

    fb = FacetBasis(mesh, elem, facets=mesh.boundaries['right'])

    @LinearForm
    def traction(v, w):
        return -(P / H) * v[1]

    f = traction.assemble(fb)

    fixed = basis.get_dofs(mesh.boundaries['left']).all()
    u     = solve(*condense(K, f, D=fixed))

    n_nodes = mesh.p.shape[1]
    (ux, _), (uy, _) = basis.split(u)
    ux_n = ux[:n_nodes]   # vertex-only values (P2 has extra edge-midpoint DOFs)
    uy_n = uy[:n_nodes]

    return {
        'max_deflection': float(np.max(np.abs(uy_n))),
        'max_von_mises':  _von_mises_max(mesh, ux_n, uy_n, lam, mu),
        'mass_per_depth': rho * L * H,
        'n_dofs':         basis.N,
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_mesh(L: float, H: float, nx: int, ny: int) -> MeshTri:
    """Structured rectangular triangular mesh with named boundary groups."""
    return MeshTri.init_tensor(
        np.linspace(0, L, nx + 1),
        np.linspace(0, H, ny + 1),
    ).with_defaults()


def _von_mises_max(mesh: MeshTri,
                   ux: np.ndarray,
                   uy: np.ndarray,
                   lam: float,
                   mu: float) -> float:
    """
    Vectorized per-element von Mises stress from vertex displacements.

    Uses the P1 constant-strain triangle approximation: strain is constant
    within each element, computed from the shape-function gradients derived
    from the three vertex coordinates.  Accurate where the stress varies
    slowly (away from stress concentrations).
    """
    # Vertex coordinates and displacements per element -- shape (3, n_elements)
    x = mesh.p[0, mesh.t]
    y = mesh.p[1, mesh.t]
    u = ux[mesh.t]
    v = uy[mesh.t]

    # Element areas
    A = 0.5 * np.abs(
        (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    )

    # Linear shape-function gradients (constant per element)
    dN_dx = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2.0 * A)
    dN_dy = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2.0 * A)

    # Strain components
    eps_xx = np.einsum('ij,ij->j', dN_dx, u)
    eps_yy = np.einsum('ij,ij->j', dN_dy, v)
    gam_xy = (np.einsum('ij,ij->j', dN_dy, u) +
               np.einsum('ij,ij->j', dN_dx, v))

    # Plane-stress constitutive law
    sig_xx = (lam + 2 * mu) * eps_xx + lam * eps_yy
    sig_yy = lam * eps_xx + (lam + 2 * mu) * eps_yy
    tau_xy = mu * gam_xy

    # Von Mises
    vm = np.sqrt(sig_xx**2 - sig_xx * sig_yy + sig_yy**2 + 3 * tau_xy**2)
    return float(np.max(vm))
