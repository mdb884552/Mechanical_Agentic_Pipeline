"""
Phase 3.5 -- Seed the RAG Store
=================================
Pre-populates the ChromaDB store with:
  1. A coarse H sweep (10 points, 4mm-25mm) -- design space coverage
  2. The Phase 3 scipy optimal result
  3. Key engineering lessons from the project

Run once before the agent:  python seed_rag.py
The agent will add more runs during its own loop.
"""

import numpy as np
from simulate import simulate
from rag_store import SimulationStore

store = SimulationStore()

BASE_L = 0.100
BASE_P = 500.0

print(f"\n{'='*55}")
print(f"  Phase 3.5 -- Seeding RAG Store")
print(f"{'='*55}")

# ── 1. H sweep ─────────────────────────────────────────────────────────────────
H_vals = np.linspace(0.004, 0.025, 10)
print(f"\n  Running {len(H_vals)}-point H sweep to seed store ...")

for H in H_vals:
    r = simulate({'L': BASE_L, 'H': H, 'load': BASE_P})
    note = "H sweep seed data"
    run_id = store.log_run({'L': BASE_L, 'H': H, 'load': BASE_P}, r, note)
    feasible = r['max_deflection'] <= 0.050e-3
    print(f"  H={H*1e3:5.1f}mm  defl={r['max_deflection']*1e3:.4f}mm  "
          f"mass={r['mass_per_depth']:.4f}kg/m  "
          f"{'OK' if feasible else 'INFEASIBLE'}")

# ── 2. Phase 3 optimal result ──────────────────────────────────────────────────
print(f"\n  Logging Phase 3 scipy.optimize result ...")

H_opt = 0.008308
r_opt = simulate({'L': BASE_L, 'H': H_opt, 'load': BASE_P})
store.log_run(
    {'L': BASE_L, 'H': H_opt, 'load': BASE_P},
    r_opt,
    "Phase 3 scipy SLSQP optimal: minimum-mass design, deflection constraint active"
)
print(f"  H={H_opt*1e3:.2f}mm  defl={r_opt['max_deflection']*1e3:.4f}mm  "
      f"mass={r_opt['mass_per_depth']:.4f}kg/m  OPTIMAL (scipy baseline)")

# ── 3. Lessons learned ─────────────────────────────────────────────────────────
print(f"\n  Loading engineering lessons ...")

lessons = [
    (
        "For an Al-6061 cantilever beam (L=100mm, P=500N), the deflection constraint "
        "is active at the optimum. Stress is far from yield -- this is a stiffness-dominated problem. "
        "The minimum feasible H is approximately 8.3mm.",
        "physics"
    ),
    (
        "Deflection scales as 1/H^3 for a rectangular cantilever (delta = 4PL^3 / (E*H^3)). "
        "Doubling H reduces deflection by 8x. Most mass reduction comes from reducing H slightly "
        "below the overdesigned baseline -- diminishing returns above H=15mm.",
        "physics"
    ),
    (
        "The deflection constraint boundary is at H_min = (4*P*L^3 / (E*delta_max))^(1/3). "
        "For this problem: H_min = 8.298mm analytically. FEA optimizer found 8.308mm (0.12% error). "
        "Always approach the constraint from the feasible side to avoid oscillation.",
        "optimization"
    ),
    (
        "When the optimizer hits the lower bound instead of the constraint, check the numerical "
        "gradient. Cache rounding (round to 7 decimals) made gradient appear zero because SLSQP "
        "finite-difference step (~15nm) was smaller than the rounding resolution (100nm). "
        "Fix: use exact float as cache key.",
        "debugging"
    ),
    (
        "Plane stress vs plane strain: using 3D Lame parameters in a 2D model implicitly assumes "
        "plane strain, which over-stiffens the model by ~1/(1-nu^2) = 11% for nu=0.33. "
        "Correct plane-stress parameters: lam* = E*nu/(1-nu^2), mu = E/(2*(1+nu)).",
        "physics"
    ),
    (
        "scikit-fem 12.x API: MeshTri.init_tensor() needs .with_defaults() for named boundary "
        "groups. basis.split(u) returns [(ux_array, ux_basis), (uy_array, uy_basis)]. "
        "P2 elements have edge-midpoint DOFs -- use ux[:n_nodes] for vertex-only plotting.",
        "api"
    ),
]

for text, category in lessons:
    store.log_lesson(text, category)
    print(f"  + [{category}] {text[:70]}...")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Store populated:")
print(f"    Simulation runs : {store.run_count()}")
print(f"    Lessons         : {store.lesson_count()}")
print(f"{'='*55}\n")
