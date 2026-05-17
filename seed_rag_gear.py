"""
Seed the RAG store with gear simulation runs and lessons.
Run once before agent_gear.py.
"""

import numpy as np
from simulate_gear import simulate, SIGMA_B_ALLOW, SIGMA_C_ALLOW, N_TEETH
from rag_store import SimulationStore

store = SimulationStore(db_path="./chroma_db_gear")

print(f"\n{'='*55}")
print(f"  Seeding Gear RAG Store")
print(f"{'='*55}")

BASE_T = 50.0

# ── 1. m sweep at fixed b=40mm ─────────────────────────────────────────────────
print(f"\n  m-sweep (b=40mm fixed) ...")
for m in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    p = {'m': m, 'b': 40.0, 'torque': BASE_T}
    r = simulate(p)
    feas = r['bending_stress'] <= SIGMA_B_ALLOW and r['contact_stress'] <= SIGMA_C_ALLOW
    note = "m-sweep seed, b=40mm"
    store.log_run(p, r, note)
    print(f"  m={m:.1f}  b=40  sigma_b={r['bending_stress']:.0f}  "
          f"sigma_c={r['contact_stress']:.0f}  mass={r['mass']*1000:.0f}g  "
          f"{'OK' if feas else 'INFEASIBLE'}")

# ── 2. b sweep at fixed m=3mm ──────────────────────────────────────────────────
print(f"\n  b-sweep (m=3mm fixed) ...")
for b in [15, 20, 30, 40, 50, 60, 70, 80]:
    p = {'m': 3.0, 'b': float(b), 'torque': BASE_T}
    r = simulate(p)
    feas = r['bending_stress'] <= SIGMA_B_ALLOW and r['contact_stress'] <= SIGMA_C_ALLOW
    store.log_run(p, r, "b-sweep seed, m=3mm")
    print(f"  m=3.0  b={b:2d}  sigma_b={r['bending_stress']:.0f}  "
          f"sigma_c={r['contact_stress']:.0f}  mass={r['mass']*1000:.0f}g  "
          f"{'OK' if feas else 'INFEASIBLE'}")

# ── 3. Gear engineering lessons ────────────────────────────────────────────────
print(f"\n  Loading gear engineering lessons ...")

lessons = [
    (
        f"For N={N_TEETH} teeth and T=50 N*m, contact (Hertz) stress is the binding constraint. "
        f"Bending (Lewis) stress is rarely active for this tooth count and loading. "
        f"The minimum feasible design lies on the sigma_c = {SIGMA_C_ALLOW} MPa boundary.",
        "physics"
    ),
    (
        "Contact stress scales as 1/sqrt(b * m^2). Both increasing b and increasing m reduce "
        "contact stress. Bending stress scales as 1/(b * m^2). Both constraints improve with "
        "larger m or larger b. Mass = rho*pi/4*(m*N)^2*b scales as m^2*b.",
        "physics"
    ),
    (
        "Mass = rho*pi/4*(m*N)^2*b is approximately constant along the contact stress boundary "
        "sigma_c = C_p*sqrt(F_t*K_o/(b*d*I)). This means sigma_c is the binding constraint "
        "and many (m, b) pairs give nearly identical mass. Approach from above (feasible side).",
        "optimization"
    ),
    (
        "Standard metric modules (ISO 54): 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10 mm. "
        "Always snap the continuous optimum to the nearest standard module. "
        "For N=15 at T=50 Nm, typical optimal modules are 3-4 mm.",
        "manufacturing"
    ),
    (
        "Face width guideline: 8*m <= b <= 16*m. Violating this leads to uneven load "
        "distribution across the tooth width (too wide) or insufficient contact area (too narrow). "
        "For m=3mm, the guideline is b in [24, 48] mm.",
        "design"
    ),
    (
        "For N=15, undercutting does not occur (minimum for 20 deg pressure angle is N=17 "
        "for full-depth teeth, but profile shifting can extend this down to ~12). "
        "N=15 requires a small positive profile shift or stub tooth to avoid undercutting in strict "
        "full-depth designs -- acceptable for most practical applications.",
        "geometry"
    ),
]

for text, cat in lessons:
    store.log_lesson(text, cat)
    print(f"  + [{cat}] {text[:70]}...")

print(f"\n{'='*55}")
print(f"  Store populated:")
print(f"    Simulation runs : {store.run_count()}")
print(f"    Lessons         : {store.lesson_count()}")
print(f"{'='*55}\n")
