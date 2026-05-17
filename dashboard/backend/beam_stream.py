"""Streaming beam optimization — yields event dicts at each agent iteration."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # aero-opt root
sys.path.insert(0, str(Path(__file__).parent))                  # dashboard/backend

import anthropic
from simulate import simulate, _YIELD_PA
from rag_store import SimulationStore

DELTA_MAX  = 0.050e-3
STRESS_MAX = 0.6 * _YIELD_PA
H_MIN, H_MAX = 0.003, 0.030
MAX_ITER   = 10
CONV_TOL   = 0.005

SYSTEM_PROMPT = f"""You are an aerospace structural optimization assistant inside an automated design loop.

PROBLEM: Minimize mass of a 2D Al-6061 cantilever beam.
  Fixed: L = 100 mm, tip load P = 500 N
  Design variable: H (beam height, meters), bounds [3mm, 30mm]

CONSTRAINTS (all must be satisfied):
  max_deflection <= {DELTA_MAX*1e3:.3f} mm   (stiffness limit)
  max_von_mises  <= {STRESS_MAX/1e6:.1f} MPa  (60% of Al-6061 yield, SF=1.67)

PHYSICS: Deflection ~ 1/H^3. Constraint boundary near H = 8.3 mm analytically.
  Approach the boundary from the feasible side.

RESPOND with ONLY valid JSON:
{{"H": <float_meters>, "reasoning": "<engineering rationale, 1-2 sentences>"}}"""


def _feasible(result):
    return (result["max_deflection"] <= DELTA_MAX and
            result["max_von_mises"]  <= STRESS_MAX)


def _build_user_msg(H, result, history, store):
    defl_ok   = result["max_deflection"] <= DELTA_MAX
    stress_ok = result["max_von_mises"]  <= STRESS_MAX

    hist_lines = []
    for h in history[-6:]:
        r = h["result"]
        hist_lines.append(
            f"  H={h['H']*1e3:.2f}mm: defl={r['max_deflection']*1e3:.4f}mm  "
            f"stress={r['max_von_mises']/1e6:.2f}MPa  "
            f"mass={r['mass_per_depth']:.4f}kg/m  "
            f"{'OK' if h['feasible'] else 'INFEASIBLE'}"
        )

    rag_hits  = store.query_runs(
        f"H={H*1e3:.1f}mm deflection={result['max_deflection']*1e3:.3f}mm "
        f"{'feasible' if _feasible(result) else 'infeasible'}",
        n_results=3,
    )
    rag_block = "\n".join(f"  - {h['document']}" for h in rag_hits) or "  (none)"

    feasible_hist = [h for h in history if h["feasible"]]
    best_note = ""
    if feasible_hist:
        best = min(feasible_hist, key=lambda h: h["result"]["mass_per_depth"])
        best_note = (f"  Best feasible so far: H={best['H']*1e3:.2f}mm  "
                     f"mass={best['result']['mass_per_depth']:.4f}kg/m\n")

    return (
        f"Current design: H={H*1e3:.2f}mm\n"
        f"Simulation result:\n"
        f"  max_deflection = {result['max_deflection']*1e3:.4f} mm  "
        f"(limit {DELTA_MAX*1e3:.3f} mm)  [{'OK' if defl_ok else 'VIOLATED'}]\n"
        f"  max_von_mises  = {result['max_von_mises']/1e6:.2f} MPa  "
        f"(limit {STRESS_MAX/1e6:.1f} MPa)  [{'OK' if stress_ok else 'VIOLATED'}]\n"
        f"  mass_per_depth = {result['mass_per_depth']:.4f} kg/m\n"
        f"{best_note}"
        f"\nIteration history:\n" + "\n".join(hist_lines) +
        f"\nSimilar RAG runs:\n{rag_block}\n\nPropose next H."
    )


def run_beam_optimization(params: dict, api_key: str):
    """Generator: yields JSON-serialisable event dicts for each step."""
    L = float(params.get("L",      0.100))
    P = float(params.get("load",   500.0))
    H = float(params.get("H_init", 0.006))

    db_path = str(Path(__file__).parent.parent.parent / "chroma_db")
    store   = SimulationStore(path=db_path)
    client  = anthropic.Anthropic(api_key=api_key)

    history = []
    best_H, best_mass = None, float("inf")

    yield {"type": "start", "L_mm": round(L*1e3, 1), "load_N": P, "H_init_mm": round(H*1e3, 2)}

    for iteration in range(MAX_ITER):
        result   = simulate({"L": L, "H": H, "load": P})
        feasible = _feasible(result)

        store.log_run({"L": L, "H": H, "load": P}, result, f"dashboard iter {iteration+1}")
        history.append({"H": H, "result": result, "feasible": feasible})

        yield {
            "type":           "iteration",
            "iter":           iteration + 1,
            "H_mm":           round(H * 1e3, 3),
            "deflection_mm":  round(result["max_deflection"] * 1e3, 4),
            "stress_mpa":     round(result["max_von_mises"] / 1e6, 1),
            "mass":           round(result["mass_per_depth"], 4),
            "feasible":       feasible,
        }

        if feasible and result["mass_per_depth"] < best_mass:
            best_mass = result["mass_per_depth"]
            best_H    = H

        # Convergence check
        feasible_hist = [h for h in history if h["feasible"]]
        if len(feasible_hist) >= 2:
            prev_mass = feasible_hist[-2]["result"]["mass_per_depth"]
            curr_mass = feasible_hist[-1]["result"]["mass_per_depth"]
            if abs(prev_mass - curr_mass) / prev_mass < CONV_TOL:
                break

        # Agent reasoning
        try:
            raw      = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_msg(H, result, history, store)}],
            ).content[0].text.strip()
            proposal  = json.loads(raw)
            new_H     = float(proposal["H"])
            reasoning = str(proposal.get("reasoning", ""))
        except Exception as exc:
            new_H     = H * 0.95 if feasible else H * 1.10
            reasoning = f"[fallback: {str(exc)[:50]}]"

        new_H = max(H_MIN, min(H_MAX, new_H))
        yield {"type": "reasoning", "iter": iteration + 1,
               "next_H_mm": round(new_H * 1e3, 3), "reasoning": reasoning}
        H = new_H

    yield {
        "type":                  "done",
        "best_H_mm":             round(best_H * 1e3, 3) if best_H else None,
        "best_mass":             round(best_mass, 4)    if best_H else None,
        "iterations":            len(history),
        "scipy_baseline_H_mm":   8.308,
        "scipy_baseline_mass":   2.2432,
    }
