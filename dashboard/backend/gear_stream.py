"""Streaming gear optimization — yields event dicts at each agent iteration."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # aero-opt root
sys.path.insert(0, str(Path(__file__).parent))                  # dashboard/backend

import anthropic
from simulate_gear import simulate, _feasible, SIGMA_B_ALLOW, SIGMA_C_ALLOW, N_TEETH
from rag_store import SimulationStore

M_MIN, M_MAX = 1.5, 6.0
B_MIN, B_MAX = 10.0, 80.0
MAX_ITER     = 8
CONV_TOL     = 0.008

SYSTEM_PROMPT = f"""You are a mechanical gear design assistant inside an automated optimization loop.

PROBLEM: Minimize mass of a spur gear pinion.
  Fixed: N = {N_TEETH} teeth, input torque T = 50 N*m, gear ratio = 3:1
  Material: AISI 4140 through-hardened (250 HB), rho = 7850 kg/m^3
  Design variables:
    m -- module (mm), bounds [{M_MIN}, {M_MAX}] mm
    b -- face width (mm), bounds [{B_MIN}, {B_MAX}] mm

CONSTRAINTS (both must be satisfied):
  bending_stress (Lewis) <= {SIGMA_B_ALLOW} MPa
  contact_stress (Hertz) <= {SIGMA_C_ALLOW} MPa

PHYSICS:
  F_t = 2000*T / (m*N)  — larger m reduces tangential force
  Bending ~ 1/(m^2 * b), Contact ~ 1/sqrt(m^2 * b), Mass ~ m^2 * b
  Contact stress is typically binding. Face ratio guideline: 8*m <= b <= 16*m.

RESPOND with ONLY valid JSON:
{{"m": <float_mm>, "b": <float_mm>, "reasoning": "<engineering rationale, 1-2 sentences>"}}"""


def _build_user_msg(m, b, result, history, store):
    bend_ok    = result["bending_stress"] <= SIGMA_B_ALLOW
    contact_ok = result["contact_stress"] <= SIGMA_C_ALLOW

    hist_lines = []
    for h in history[-6:]:
        r = h["result"]
        hist_lines.append(
            f"  m={h['m']:.2f}mm b={h['b']:.1f}mm: "
            f"bend={r['bending_stress']:.1f}MPa  "
            f"contact={r['contact_stress']:.1f}MPa  "
            f"mass={r['mass']*1000:.1f}g  "
            f"{'OK' if h['feasible'] else 'INFEASIBLE'}"
        )

    try:
        rag_hits = store.query_runs(
            f"m={m:.2f}mm b={b:.1f}mm bending={result['bending_stress']:.1f}MPa",
            n_results=3,
        )
    except Exception:
        rag_hits = []
    rag_block = "\n".join(f"  - {h['document']}" for h in rag_hits) or "  (none)"

    feasible_hist = [h for h in history if h["feasible"]]
    best_note = ""
    if feasible_hist:
        best = min(feasible_hist, key=lambda h: h["result"]["mass"])
        best_note = (f"  Best feasible so far: m={best['m']:.2f}mm b={best['b']:.1f}mm  "
                     f"mass={best['result']['mass']*1000:.1f}g\n")

    return (
        f"Current design: m={m:.2f}mm, b={b:.1f}mm\n"
        f"Simulation result:\n"
        f"  bending_stress = {result['bending_stress']:.1f} MPa  "
        f"(limit {SIGMA_B_ALLOW} MPa)  [{'OK' if bend_ok else 'VIOLATED'}]\n"
        f"  contact_stress = {result['contact_stress']:.1f} MPa  "
        f"(limit {SIGMA_C_ALLOW} MPa)  [{'OK' if contact_ok else 'VIOLATED'}]\n"
        f"  mass = {result['mass']*1000:.1f} g\n"
        f"  pitch_diam = {result['pitch_diam_mm']:.1f} mm\n"
        f"{best_note}"
        f"\nIteration history:\n" + "\n".join(hist_lines) +
        f"\nSimilar RAG runs:\n{rag_block}\n\nPropose next (m, b)."
    )


def run_gear_optimization(params: dict, api_key: str):
    """Generator: yields JSON-serialisable event dicts for each step."""
    m      = float(params.get("m_init", 2.0))
    b      = float(params.get("b_init", 25.0))
    torque = float(params.get("torque", 50.0))

    db_path = str(Path(__file__).parent.parent.parent / "chroma_db_gear")
    store   = SimulationStore(path=db_path)
    client  = anthropic.Anthropic(api_key=api_key)

    history = []
    best_m, best_b, best_mass = None, None, float("inf")

    yield {"type": "start", "m_init_mm": round(m, 2), "b_init_mm": round(b, 1), "torque_nm": torque}

    for iteration in range(MAX_ITER):
        result   = simulate({"m": m, "b": b, "torque": torque})
        feasible = _feasible(result)

        store.log_run({"m": m, "b": b, "torque": torque}, result, f"dashboard iter {iteration+1}")
        history.append({"m": m, "b": b, "result": result, "feasible": feasible})

        yield {
            "type":           "iteration",
            "iter":           iteration + 1,
            "m_mm":           round(m, 3),
            "b_mm":           round(b, 1),
            "bending_mpa":    round(result["bending_stress"], 1),
            "contact_mpa":    round(result["contact_stress"], 1),
            "mass_g":         round(result["mass"] * 1000, 1),
            "pitch_diam_mm":  round(result["pitch_diam_mm"], 1),
            "face_ratio":     round(result["face_ratio"], 1),
            "feasible":       feasible,
        }

        if feasible and result["mass"] < best_mass:
            best_mass = result["mass"]
            best_m, best_b = m, b

        # Convergence
        feasible_hist = [h for h in history if h["feasible"]]
        if len(feasible_hist) >= 2:
            prev = feasible_hist[-2]["result"]["mass"]
            curr = feasible_hist[-1]["result"]["mass"]
            if abs(prev - curr) / prev < CONV_TOL:
                break

        # Agent reasoning
        try:
            raw      = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                          "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": _build_user_msg(m, b, result, history, store)}],
            ).content[0].text.strip()
            proposal  = json.loads(raw)
            new_m     = float(proposal["m"])
            new_b     = float(proposal["b"])
            reasoning = str(proposal.get("reasoning", ""))
        except Exception as exc:
            new_m     = m * (0.95 if feasible else 1.10)
            new_b     = b * (0.95 if feasible else 1.10)
            reasoning = f"[fallback: {str(exc)[:50]}]"

        new_m = max(M_MIN, min(M_MAX, new_m))
        new_b = max(B_MIN, min(B_MAX, new_b))
        yield {"type": "reasoning", "iter": iteration + 1,
               "next_m_mm": round(new_m, 3), "next_b_mm": round(new_b, 1), "reasoning": reasoning}
        m, b = new_m, new_b

    yield {
        "type":              "done",
        "best_m_mm":         round(best_m, 3)       if best_m else None,
        "best_b_mm":         round(best_b, 1)        if best_b else None,
        "best_mass_g":       round(best_mass * 1000, 1) if best_m else None,
        "iterations":        len(history),
        "scipy_baseline_m":  4.30,
        "scipy_baseline_b":  30.1,
        "scipy_baseline_g":  845,
    }
