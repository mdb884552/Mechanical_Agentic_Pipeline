"""
Gear Optimization Agent -- LangGraph + Claude API
===================================================
Mirrors agent.py but for 2-variable gear optimization.
Agent proposes (m, b) pairs to minimize mass subject to
AGMA bending and contact stress constraints.

Usage:
    python agent_gear.py
    python agent_gear.py --local --model llama3.2
"""

import os, json, argparse, requests, anthropic
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

from simulate_gear import (simulate, _feasible, nearest_std_module,
                            SIGMA_B_ALLOW, SIGMA_C_ALLOW, N_TEETH,
                            LEWIS_Y, C_P, K_O, I_FACTOR, RHO)
from rag_store import SimulationStore

# ── CLI ────────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--local", action="store_true")
_p.add_argument("--model", default=None)
ARGS      = _p.parse_args()
USE_LOCAL = ARGS.local
MODEL     = ARGS.model or ("llama3.2" if USE_LOCAL else "claude-sonnet-4-6")

# ── Constants ──────────────────────────────────────────────────────────────────
TORQUE   = 50.0
M_MIN, M_MAX = 1.5, 6.0
B_MIN, B_MAX = 10.0, 80.0
MAX_ITER = 14
CONV_TOL = 0.008

SCIPY_BASELINE = None   # filled after optimizer run (or set manually)

SYSTEM_PROMPT = f"""You are a mechanical gear design assistant inside an automated optimization loop.

PROBLEM: Minimize the mass of a spur gear pinion.
  Fixed: N = {N_TEETH} teeth, input torque T = {TORQUE} N*m, gear ratio = 3:1
  Material: AISI 4140 through-hardened (250 HB), rho = 7850 kg/m^3
  Design variables:
    m -- module (mm), bounds [{M_MIN}, {M_MAX}] mm
    b -- face width (mm), bounds [{B_MIN}, {B_MAX}] mm

CONSTRAINTS (both must be satisfied):
  bending_stress  (Lewis)  <= {SIGMA_B_ALLOW} MPa
  contact_stress  (Hertz)  <= {SIGMA_C_ALLOW} MPa

PHYSICS:
  Tangential force: F_t = 2000*T / (m*N)    -- larger m reduces force
  Bending:  sigma_b = F_t / (b * m * Y)     -- scales as 1/(m^2 * b)
  Contact:  sigma_c = C_p * sqrt(F_t*Ko/(b*d*I)) -- scales as 1/sqrt(m^2 * b)
  Mass:     mass = rho * pi/4 * (m*N)^2 * b -- scales as m^2 * b

  Contact stress is typically the binding constraint for N=15 at this torque.
  The minimum mass is approximately constant along the contact constraint boundary.
  Both m and b must be chosen together. Face width guideline: 8*m <= b <= 16*m.

RESPOND with ONLY valid JSON -- no other text, no markdown:
{{"m": <float_mm>, "b": <float_mm>, "reasoning": "<engineering rationale, 1-2 sentences>"}}"""


# ── LLM routing ────────────────────────────────────────────────────────────────
def _call_llm(user_msg):
    return _call_ollama(user_msg) if USE_LOCAL else _call_claude(user_msg)

def _call_claude(user_msg):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Use --local or add to .env")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text.strip()

def _call_ollama(user_msg):
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": MODEL, "stream": False, "format": "json",
                  "options": {"temperature": 0.1},
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": user_msg}]},
            timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama not running. Run: ollama serve")


# ── State ──────────────────────────────────────────────────────────────────────
class GearState(TypedDict):
    params:      dict
    result:      dict | None
    iteration:   int
    converged:   bool
    best_params: dict | None
    best_result: dict | None
    best_mass:   float
    history:     list[dict]
    agent_trace: list[str]

store = SimulationStore(db_path="./chroma_db_gear")


# ── Nodes ──────────────────────────────────────────────────────────────────────
def run_simulation_node(state):
    p = state['params']
    r = simulate(p)
    feas = _feasible(r)
    note = ("FEASIBLE" if feas else
            f"INFEASIBLE sb={r['bending_stress']:.0f}MPa sc={r['contact_stress']:.0f}MPa")
    store.log_run(p, r, f"agent iter {state['iteration']+1} -- {note}")

    print(f"\n  [Iter {state['iteration']+1:2d}]  "
          f"m={p['m']:5.2f}mm  b={p['b']:5.1f}mm  "
          f"sigma_b={r['bending_stress']:6.1f}MPa  "
          f"sigma_c={r['contact_stress']:6.1f}MPa  "
          f"mass={r['mass']*1000:6.1f}g  "
          f"{'[OK]' if feas else '[INFEASIBLE]'}")

    history = state['history'] + [{'params': dict(p), 'result': r, 'feasible': feas}]
    return {'result': r, 'history': history, 'iteration': state['iteration'] + 1}


def evaluate_node(state):
    r = state['result']
    feas = _feasible(r)
    best_mass   = state['best_mass']
    best_params = state['best_params']
    best_result = state['best_result']
    converged   = False

    if feas:
        m_cur = r['mass']
        if best_params is not None:
            improvement = (best_mass - m_cur) / best_mass
            if abs(improvement) < CONV_TOL:
                converged = True
        if m_cur < best_mass:
            best_mass   = m_cur
            best_params = dict(state['params'])
            best_result = r

    if state['iteration'] >= MAX_ITER:
        converged = True

    return {'converged': converged, 'best_mass': best_mass,
            'best_params': best_params, 'best_result': best_result}


def agent_reason_node(state):
    r    = state['result']
    p    = state['params']
    feas = _feasible(r)

    query = (f"m={p['m']:.1f}mm b={p['b']:.0f}mm "
             f"sigma_b={r['bending_stress']:.0f}MPa "
             f"sigma_c={r['contact_stress']:.0f}MPa "
             f"{'feasible' if feas else 'infeasible'}")
    hits  = store.query_runs(query, n_results=3)
    rag   = "\n".join(f"  - {h['document']}" for h in hits) or "  (none yet)"

    hist_lines = []
    for h in state['history'][-6:]:
        hp, hr = h['params'], h['result']
        hist_lines.append(
            f"  m={hp['m']:.2f}mm b={hp['b']:.1f}mm: "
            f"sb={hr['bending_stress']:.0f}MPa sc={hr['contact_stress']:.0f}MPa "
            f"mass={hr['mass']*1000:.0f}g {'OK' if h['feasible'] else 'INFEASIBLE'}"
        )

    best_note = ""
    if state['best_params']:
        bp, bm = state['best_params'], state['best_mass']
        best_note = f"  Best feasible: m={bp['m']:.2f}mm b={bp['b']:.1f}mm mass={bm*1000:.0f}g\n"

    b_ok = r['bending_stress'] <= SIGMA_B_ALLOW
    c_ok = r['contact_stress']  <= SIGMA_C_ALLOW

    user_msg = (
        f"Current design: m={p['m']:.2f} mm, b={p['b']:.1f} mm\n"
        f"Results:\n"
        f"  bending_stress = {r['bending_stress']:.1f} MPa  (limit {SIGMA_B_ALLOW} MPa)  "
        f"[{'OK' if b_ok else 'VIOLATED'}]\n"
        f"  contact_stress = {r['contact_stress']:.1f} MPa  (limit {SIGMA_C_ALLOW} MPa)  "
        f"[{'OK' if c_ok else 'VIOLATED'}]\n"
        f"  mass           = {r['mass']*1000:.1f} g\n"
        f"  pitch_diameter = {r['pitch_diam_mm']:.1f} mm\n"
        f"  face_ratio b/m = {r['face_ratio']:.1f}  (guideline 8-16)\n"
        f"{best_note}"
        f"\nIteration history:\n" + "\n".join(hist_lines) + "\n"
        f"\nSimilar designs from RAG:\n{rag}\n"
        f"\nPropose next (m, b). Minimize mass while meeting both constraints."
    )

    raw = _call_llm(user_msg)

    try:
        prop = json.loads(raw)
        new_m = float(prop['m'])
        new_b = float(prop['b'])
        reason = str(prop.get('reasoning', ''))
    except (json.JSONDecodeError, KeyError, ValueError):
        # fallback: increase both by 10% if infeasible, decrease b by 5% if feasible
        if not feas:
            new_m = p['m'] * 1.10
            new_b = p['b'] * 1.15
        else:
            new_m = p['m']
            new_b = p['b'] * 0.95
        reason = f"[parse fallback] raw={raw[:60]}"

    new_m = float(np.clip(new_m, M_MIN, M_MAX))
    new_b = float(np.clip(new_b, B_MIN, B_MAX))

    safe_reason = reason.encode('ascii', errors='replace').decode('ascii')
    print(f"  Agent -> m={new_m:.2f}mm  b={new_b:.1f}mm  | {safe_reason[:85]}")

    trace = state['agent_trace'] + [
        f"Iter {state['iteration']:2d}: m={new_m:.2f}mm b={new_b:.1f}mm -- {reason}"
    ]
    return {'params': {'m': new_m, 'b': new_b, 'torque': TORQUE},
            'agent_trace': trace}


# ── Graph ──────────────────────────────────────────────────────────────────────
def route(state):
    return "end" if state['converged'] else "agent_reason"

def build_graph():
    g = StateGraph(GearState)
    g.add_node("run_simulation", run_simulation_node)
    g.add_node("evaluate",       evaluate_node)
    g.add_node("agent_reason",   agent_reason_node)
    g.set_entry_point("run_simulation")
    g.add_edge("run_simulation", "evaluate")
    g.add_conditional_edges("evaluate", route, {"end": END, "agent_reason": "agent_reason"})
    g.add_edge("agent_reason", "run_simulation")
    return g.compile()


# ── Visualization ──────────────────────────────────────────────────────────────
def plot_results(final_state, initial_params):
    history = final_state['history']
    if not history:
        return

    ms   = [initial_params['m']] + [h['params']['m'] for h in history]
    bs   = [initial_params['b']] + [h['params']['b'] for h in history]
    masses  = [h['result']['mass']*1000 for h in history]
    sigs_b  = [h['result']['bending_stress'] for h in history]
    sigs_c  = [h['result']['contact_stress']  for h in history]
    feas    = [h['feasible'] for h in history]

    from simulate_gear import simulate as _sim
    m_grid = np.linspace(1.5, 6.0, 100)
    b_grid = np.linspace(10,  80,  100)
    MM, BB = np.meshgrid(m_grid, b_grid)
    SB = np.vectorize(lambda m, b: _sim({'m': m, 'b': b})['bending_stress'])(MM, BB)
    SC = np.vectorize(lambda m, b: _sim({'m': m, 'b': b})['contact_stress'])(MM, BB)
    MASS_G = np.vectorize(lambda m, b: _sim({'m': m, 'b': b})['mass']*1000)(MM, BB)
    infeasible = (SB > SIGMA_B_ALLOW) | (SC > SIGMA_C_ALLOW)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="#F8F9FA")
    bp = final_state['best_params']
    br = final_state['best_result']
    fig.suptitle(
        f"Gear Optimization Agent — LangGraph + Claude  "
        f"(N={N_TEETH}, T={TORQUE} N·m)\n"
        f"Best: m={bp['m']:.2f} mm  b={bp['b']:.1f} mm  "
        f"mass={br['mass']*1000:.0f} g  "
        f"σ_b={br['bending_stress']:.0f} MPa  σ_c={br['contact_stress']:.0f} MPa",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: trajectory in (m,b) space
    ax1 = axes[0]
    ax1.contourf(MM, BB, np.where(infeasible,1,0), levels=[0.5,1.5],
                 colors=["#FFCCCC"], alpha=0.45)
    cs_b = ax1.contour(MM, BB, SB, levels=[SIGMA_B_ALLOW], colors=["#D62828"], lw=2)
    cs_c = ax1.contour(MM, BB, SC, levels=[SIGMA_C_ALLOW], colors=["#1D6FA4"], lw=2)
    ax1.clabel(cs_b, fmt=f"σ_b={SIGMA_B_ALLOW:.0f}", fontsize=8)
    ax1.clabel(cs_c, fmt=f"σ_c={SIGMA_C_ALLOW:.0f}", fontsize=8)
    cs_m = ax1.contour(MM, BB, MASS_G, levels=[500,600,700,800,1000,1300],
                       colors=["#888"], lw=0.7, linestyles="dashed")
    ax1.clabel(cs_m, fmt="%d g", fontsize=7)

    colors = ['#2CA02C' if f else '#D62828' for f in feas]
    for i in range(len(ms)-1):
        ax1.annotate("", xy=(ms[i+1], bs[i+1]), xytext=(ms[i], bs[i]),
                     arrowprops=dict(arrowstyle="-|>", color="#444",
                                     lw=1.2, mutation_scale=10))
    ax1.scatter(ms[1:], bs[1:], c=colors, zorder=5, s=50)
    ax1.plot(ms[0], bs[0], "s", color="#FF7F0E", ms=10, zorder=6, label="Start")
    if bp:
        m_std = nearest_std_module(bp['m'])
        ax1.plot(bp['m'], bp['b'], "*", color="#2CA02C", ms=16, zorder=7,
                 label=f"Best  m={bp['m']:.2f}mm")
        ax1.plot(m_std, bp['b'], "D", color="#9467BD", ms=10, zorder=7,
                 label=f"Std m={m_std}mm")

    infeas_p = mpatches.Patch(color="#FFCCCC", alpha=0.5, label="Infeasible")
    ax1.legend(handles=ax1.get_legend_handles_labels()[0]+[infeas_p],
               labels=ax1.get_legend_handles_labels()[1]+["Infeasible"],
               fontsize=8)
    ax1.set_xlabel("Module m (mm)"); ax1.set_ylabel("Face width b (mm)")
    ax1.set_title("Agent Trajectory in Design Space", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    iters = list(range(1, len(history)+1))

    # Panel 2: mass vs iteration
    ax2 = axes[1]
    ax2.plot(iters, masses, 'k-o', ms=5, lw=1.5, zorder=4)
    ax2.scatter(iters, masses, c=colors, zorder=5, s=50)
    if final_state['best_mass'] < float('inf'):
        ax2.axhline(final_state['best_mass']*1000, color='#2CA02C',
                    lw=1.2, ls='--', label=f"Best = {final_state['best_mass']*1000:.0f} g")
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Mass (g)")
    ax2.set_title("Mass per Iteration", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    feas_p   = mpatches.Patch(color='#2CA02C', label='Feasible')
    infeas_p2 = mpatches.Patch(color='#D62828', label='Infeasible')
    ax2.legend([feas_p, infeas_p2]+ax2.get_legend_handles_labels()[0],
               ['Feasible','Infeasible']+ax2.get_legend_handles_labels()[1], fontsize=8)

    # Panel 3: stress vs iteration
    ax3 = axes[2]
    ax3.plot(iters, sigs_b, 's-', color='#D62828', ms=5, lw=1.5, label="σ_b bending")
    ax3.plot(iters, sigs_c, '^-', color='#1D6FA4', ms=5, lw=1.5, label="σ_c contact")
    ax3.axhline(SIGMA_B_ALLOW, color='#D62828', lw=1.0, ls=':', alpha=0.7,
                label=f"Limit σ_b = {SIGMA_B_ALLOW} MPa")
    ax3.axhline(SIGMA_C_ALLOW, color='#1D6FA4', lw=1.0, ls=':', alpha=0.7,
                label=f"Limit σ_c = {SIGMA_C_ALLOW} MPa")
    ax3.set_xlabel("Iteration"); ax3.set_ylabel("Stress (MPa)")
    ax3.set_title("Stress Convergence", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("agent_gear.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print("\n  Plot saved: agent_gear.png")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    backend = f"Ollama ({MODEL})" if USE_LOCAL else f"Claude API ({MODEL})"
    print(f"\n{'='*60}")
    print(f"  Gear Optimization Agent")
    print(f"  Backend  : {backend}")
    print(f"  Problem  : N={N_TEETH} teeth, T={TORQUE} N*m, AISI 4140")
    print(f"  Limits   : sigma_b<={SIGMA_B_ALLOW} MPa  sigma_c<={SIGMA_C_ALLOW} MPa")
    print(f"  RAG store: {store.run_count()} runs  {store.lesson_count()} lessons")
    print(f"{'='*60}")

    initial = {'m': 2.0, 'b': 20.0, 'torque': TORQUE}
    init_state: GearState = {
        'params': initial, 'result': None, 'iteration': 0,
        'converged': False, 'best_params': None, 'best_result': None,
        'best_mass': float('inf'), 'history': [], 'agent_trace': [],
    }

    graph = build_graph()
    final = graph.invoke(init_state)

    print(f"\n{'='*60}")
    print(f"  Optimization Complete  ({final['iteration']} iterations)")
    print(f"{'='*60}")

    if final['best_params']:
        bp = final['best_params']
        br = final['best_result']
        m_std = nearest_std_module(bp['m'])
        r_std = simulate({'m': m_std, 'b': bp['b'], 'torque': TORQUE})

        print(f"\n  Best feasible design (continuous):")
        print(f"    m            = {bp['m']:.3f} mm")
        print(f"    b            = {bp['b']:.1f} mm")
        print(f"    pitch diam   = {br['pitch_diam_mm']:.1f} mm")
        print(f"    bending      = {br['bending_stress']:.1f} MPa  (limit {SIGMA_B_ALLOW})")
        print(f"    contact      = {br['contact_stress']:.1f} MPa  (limit {SIGMA_C_ALLOW})")
        print(f"    mass         = {br['mass']*1000:.1f} g")
        print(f"\n  Snapped to standard module m = {m_std} mm:")
        print(f"    bending      = {r_std['bending_stress']:.1f} MPa  "
              f"{'[OK]' if r_std['bending_stress']<=SIGMA_B_ALLOW else '[VIOLATED]'}")
        print(f"    contact      = {r_std['contact_stress']:.1f} MPa  "
              f"{'[OK]' if r_std['contact_stress']<=SIGMA_C_ALLOW else '[VIOLATED]'}")
        print(f"    mass         = {r_std['mass']*1000:.1f} g")
        print(f"    face ratio   = {r_std['face_ratio']:.1f}  (guideline 8-16)")

        print(f"\n  Agent reasoning trace:")
        for line in final['agent_trace']:
            safe = line.encode('ascii', errors='replace').decode('ascii')
            print(f"    {safe}")

        plot_results(final, initial)
    else:
        print("  No feasible design found within iteration limit.")

    print(f"\n  Total runs in RAG: {store.run_count()}")
    print(f"{'='*60}\n")
