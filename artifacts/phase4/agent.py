"""
Phase 4 -- LangGraph AI Optimization Agent
============================================
A structured state-machine workflow (LangGraph) where an AI agent reasons
over simulation results and RAG-retrieved context to propose design
improvements -- with full observability at every node.

Workflow
--------
  START
    |
    v
  run_simulation --> evaluate --> [converged] --> END
                              --> agent_reason --> run_simulation (loop)

Nodes
-----
  run_simulation  Call simulate(), log result to RAG store
  evaluate        Check constraints, track best feasible design, detect convergence
  agent_reason    Query RAG, call LLM, parse proposed new H

Usage
-----
  Claude API (best quality):
    python agent.py
    python agent.py --model claude-opus-4-7

  Local Ollama (no API key required):
    ollama pull llama3.2
    python agent.py --local
    python agent.py --local --model qwen2.5

Setup
-----
  1. Run seed_rag.py first
  2. For Claude: create .env with ANTHROPIC_API_KEY=sk-ant-...
  3. For Ollama: install Ollama, run 'ollama serve', pull a model
"""

import os
import json
import argparse
import requests
import anthropic
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

from simulate import simulate, _YIELD_PA
from rag_store import SimulationStore

# ── CLI args ───────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Phase 4 -- LangGraph Optimization Agent")
_parser.add_argument("--local", action="store_true",
                     help="Use local Ollama LLM instead of Claude API")
_parser.add_argument("--model", default=None,
                     help="Override model name (default: claude-sonnet-4-6 or llama3.2)")
ARGS     = _parser.parse_args()
USE_LOCAL = ARGS.local
MODEL     = ARGS.model or ("llama3.2" if USE_LOCAL else "claude-sonnet-4-6")

# ── Constants ──────────────────────────────────────────────────────────────────
L           = 0.100
P           = 500.0
DELTA_MAX   = 0.050e-3
STRESS_MAX  = 0.6 * _YIELD_PA   # 165.6 MPa
H_MIN       = 0.003
H_MAX       = 0.030
MAX_ITER    = 10
CONV_TOL    = 0.005             # converge when mass improvement < 0.5%

SCIPY_BASELINE = {'H_mm': 8.308, 'mass': 2.2432}

SYSTEM_PROMPT = f"""You are an aerospace structural optimization assistant inside an automated design loop.

PROBLEM: Minimize mass of a 2D Al-6061 cantilever beam.
  Fixed: L = 100 mm, tip load P = 500 N
  Design variable: H (beam height, meters), bounds [{H_MIN*1e3:.0f}mm, {H_MAX*1e3:.0f}mm]

CONSTRAINTS (all must be satisfied):
  max_deflection <= {DELTA_MAX*1e3:.3f} mm   (stiffness limit)
  max_von_mises  <= {STRESS_MAX/1e6:.1f} MPa  (60% of Al-6061 yield, safety factor 1.67)

PHYSICS REMINDER:
  Deflection ~ 1/H^3. The constraint boundary is near H = 8.3 mm analytically.
  Stress is far from yield for this load case -- deflection is the binding constraint.
  Approach the boundary from above (feasible side) to avoid infeasible oscillation.

YOUR TASK: Propose the next H to simulate. Balance:
  1. Feasibility (meet both constraints)
  2. Mass reduction (smaller H = less mass)
  3. Use RAG context to avoid repeating failed designs

RESPOND with ONLY a valid JSON object -- no other text, no markdown:
{{"H": <float_meters>, "reasoning": "<engineering rationale, 1-2 sentences>"}}"""


# ── LLM routing ────────────────────────────────────────────────────────────────

def _call_llm(user_msg: str) -> str:
    """Route to Claude API or Ollama based on --local flag."""
    return _call_ollama(user_msg) if USE_LOCAL else _call_claude(user_msg)


def _call_claude(user_msg: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "\n\nANTHROPIC_API_KEY not set.\n"
            "Add it to .env:  ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or run with --local to use Ollama instead.\n"
        )
    client   = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


def _call_ollama(user_msg: str) -> str:
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model":   MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "format":  "json",   # forces valid JSON output
                "stream":  False,
                "options": {"temperature": 0.1},  # low temp = consistent JSON
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "\n\nOllama is not running.\n"
            "Start it:      ollama serve\n"
            f"Pull a model:  ollama pull {MODEL}\n"
        )
    except requests.exceptions.HTTPError as e:
        if "model" in str(e).lower():
            raise RuntimeError(
                f"\n\nOllama model '{MODEL}' not found.\n"
                f"Pull it first:  ollama pull {MODEL}\n"
            )
        raise


def _check_ollama() -> bool:
    """Return True if Ollama is reachable and the requested model is available."""
    try:
        models = requests.get("http://localhost:11434/api/tags", timeout=5).json()
        names  = [m["name"].split(":")[0] for m in models.get("models", [])]
        return MODEL.split(":")[0] in names
    except Exception:
        return False


# ── State ──────────────────────────────────────────────────────────────────────

class OptState(TypedDict):
    params:      dict
    result:      dict | None
    iteration:   int
    converged:   bool
    best_params: dict | None
    best_result: dict | None
    best_mass:   float
    history:     list[dict]
    agent_trace: list[str]


def _feasible(result: dict) -> bool:
    return (result['max_deflection'] <= DELTA_MAX and
            result['max_von_mises']  <= STRESS_MAX)


store = SimulationStore()


# ── Nodes ──────────────────────────────────────────────────────────────────────

def run_simulation_node(state: OptState) -> dict:
    params   = state['params']
    result   = simulate(params)
    feasible = _feasible(result)

    note = ("FEASIBLE" if feasible else
            f"INFEASIBLE defl={result['max_deflection']*1e3:.4f}mm "
            f"stress={result['max_von_mises']/1e6:.1f}MPa")
    store.log_run(params, result, f"agent iter {state['iteration']+1} -- {note}")

    print(f"\n  [Iter {state['iteration']+1:2d}] "
          f"H={params['H']*1e3:6.2f}mm  "
          f"defl={result['max_deflection']*1e3:.4f}mm  "
          f"stress={result['max_von_mises']/1e6:6.2f}MPa  "
          f"mass={result['mass_per_depth']:.4f}kg/m  "
          f"{'[OK]' if feasible else '[INFEASIBLE]'}")

    history = state['history'] + [
        {'params': dict(params), 'result': result, 'feasible': feasible}
    ]
    return {'result': result, 'history': history, 'iteration': state['iteration'] + 1}


def evaluate_node(state: OptState) -> dict:
    result   = state['result']
    feasible = _feasible(result)

    best_mass   = state['best_mass']
    best_params = state['best_params']
    best_result = state['best_result']
    converged   = False

    if feasible:
        current_mass = result['mass_per_depth']
        if best_params is not None:
            improvement = (best_mass - current_mass) / best_mass
            if -CONV_TOL < improvement < CONV_TOL:
                converged = True
        if current_mass < best_mass:
            best_mass   = current_mass
            best_params = dict(state['params'])
            best_result = result

    if state['iteration'] >= MAX_ITER:
        converged = True

    return {
        'converged':   converged,
        'best_mass':   best_mass,
        'best_params': best_params,
        'best_result': best_result,
    }


def agent_reason_node(state: OptState) -> dict:
    result   = state['result']
    feasible = _feasible(result)

    # Query RAG
    query    = (f"H={state['params']['H']*1e3:.1f}mm "
                f"deflection={result['max_deflection']*1e3:.3f}mm "
                f"{'feasible' if feasible else 'infeasible'}")
    rag_hits  = store.query_runs(query, n_results=3)
    rag_block = "\n".join(f"  - {h['document']}" for h in rag_hits) or "  (none yet)"

    hist_lines = []
    for h in state['history'][-6:]:
        hist_lines.append(
            f"  H={h['params']['H']*1e3:.2f}mm: "
            f"defl={h['result']['max_deflection']*1e3:.4f}mm  "
            f"stress={h['result']['max_von_mises']/1e6:.2f}MPa  "
            f"mass={h['result']['mass_per_depth']:.4f}kg/m  "
            f"{'OK' if h['feasible'] else 'INFEASIBLE'}"
        )
    hist_block = "\n".join(hist_lines) if hist_lines else "  (first iteration)"

    defl_ok   = result['max_deflection'] <= DELTA_MAX
    stress_ok = result['max_von_mises']  <= STRESS_MAX
    best_note = ""
    if state['best_params']:
        bp = state['best_params']
        best_note = (f"  Best feasible so far: "
                     f"H={bp['H']*1e3:.2f}mm  mass={state['best_mass']:.4f}kg/m\n")

    user_msg = (
        f"Current design: H={state['params']['H']*1e3:.2f}mm\n"
        f"Simulation result:\n"
        f"  max_deflection = {result['max_deflection']*1e3:.4f} mm  "
        f"(limit {DELTA_MAX*1e3:.3f} mm)  [{'OK' if defl_ok else 'VIOLATED'}]\n"
        f"  max_von_mises  = {result['max_von_mises']/1e6:.2f} MPa  "
        f"(limit {STRESS_MAX/1e6:.1f} MPa)  [{'OK' if stress_ok else 'VIOLATED'}]\n"
        f"  mass_per_depth = {result['mass_per_depth']:.4f} kg/m\n"
        f"{best_note}"
        f"\nIteration history:\n{hist_block}\n"
        f"\nSimilar runs from RAG database:\n{rag_block}\n"
        f"\nPropose the next H."
    )

    raw = _call_llm(user_msg)

    try:
        proposal  = json.loads(raw)
        new_H     = float(proposal['H'])
        reasoning = str(proposal.get('reasoning', ''))
    except (json.JSONDecodeError, KeyError, ValueError):
        new_H     = (state['params']['H'] * 1.10
                     if not feasible else state['params']['H'] * 0.95)
        reasoning = f"[parse fallback] raw={raw[:60]}"

    new_H = float(max(H_MIN, min(H_MAX, new_H)))

    print(f"  Agent  -> H={new_H*1e3:.2f}mm  | {reasoning[:90]}")

    trace = state['agent_trace'] + [
        f"Iter {state['iteration']:2d}: H={new_H*1e3:.2f}mm -- {reasoning}"
    ]
    return {'params': {**state['params'], 'H': new_H}, 'agent_trace': trace}


# ── Graph ──────────────────────────────────────────────────────────────────────

def route_after_evaluate(state: OptState) -> str:
    return "end" if state['converged'] else "agent_reason"


def build_graph():
    g = StateGraph(OptState)
    g.add_node("run_simulation", run_simulation_node)
    g.add_node("evaluate",       evaluate_node)
    g.add_node("agent_reason",   agent_reason_node)
    g.set_entry_point("run_simulation")
    g.add_edge("run_simulation", "evaluate")
    g.add_conditional_edges(
        "evaluate", route_after_evaluate,
        {"end": END, "agent_reason": "agent_reason"},
    )
    g.add_edge("agent_reason", "run_simulation")
    return g.compile()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    backend = f"Ollama ({MODEL})" if USE_LOCAL else f"Claude API ({MODEL})"

    print(f"\n{'='*60}")
    print(f"  Phase 4 -- LangGraph AI Optimization Agent")
    print(f"  Backend : {backend}")
    print(f"  L={L*1e3:.0f}mm  P={P:.0f}N  Al-6061")
    print(f"  Constraints: delta<={DELTA_MAX*1e3:.3f}mm  stress<={STRESS_MAX/1e6:.1f}MPa")
    print(f"  RAG store: {store.run_count()} runs  {store.lesson_count()} lessons")
    print(f"  scipy baseline: H={SCIPY_BASELINE['H_mm']:.3f}mm  "
          f"mass={SCIPY_BASELINE['mass']:.4f}kg/m")
    print(f"{'='*60}")

    if USE_LOCAL:
        if not _check_ollama():
            print(f"\n  WARNING: Ollama not reachable or model '{MODEL}' not pulled.")
            print(f"  Start Ollama:  ollama serve")
            print(f"  Pull model:    ollama pull {MODEL}\n")

    initial_state: OptState = {
        'params':      {'L': L, 'H': 0.006, 'load': P},   # start infeasible
        'result':      None,
        'iteration':   0,
        'converged':   False,
        'best_params': None,
        'best_result': None,
        'best_mass':   float('inf'),
        'history':     [],
        'agent_trace': [],
    }

    graph       = build_graph()
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"  Optimization Complete  ({final_state['iteration']} iterations)")
    print(f"{'='*60}")

    if final_state['best_params']:
        bp = final_state['best_params']
        br = final_state['best_result']
        pct = (br['mass_per_depth'] - SCIPY_BASELINE['mass']) / SCIPY_BASELINE['mass'] * 100

        print(f"\n  Best feasible design:")
        print(f"    H          = {bp['H']*1e3:.3f} mm")
        print(f"    Mass/depth = {br['mass_per_depth']:.4f} kg/m")
        print(f"    Deflection = {br['max_deflection']*1e3:.4f} mm  "
              f"(limit {DELTA_MAX*1e3:.3f} mm)")
        print(f"    Stress     = {br['max_von_mises']/1e6:.2f} MPa  "
              f"(limit {STRESS_MAX/1e6:.1f} MPa)")
        print(f"\n  vs scipy.optimize baseline:")
        print(f"    scipy : H={SCIPY_BASELINE['H_mm']:.3f}mm  "
              f"mass={SCIPY_BASELINE['mass']:.4f}kg/m")
        print(f"    agent : H={bp['H']*1e3:.3f}mm  "
              f"mass={br['mass_per_depth']:.4f}kg/m  ({pct:+.1f}% vs scipy)")
    else:
        print("  No feasible design found within iteration limit.")

    print(f"\n  Agent reasoning trace:")
    for line in final_state['agent_trace']:
        safe = line.encode('ascii', errors='replace').decode('ascii')
        print(f"    {safe}")

    print(f"\n  Total runs logged to RAG: {store.run_count()}")
    print(f"{'='*60}\n")
