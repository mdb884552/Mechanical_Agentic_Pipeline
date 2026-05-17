# Aero-Opt: Agentic AI Pipeline for Structural Design Optimization
## Project Summary — Resume & Cover Letter Reference

---

## What This Project Is

An end-to-end autonomous pipeline that ingests a structural design problem, runs physics-based simulation, retrieves relevant past results via RAG, and uses an AI agent orchestrated by LangGraph to drive parametric optimization — closing the loop with a 3D-printed physical part and load test to validate predictions against reality.

This project is a direct analog of business process automation: a domain-specific simulation engine (the "system of record") is wrapped in a structured, observable AI workflow that replaces manual iterative engineering work. The same architecture — tool-calling agent, RAG memory, LangGraph orchestration, local-or-cloud LLM — applies to any domain where an AI needs to reason over structured data, call external processes, and improve decisions over time.

Built entirely in open-source Python. The aerospace domain provides a technically rigorous testbed; the engineering patterns transfer directly to AI/automation roles.

---

## Current Status — Phases 1 through 4 Complete

### Phase 1 — Physics Simulator (COMPLETE)
Built and validated a 2D plane-stress finite element solver from scratch using scikit-fem.

**Test case:** Cantilever beam (Al-6061, L=100mm, H=10mm, P=500N)
**Validation:** FEA tip deflection vs. Euler-Bernoulli analytical solution
**Result:** 0.54% error — within the 2% engineering acceptance threshold

**Technical depth:**
- Derived plane-stress Lamé parameters from the 2D constitutive matrix — using 3D parameters would over-stiffen the model by ~11%, a silent correctness failure that would have caused the optimizer to converge on the wrong answer
- P2 (quadratic) triangular elements with edge-midpoint DOFs, structured mesh with named boundary groups, distributed traction load, Dirichlet clamp via DOF condensation

**Output artifact:** Deformed shape plot + deflection profile vs. analytical curve (0.54% error, indistinguishable by eye)

---

### Phase 2 — Parametric Simulation Callable (COMPLETE)
Wrapped the Phase 1 solver into a clean callable interface:

```python
result = simulate({'L': 0.10, 'H': 0.015, 'load': 500.0})
# returns: max_deflection (m), max_von_mises (Pa), mass_per_depth (kg/m), n_dofs
```

**What this enables:** Any downstream process — optimizer, agent, LangGraph node — calls `simulate()` as a pure function with no human interaction required. This is the adapter pattern between the physics engine and the AI layer.

**Technical depth:**
- Mesh auto-scales with geometry aspect ratio to maintain element quality across the full parameter space
- Von Mises stress: vectorized P1 constant-strain approximation using NumPy einsum — no Python loops in the hot path
- Sanity check reproduces Phase 1 result to 0.56% — regression test built into Phase 2 from day one

**Parameter sweep result:** Deflection follows the theoretical ∝ 1/H³ scaling exactly — confirms the simulator is physics-consistent, not just numerically converged.

**Output artifact:** 2×3 design-space plot (deflection / stress / mass-deflection tradeoff across H and L sweeps)

---

### Phase 3 — Constrained Optimization Loop (COMPLETE)
Wired `simulate()` into `scipy.optimize` (SLSQP) to find minimum-mass designs subject to engineering constraints. Validated the optimization loop with classical methods *before* adding AI — establishing a ground-truth baseline to compare the agent against.

**Problem:** Minimize `mass_per_depth = ρ·L·H` subject to deflection ≤ 0.05mm and von Mises ≤ 165.6MPa (60% yield)

**Results:**
- Initial design: H=10mm, mass=2.700 kg/m
- Optimal design: H=8.308mm, mass=2.243 kg/m — **16.9% mass reduction**
- FEA optimizer vs. beam theory analytical: **0.12% agreement**
- Active constraint: deflection (binding) — stress far from yield, confirming stiffness-dominated problem

**Notable debugging:** Cache key rounding (0.1µm resolution) caused SLSQP's finite-difference gradient to appear as zero — constraint was silently ignored and optimizer hit the lower bound. Fix: exact float as cache key. Demonstrates attention to numerical correctness in AI-adjacent optimization code.

**Output artifact:** 3-panel plot — constraint landscape, infeasible region, convergence history (6 iterations)

---

### Phase 3.5 — Local RAG over Simulation Results & Lessons Learned (COMPLETE)
Built a local ChromaDB vector database that stores every simulation run and the full lessons-learned log, making past experience semantically retrievable by the agent.

**Architecture:**
- **Vector DB:** ChromaDB (persistent local, no server required)
- **Embeddings:** ONNX all-MiniLM-L6-v2 (runs locally, no API key)
- **What gets embedded:** Every `simulate()` call (params + results as natural-language document), 6 engineering lessons (physics insights, debugging notes, API quirks)
- **Query interface:** `store.query_runs("infeasible deflection near 8mm")` returns semantically similar past runs with full metadata

**Store at completion:** 26 simulation runs + 6 lessons indexed and queryable

**Why this matters for AI automation:** Without memory, an agent repeats the same mistakes across runs. RAG over structured results is the pattern used in enterprise AI systems where the agent needs to reason over a growing history of decisions — customer interactions, process runs, incident logs.

---

### Phase 4 — LangGraph Agent Workflow (COMPLETE)
Replaced the deterministic optimizer with a structured LangGraph state machine where Claude reasons over simulation results and RAG-retrieved context to propose design improvements — every decision explicit, logged, and inspectable.

**LangGraph state graph:**
```
START → run_simulation → evaluate → [converged] → END
                                  → agent_reason → run_simulation (loop)
```

**Agent behavior — actual run results:**

| Iter | H | Status | Agent reasoning |
|---|---|---|---|
| 1 | 6.00mm | INFEASIBLE | Computed 1/H³ scaling: needed H×(0.1325/0.050)^(1/3) ≈ 1.38× → proposed 9mm |
| 2 | 9.00mm | OK | Jumped to analytical boundary H=8.3mm |
| 3 | 8.30mm | INFEASIBLE | Just barely over limit — increased to 8.50mm |
| 4–6 | bisecting | OK | Binary search on the constraint boundary |
| **7** | **8.32mm** | **OK** | **Converged — 0.1% from scipy baseline** |

**Final result:** Agent found H=8.32mm (mass=2.246 kg/m) vs. scipy's H=8.308mm (mass=2.243 kg/m) — **0.1% difference in 7 iterations**, starting from an infeasible design.

**LLM options (both implemented):**
- `python agent.py` — Claude API (cloud, highest quality)
- `python agent.py --local --model llama3.2` — Ollama local LLM (on-premise, no data leaves machine)

**Observability features:**
- Full state at every node — params, result, feasibility, best design tracked explicitly
- RAG query logged at each agent step — shows what context the agent retrieved
- Human-readable reasoning trace printed per iteration
- Conditional branching with explicit edge logic — no hidden agent decisions

---

## Remaining Roadmap

### Phase 5 — Physical Validation (UPCOMING)
Close the simulation-to-reality loop with a 3D printer and load test.

1. Export optimized geometry to STL
2. Print in PLA or PETG
3. Load-test: calibrated weights + calipers
4. Compare measured deflection to FEA prediction — quantify the simulation-to-reality gap

Most AI/simulation projects end at a dashboard. Physical validation is what separates engineering from analysis. The quantified delta is a result in itself.

---

## Tech Stack

| Layer | Tool | Role |
|---|---|---|
| FEA solver | scikit-fem 12.x | Physics simulation engine |
| Mesh generation | gmsh 4.15 | Parametric geometry (non-rectangular, Phase 5) |
| Optimization | scipy.optimize SLSQP | Baseline constrained optimizer |
| Agent orchestration | **LangGraph 1.2** | Structured state graph with observability |
| AI reasoning | **Claude API (Sonnet 4.6)** | Cloud LLM — reasoning and JSON proposals |
| Local LLM | **Ollama (llama3.2)** | On-premise option — `--local` flag |
| Agent memory | **ChromaDB + ONNX embeddings** | Local RAG over 26 simulation runs + lessons |
| Numerical | NumPy (vectorized einsum) | No Python loops in hot paths |
| Visualization | Matplotlib (headless Agg) | Saved plot artifacts |
| Hardware | FDM 3D printer + calipers | Physical validation loop (Phase 5) |

---

## Skills Demonstrated

**AI & Automation (directly relevant to this role):**
- **LangGraph** — structured agent orchestration with explicit state, conditional branching, and full observability. Built and ran a working optimization loop. The same pattern applies directly to business process automation workflows.
- **Local RAG** — ChromaDB vector database with ONNX embeddings, seeded with simulation history and engineering lessons. Agent retrieves relevant past context before reasoning each iteration.
- **Dual LLM deployment** — Claude API for cloud inference, Ollama for local/on-premise. Single `--local` flag switches backends. Demonstrates awareness of data-sovereignty constraints in enterprise settings.
- **Tool-calling agent design** — AI agent calls domain-specific tools (`simulate()`, `store.query_runs()`) and reasons over their outputs. This is the foundational pattern behind autonomous AI systems in business processes.
- **Numerical debugging** — identified and fixed a subtle cache-rounding bug that silently zeroed SLSQP's constraint gradient. Demonstrates rigor in AI-adjacent numerical code.

**Software Engineering:**
- Python scientific stack: NumPy, SciPy, scikit-fem, gmsh, ChromaDB, LangGraph, Anthropic SDK
- Vectorized numerical methods (einsum, broadcasting) — performance-conscious implementation
- Callable simulation interfaces — wrapping domain solvers into optimizer/agent-ready black boxes
- Regression testing from Phase 2 onward — sanity check on every run

**Engineering Judgment:**
- Validated the simulator before connecting AI — if physics is wrong, the agent confidently optimizes the wrong objective. Analog: validate your data pipeline before training a model.
- Established scipy baseline before adding LangGraph — agent result (0.1% off scipy) is verifiable, not just a black box output
- Deferred complexity until the simpler path was proven — no premature abstraction

---

## Business Process Mapping Relevance

The core pattern — **wrap a domain process in a callable interface → build RAG memory over its history → orchestrate with LangGraph → swap local or cloud LLM as needed** — is exactly the architecture used to automate business processes with AI:

| This Project | Business Automation Analog |
|---|---|
| `simulate(params)` | ERP/CRM API call, document processing function, pricing engine |
| RAG over simulation runs + lessons | RAG over past tickets, contracts, customer history, incident logs |
| LangGraph state graph | Approval workflow, multi-step business process, exception handling |
| Constraint checking (deflection, stress) | Business rules validation, compliance checks, SLA thresholds |
| `--local` Ollama flag | On-premise deployment for data-sensitive enterprise environments |
| scipy baseline before agent | A/B validation — AI result measured against deterministic ground truth |
| Physical validation (Phase 5) | Pilot rollout, user acceptance testing, production monitoring |

The aerospace domain is the rigorous testbed. The architecture is the transferable skill.

---

## Project Context

- **Timeline:** Summer 2026, solo build
- **Student:** Rising ME senior, Texas A&M University
- **Goal:** Full-stack AI/automation pipeline for internship applications at aerospace and AI/automation companies
- **Open source:** Full code on GitHub (in progress)
- **Physical deliverable:** 3D-printed optimized part + quantified simulation-vs-reality comparison (Phase 5)

---
*Updated 2026-05-16 — Phases 1–4 complete, Phase 5 in progress*
