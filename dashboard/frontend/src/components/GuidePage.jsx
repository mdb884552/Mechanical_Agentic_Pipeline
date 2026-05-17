export function GuidePage() {
  return (
    <div style={{ maxWidth: 820, margin: '0 auto' }}>

      {/* ── Hero ── */}
      <div className="card" style={{ marginBottom: 20, borderColor: 'rgba(59,130,246,0.3)', background: 'rgba(59,130,246,0.04)' }}>
        <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text)', marginBottom: 6 }}>
          How to Use This App
        </div>
        <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7 }}>
          This dashboard is a live demo of an AI-driven structural optimization pipeline.
          An AI agent — powered by Claude and a local RAG memory — iterates toward minimum-mass
          designs subject to real physics constraints. Each tab shows a different angle on why
          the agent adds value beyond a classical formula optimizer.
        </div>
      </div>

      {/* ── Tab guide cards ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

        <GuideCard
          tab="Beam"
          color="#3b82f6"
          icon="📐"
          tagline="Run the agent on a cantilever beam — watch it reason step by step."
          steps={[
            'Set the initial beam height H, length L, and tip load using the sliders.',
            'Click ▶ Run Optimization. The agent starts from your initial H and evaluates the FEA physics.',
            'Watch the Agent Iterations panel: each row is one FEA evaluation. Green = feasible (meets constraints), red = infeasible.',
            'After each evaluation, the reasoning row shows why the agent chose the next design.',
            'The agent converges when two consecutive feasible solutions improve by less than 0.5%.',
            'The FEA Visualization at the bottom renders the deformed beam at the optimal design.',
          ]}
          notes={[
            'scipy baseline: H ≈ 8.308 mm, mass ≈ 2.243 kg/m. The agent should land within 0.5% of this.',
            'Starting from H = 6 mm (infeasible) is intentional — it shows the agent navigating from an invalid design.',
            'Constraints: deflection ≤ 0.05 mm, von Mises stress ≤ 165.6 MPa (60% of Al-6061 yield).',
          ]}
        />

        <GuideCard
          tab="Gear"
          color="#10b981"
          icon="⚙️"
          tagline="Same agent framework, different domain — spur gear pinion optimization."
          steps={[
            'Set initial module m (tooth size), face width b, and torque.',
            'Click ▶ Run Optimization. The agent optimizes two variables simultaneously.',
            'Each iteration shows m, b, bending stress, contact stress, and mass.',
            'The agent tracks which constraint is binding — bending or contact — and adjusts accordingly.',
          ]}
          notes={[
            'The gear uses analytical Lewis bending + Hertz contact equations, not FEA. Same agent, different simulator.',
            'Contact stress is typically the binding constraint (bending is often only ~22% utilized).',
            'After convergence, module is snapped to the nearest standard value (0.5 mm increments).',
          ]}
        />

        <GuideCard
          tab="Race"
          color="#8b5cf6"
          icon="🏁"
          tagline="Head-to-head: scipy SLSQP formula vs. the AI agent. Same problem, same physics."
          steps={[
            'Set shared parameters — H, length, load — using the controls at the top.',
            'Click ▶ Start Race. Both optimizers launch simultaneously.',
            'Watch the left column (scipy): iterations appear as rows of numbers, converging toward the constraint boundary.',
            'Watch the right column (agent): same rows, plus reasoning text after each iteration explaining each decision.',
            'After both converge, read the "What just happened" panel — it summarizes the key differences.',
          ]}
          notes={[
            'Both arrive at essentially the same answer (≈ 8.308 mm). The difference is explainability and generalization.',
            'scipy has no memory — it starts cold every run. The agent queries 26+ past runs from its RAG store.',
            'scipy needs the objective and constraints rewritten for every new problem. The agent framework ran beam and gear unchanged.',
          ]}
        />

        <GuideCard
          tab="Interview"
          color="#f59e0b"
          icon="💬"
          tagline="Describe your design intent in plain language — the agent builds the problem spec for you."
          steps={[
            'The agent asks what type of component you want to optimize (beam or gear).',
            'Answer in plain language — "a 200mm beam" or "a gear under 80 Newton-meters" is enough.',
            'The agent asks one question at a time until it has the required parameters.',
            'When complete, a Parameters Confirmed card shows what was extracted.',
            'Click ▶ Start Optimization — the agent runs with those parameters and streams results live.',
          ]}
          notes={[
            'This demonstrates the greenfield use case: an engineer describes intent, the tool builds the spec.',
            'Optional parameters (deflection limit, material safety factor) use engineering defaults — you don\'t need to specify them.',
            'Click ↺ Reset at any time to start a new interview.',
          ]}
        />

        <GuideCard
          tab="Compare"
          color="#6b7280"
          icon="📊"
          tagline="Best results across all runs stored in the RAG database."
          steps={[
            'This page shows the best feasible design found across all optimization runs in the current session.',
            'Run beam and gear optimizations first — this page populates as results accumulate.',
            'The bar chart compares initial design mass vs. optimized mass for each domain.',
          ]}
          notes={[
            'Results persist in ChromaDB for the lifetime of the server session.',
            'The "vs scipy (H)" metric shows how close the agent landed to the analytical ground truth.',
          ]}
        />

      </div>

      {/* ── Glossary ── */}
      <div className="card" style={{ marginTop: 20 }}>
        <div className="card-title">Glossary</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 24px' }}>
          {[
            ['H (beam height)', 'The single design variable — the agent adjusts this to minimize mass while satisfying constraints.'],
            ['Max deflection', 'Tip displacement under load. Constraint: ≤ 0.05 mm. Deflection ∝ 1/H³ — very sensitive to height.'],
            ['Max von Mises', 'Combined stress measure. Constraint: ≤ 165.6 MPa (60% of Al-6061 yield strength, safety factor 1.67).'],
            ['Mass / Depth', 'Beam mass per unit depth (kg/m). Minimizing this = minimizing material use for a given cross-section.'],
            ['scipy SLSQP', 'Sequential Least Squares Programming — a gradient-based optimizer. Fast and deterministic but produces no explanation.'],
            ['RAG', 'Retrieval-Augmented Generation. The agent queries a ChromaDB vector store of past runs before each decision.'],
            ['Feasible', 'A design that satisfies all constraints. Green badge = OK. Red = at least one constraint violated.'],
            ['Active constraint', 'The constraint that is binding (at its limit). For this beam, deflection is always active at the optimum.'],
            ['Module m (gear)', 'Tooth size in mm. Determines pitch diameter and bending strength. Must snap to standard values.'],
            ['Face width b (gear)', 'Axial length of gear teeth. More face width = more contact area = lower contact stress.'],
            ['Contact stress (Hertz)', 'Surface stress between meshing teeth. Usually the binding constraint — limits how thin the gear can be.'],
            ['LangGraph', 'The graph-based agent orchestration framework. Each node is a defined step: simulate → evaluate → reason → repeat.'],
          ].map(([term, def]) => (
            <div key={term} style={{ padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: 'var(--accent)', marginBottom: 2 }}>{term}</div>
              <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.5 }}>{def}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Why the agent matters ── */}
      <div className="card" style={{ marginTop: 14 }}>
        <div className="card-title">Why the Agent, Not Just the Formula?</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[
            { q: 'scipy already finds the optimum — what does the agent add?',
              a: 'scipy returns a number. The agent returns a number plus an engineering rationale at every step. For a report, a design review, or a hand-off to a colleague, the reasoning trace is as important as the answer.' },
            { q: 'Can\'t I just use a formula for deflection and skip the FEA?',
              a: 'For a uniform rectangular beam, yes. For a bracket with cutouts, a gear with complex tooth geometry, or a casting with variable wall thickness, there is no closed-form formula. The agent architecture scales to arbitrary geometry; a formula does not.' },
            { q: 'Why does the agent use RAG memory?',
              a: 'Before proposing each next design, the agent queries similar past runs — "I\'ve seen infeasible behavior near H=7mm before, here\'s what worked." This is what organizational institutional memory looks like when it\'s made explicit.' },
            { q: 'What does this look like in production?',
              a: 'An engineer opens a SolidWorks model, marks dimensions as variables, applies loads by clicking faces, and hits Optimize. The pipeline runs in the background and writes optimal values back to the model. The Interview tab is a preview of that intent-capture step.' },
          ].map(({ q, a }) => (
            <div key={q} style={{ padding: '12px 14px', background: 'var(--surface)', borderRadius: 8, border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 6 }}>{q}</div>
              <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>{a}</div>
            </div>
          ))}
        </div>
      </div>

    </div>
  )
}

function GuideCard({ tab, color, icon, tagline, steps, notes }) {
  return (
    <div className="card" style={{ borderLeft: `3px solid ${color}` }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{ fontSize: 18 }}>{icon}</span>
        <span style={{ fontWeight: 700, fontSize: 14, color }}>{tab} Tab</span>
        <span style={{ fontSize: 12, color: 'var(--dim)' }}>—</span>
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>{tagline}</span>
      </div>

      <div style={{ marginBottom: 10 }}>
        {steps.map((s, i) => (
          <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 5, fontSize: 12 }}>
            <span style={{ color, fontWeight: 700, minWidth: 16, marginTop: 1 }}>{i + 1}.</span>
            <span style={{ color: 'var(--text)', lineHeight: 1.5 }}>{s}</span>
          </div>
        ))}
      </div>

      {notes.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          {notes.map((n, i) => (
            <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4, fontSize: 11, color: 'var(--dim)', lineHeight: 1.5 }}>
              <span style={{ marginTop: 1 }}>→</span>
              <span>{n}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
