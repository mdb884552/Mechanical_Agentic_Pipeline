import { useState, useRef } from 'react'
import { streamBeamOpt, streamScipyOpt } from '../api'
import { ConvergenceChart } from './ConvergenceChart'
import { IterationFeed }    from './IterationFeed'

const H_MIN = 3, H_MAX = 30

export function RacePage() {
  const [hInit, setHInit] = useState(10)
  const [L,     setL]     = useState(100)
  const [load,  setLoad]  = useState(500)

  const [scipyEvents, setScipyEvents] = useState([])
  const [agentEvents, setAgentEvents] = useState([])
  const [running,     setRunning]     = useState(false)
  const [scipyDone,   setScipyDone]   = useState(null)
  const [agentDone,   setAgentDone]   = useState(null)

  const cancelRefs = useRef([])

  const start = () => {
    setScipyEvents([])
    setAgentEvents([])
    setScipyDone(null)
    setAgentDone(null)
    setRunning(true)

    const params = { H_init: hInit / 1e3, L: L / 1e3, load }

    const cancelScipy = streamScipyOpt(params, ev => {
      setScipyEvents(prev => [...prev, ev])
      if (ev.type === 'done' || ev.type === 'error') setScipyDone(ev)
    })

    const cancelAgent = streamBeamOpt(params, ev => {
      setAgentEvents(prev => [...prev, ev])
      if (ev.type === 'done' || ev.type === 'error') {
        setAgentDone(ev)
        setRunning(false)
      }
    })

    cancelRefs.current = [cancelScipy, cancelAgent]
  }

  const stop = () => {
    cancelRefs.current.forEach(fn => fn?.())
    setRunning(false)
  }

  const bothDone = scipyDone && agentDone

  return (
    <div>
      {/* ── Intro ── */}
      <div className="card" style={{ marginBottom: 20, borderColor: 'rgba(59,130,246,0.3)', background: 'rgba(59,130,246,0.05)' }}>
        <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7 }}>
          <strong style={{ color: 'var(--text)' }}>Same problem. Same physics. Watch what's different.</strong>
          {' '}scipy SLSQP and the AI agent both minimize beam mass subject to the same constraints.
          scipy returns a number. The agent returns a number <em>and explains why it made each move</em>.
        </div>
      </div>

      {/* ── Controls ── */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">Shared Parameters</div>
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: 16, alignItems: 'end' }}>
          <div className="field">
            <label>Initial H — {hInit.toFixed(1)} mm</label>
            <input type="range" min={H_MIN} max={H_MAX} step={0.5}
              value={hInit} onChange={e => setHInit(+e.target.value)} />
            <div className="range-labels"><span>{H_MIN} mm</span><span>{H_MAX} mm</span></div>
          </div>
          <div className="field">
            <label>Length L (mm)</label>
            <input type="number" min="50" max="500" step="10"
              value={L} onChange={e => setL(+e.target.value)} />
          </div>
          <div className="field">
            <label>Tip Load (N)</label>
            <input type="number" min="10" max="10000" step="10"
              value={load} onChange={e => setLoad(+e.target.value)} />
          </div>
        </div>
        <div style={{ marginTop: 12, fontSize: 11, color: 'var(--dim)' }}>
          Material: Al-6061 · Constraints: δ ≤ 0.05 mm, σ ≤ 165.6 MPa · Analytical optimum: H ≈ 8.308 mm
        </div>
        <button
          className={`btn-primary ${running ? 'running' : ''}`}
          style={{ marginTop: 16, width: '100%' }}
          onClick={running ? stop : start}
        >
          {running ? '⏹ Stop' : '▶ Start Race'}
        </button>
      </div>

      {/* ── Side-by-side columns ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <RaceColumn
          label="scipy SLSQP"
          tag="Formula Optimizer"
          tagColor="#6b7280"
          accentColor="#8b5cf6"
          events={scipyEvents}
          done={scipyDone}
          domain="beam"
          description="Gradient descent: evaluates the objective function and constraints at each step, uses finite differences to approximate the gradient, follows steepest descent to the constraint boundary."
          noReasoning
        />
        <RaceColumn
          label="AI Agent"
          tag="Reasoning Optimizer"
          tagColor="#3b82f6"
          accentColor="#3b82f6"
          events={agentEvents}
          done={agentDone}
          domain="beam"
          description="Iterative reasoning: evaluates physics, queries memory of past runs, then articulates why it proposes each next design before running the next simulation."
          running={running}
        />
      </div>

      {/* ── Post-race insight panel ── */}
      {bothDone && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-title">What just happened</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <InsightCard
              color="#8b5cf6"
              title="scipy SLSQP"
              rows={[
                ['Approach', 'Gradient-based, deterministic'],
                ['Memory', 'None — starts cold every run'],
                ['Explanation', 'No — returns a number only'],
                ['Domain transfer', 'Requires rewriting objective & constraints'],
                ['Result', scipyDone?.best_H_mm ? `H = ${scipyDone.best_H_mm} mm · ${scipyDone.best_mass} kg/m` : '—'],
              ]}
            />
            <InsightCard
              color="#3b82f6"
              title="AI Agent"
              rows={[
                ['Approach', 'Iterative reasoning with LangGraph'],
                ['Memory', 'RAG store — queries past runs each iteration'],
                ['Explanation', 'Yes — engineering rationale at every step'],
                ['Domain transfer', 'Same framework ran beam and gear unchanged'],
                ['Result', agentDone?.best_H_mm ? `H = ${agentDone.best_H_mm} mm · ${agentDone.best_mass} kg/m` : '—'],
              ]}
            />
          </div>
          <div style={{ marginTop: 14, padding: '10px 14px', background: 'rgba(59,130,246,0.07)', borderRadius: 6, fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>
            Both arrive at essentially the same answer (analytical optimum: H ≈ 8.308 mm).
            The difference is what the agent adds <em>beyond</em> the number: explainability, memory, and
            the ability to generalize to any domain without rewriting the optimizer.
          </div>
        </div>
      )}
    </div>
  )
}

function RaceColumn({ label, tag, tagColor, accentColor, events, done, domain, description, noReasoning, running }) {
  const iterCount = events.filter(e => e.type === 'iteration').length
  const isError   = done?.type === 'error'
  const isDone    = done?.type === 'done'

  // Filter out reasoning events for the scipy column so the contrast is visible
  const visibleEvents = noReasoning
    ? events.filter(e => e.type !== 'reasoning')
    : events

  return (
    <div>
      {/* Column header */}
      <div style={{ marginBottom: 12, padding: '12px 16px', background: 'var(--card)', borderRadius: 8, border: `1px solid var(--border)`, borderTop: `3px solid ${accentColor}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <span style={{ fontWeight: 700, fontSize: 15, color: accentColor }}>{label}</span>
          <span style={{ fontSize: 11, color: tagColor, background: 'var(--surface)', padding: '2px 8px', borderRadius: 12, border: '1px solid var(--border)' }}>{tag}</span>
          {running && !isDone && !isError && (
            <span className="spinner" style={{ marginLeft: 'auto' }} />
          )}
          {isDone && <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--success)' }}>✓ Done ({done.iterations} iters)</span>}
        </div>
        <div style={{ fontSize: 11, color: 'var(--dim)', lineHeight: 1.5 }}>{description}</div>
      </div>

      {/* Convergence chart */}
      <div className="card" style={{ marginBottom: 12 }}>
        <div className="card-title">Mass Convergence</div>
        <ConvergenceChart events={events} domain={domain} baseline={done} />
      </div>

      {/* Result badge */}
      {isDone && (
        <div style={{ marginBottom: 12, padding: '10px 14px', background: `rgba(${accentColor === '#3b82f6' ? '59,130,246' : '139,92,246'},0.08)`, borderRadius: 6, border: `1px solid ${accentColor}33`, display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
          <span style={{ color: 'var(--muted)' }}>Optimal H</span>
          <span style={{ fontWeight: 700, fontFamily: 'monospace', color: accentColor }}>{done.best_H_mm} mm</span>
          <span style={{ color: 'var(--muted)' }}>Mass</span>
          <span style={{ fontWeight: 700, fontFamily: 'monospace', color: accentColor }}>{done.best_mass} kg/m</span>
        </div>
      )}

      {/* Iteration feed */}
      <div className="card">
        <div className="card-title">
          Iterations
          {noReasoning && (
            <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--dim)', fontWeight: 400 }}>
              (no reasoning — formula only)
            </span>
          )}
        </div>
        <IterationFeed events={visibleEvents} domain={domain} />
      </div>
    </div>
  )
}

function InsightCard({ color, title, rows }) {
  return (
    <div style={{ padding: '12px 14px', background: 'var(--surface)', borderRadius: 8, border: `1px solid ${color}44` }}>
      <div style={{ fontWeight: 600, fontSize: 13, color, marginBottom: 10 }}>{title}</div>
      {rows.map(([label, value]) => (
        <div key={label} style={{ display: 'flex', gap: 8, marginBottom: 6, fontSize: 12 }}>
          <span style={{ color: 'var(--dim)', minWidth: 130 }}>{label}</span>
          <span style={{ color: 'var(--muted)' }}>{value}</span>
        </div>
      ))}
    </div>
  )
}
