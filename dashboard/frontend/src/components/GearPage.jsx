import { useState, useRef } from 'react'
import { streamGearOpt } from '../api'
import { ConvergenceChart } from './ConvergenceChart'
import { IterationFeed }    from './IterationFeed'
import { HistoryTable }     from './HistoryTable'

export function GearPage() {
  const [mInit,  setMInit]  = useState(2.0)
  const [bInit,  setBInit]  = useState(25.0)
  const [torque, setTorque] = useState(50)
  const [events,  setEvents]  = useState([])
  const [running, setRunning] = useState(false)
  const [done,    setDone]    = useState(null)
  const cancelRef = useRef(null)

  const start = () => {
    setEvents([])
    setDone(null)
    setRunning(true)

    cancelRef.current = streamGearOpt(
      { m_init: mInit, b_init: bInit, torque },
      (ev) => {
        setEvents(prev => [...prev, ev])
        if (ev.type === 'done' || ev.type === 'error') {
          setRunning(false)
          if (ev.type === 'done') setDone(ev)
        }
      },
    )
  }

  const stop = () => {
    cancelRef.current?.()
    setRunning(false)
  }

  return (
    <div>
      <div className="main-grid">
        {/* ── Left: controls ── */}
        <div>
          <div className="card">
            <div className="card-title">Parameters</div>

            <div className="field">
              <label>Initial module m — {mInit.toFixed(2)} mm</label>
              <input type="range" min={1.5} max={6} step={0.1}
                value={mInit} onChange={e => setMInit(+e.target.value)} />
              <div className="range-labels"><span>1.5 mm</span><span>6 mm</span></div>
            </div>

            <div className="field">
              <label>Initial face width b — {bInit.toFixed(1)} mm</label>
              <input type="range" min={10} max={80} step={1}
                value={bInit} onChange={e => setBInit(+e.target.value)} />
              <div className="range-labels"><span>10 mm</span><span>80 mm</span></div>
            </div>

            <div className="field">
              <label>Input Torque (N·m)</label>
              <input type="number" min="10" max="500" step="5"
                value={torque} onChange={e => setTorque(+e.target.value)} />
            </div>

            <div style={{ marginTop: 8, marginBottom: 16, padding: '10px 12px', background: 'var(--card)', borderRadius: 6, fontSize: 12, color: 'var(--dim)', lineHeight: 1.6 }}>
              N=15 teeth · 3:1 ratio · AISI 4140 (250 HB)<br />
              Constraints: σ_b ≤ 180 MPa, σ_c ≤ 550 MPa<br />
              scipy baseline: m=4.30 mm, b=30.1 mm → 845 g
            </div>

            <button
              className={`btn-primary ${running ? 'running' : ''}`}
              onClick={running ? stop : start}
            >
              {running ? '⏹ Stop' : '▶ Run Optimization'}
            </button>
          </div>

          {done && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-title">Result</div>
              <ResultRow label="Optimal m"   value={`${done.best_m_mm} mm`} accent />
              <ResultRow label="Optimal b"   value={`${done.best_b_mm} mm`} accent />
              <ResultRow label="Mass"        value={`${done.best_mass_g} g`} />
              <ResultRow label="Iterations"  value={done.iterations} />
              {done.best_mass_g && (
                <ResultRow
                  label="vs scipy"
                  value={`${((done.best_mass_g - done.scipy_baseline_g) / done.scipy_baseline_g * 100).toFixed(1)}%`}
                  color={done.best_mass_g <= done.scipy_baseline_g ? 'var(--success)' : 'var(--warning)'}
                />
              )}
            </div>
          )}
        </div>

        {/* ── Right: chart + feed ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-title">Mass Convergence</div>
            <ConvergenceChart events={events} domain="gear" baseline={done} />
          </div>
          <div className="card">
            <div className="card-title">
              Agent Iterations
              {running && <span className="spinner" style={{ marginLeft: 10 }} />}
            </div>
            <IterationFeed events={events} domain="gear" />
          </div>
        </div>
      </div>

      <div style={{ marginTop: 20 }}>
        <HistoryTable domain="gear" />
      </div>
    </div>
  )
}

function ResultRow({ label, value, accent, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
      <span style={{ color: 'var(--muted)' }}>{label}</span>
      <span style={{ fontWeight: 600, fontFamily: 'monospace', color: color ?? (accent ? 'var(--accent)' : 'var(--text)') }}>{value}</span>
    </div>
  )
}
