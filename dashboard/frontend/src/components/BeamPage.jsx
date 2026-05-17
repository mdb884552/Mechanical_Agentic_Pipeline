import { useState, useRef } from 'react'
import { streamBeamOpt } from '../api'
import { ConvergenceChart } from './ConvergenceChart'
import { IterationFeed }    from './IterationFeed'
import { HistoryTable }     from './HistoryTable'
import { FEAViz }           from './FEAViz'

const H_MIN = 3, H_MAX = 30  // mm

export function BeamPage() {
  const [hInit, setHInit] = useState(6)
  const [L,     setL]     = useState(100)
  const [load,  setLoad]  = useState(500)
  const [events, setEvents]  = useState([])
  const [running, setRunning] = useState(false)
  const [done,    setDone]    = useState(null)
  const cancelRef = useRef(null)

  const start = () => {
    setEvents([])
    setDone(null)
    setRunning(true)

    cancelRef.current = streamBeamOpt(
      { H_init: hInit / 1e3, L: L / 1e3, load },
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

  const iterEvents = events.filter(e => e.type === 'iteration')
  const baseline   = done

  return (
    <div>
      <div className="main-grid">
        {/* ── Left: controls ── */}
        <div>
          <div className="card">
            <div className="card-title">Parameters</div>

            <div className="field">
              <label>Initial H — {hInit.toFixed(1)} mm</label>
              <input type="range" min={H_MIN} max={H_MAX} step={0.1}
                value={hInit} onChange={e => setHInit(+e.target.value)} />
              <div className="range-labels"><span>{H_MIN} mm</span><span>{H_MAX} mm</span></div>
            </div>

            <div className="field-row">
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

            <div style={{ marginTop: 8, marginBottom: 16, padding: '10px 12px', background: 'var(--card)', borderRadius: 6, fontSize: 12, color: 'var(--dim)', lineHeight: 1.6 }}>
              Material: Al-6061 · Constraint: δ ≤ 0.05 mm, σ ≤ 165.6 MPa<br />
              scipy baseline: H = 8.308 mm · mass = 2.243 kg/m
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
              <ResultRow label="Optimal H"   value={`${done.best_H_mm} mm`} accent />
              <ResultRow label="Mass/Depth"  value={`${done.best_mass} kg/m`} />
              <ResultRow label="Iterations"  value={done.iterations} />
              {done.best_H_mm && (
                <ResultRow
                  label="vs scipy"
                  value={`${((done.best_mass - done.scipy_baseline_mass) / done.scipy_baseline_mass * 100).toFixed(2)}%`}
                  color={done.best_mass <= done.scipy_baseline_mass ? 'var(--success)' : 'var(--warning)'}
                />
              )}
            </div>
          )}
        </div>

        {/* ── Right: chart + feed ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-title">Mass Convergence</div>
            <ConvergenceChart events={events} domain="beam" baseline={done} />
          </div>
          <div className="card">
            <div className="card-title">
              Agent Iterations
              {running && <span className="spinner" style={{ marginLeft: 10 }} />}
            </div>
            <IterationFeed events={events} domain="beam" />
          </div>
        </div>
      </div>

      {/* ── FEA viz ── */}
      <FEAViz
        defaultH={done?.best_H_mm ? done.best_H_mm / 1e3 : 0.008308}
        defaultL={L / 1e3}
        defaultLoad={load}
      />

      {/* ── History ── */}
      <div style={{ marginTop: 20 }}>
        <HistoryTable domain="beam" />
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
