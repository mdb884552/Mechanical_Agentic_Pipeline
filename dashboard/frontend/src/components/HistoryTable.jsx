import { useState, useEffect, useCallback } from 'react'
import { fetchRuns } from '../api'

export function HistoryTable({ domain }) {
  const [runs, setRuns]       = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(() => {
    setLoading(true)
    fetchRuns(domain, 60)
      .then(d => { setRuns(d.runs ?? []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [domain])

  useEffect(() => { load() }, [load])

  const isBeam = domain === 'beam'

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 14 }}>
        <span className="card-title" style={{ marginBottom: 0 }}>Run History</span>
        <button className="btn-secondary refresh-btn" onClick={load}>↻ Refresh</button>
      </div>

      {loading ? (
        <div style={{ color: 'var(--dim)', fontSize: 13, padding: '16px 0' }}>Loading…</div>
      ) : runs.length === 0 ? (
        <div className="table-empty">No runs logged yet — start an optimization above.</div>
      ) : (
        <div className="history-wrap" style={{ maxHeight: 320, overflowY: 'auto' }}>
          <table className="history-table">
            <thead>
              <tr>
                <th>Time</th>
                {isBeam ? (
                  <>
                    <th>H (mm)</th>
                    <th>Defl (mm)</th>
                    <th>Stress (MPa)</th>
                    <th>Mass (kg/m)</th>
                  </>
                ) : (
                  <>
                    <th>m (mm)</th>
                    <th>b (mm)</th>
                    <th>Bend (MPa)</th>
                    <th>Contact (MPa)</th>
                    <th>Mass (g)</th>
                  </>
                )}
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run, i) => (
                <HistRow key={i} run={run} isBeam={isBeam} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function HistRow({ run, isBeam }) {
  const m  = run.metadata
  const ts = m.timestamp
    ? new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '—'

  return (
    <tr className={m.feasible ? 'row-ok' : 'row-fail'}>
      <td style={{ color: 'var(--dim)' }}>{ts}</td>
      {isBeam ? (
        <>
          <td>{((m.H ?? 0) * 1e3).toFixed(2)}</td>
          <td>{((m.max_deflection ?? 0) * 1e3).toFixed(4)}</td>
          <td>{((m.max_von_mises ?? 0) / 1e6).toFixed(1)}</td>
          <td>{(m.mass_per_depth ?? 0).toFixed(4)}</td>
        </>
      ) : (
        <>
          <td>{(m.m ?? 0).toFixed(2)}</td>
          <td>{(m.b ?? 0).toFixed(1)}</td>
          <td>{(m.bending_stress ?? 0).toFixed(1)}</td>
          <td>{(m.contact_stress ?? 0).toFixed(1)}</td>
          <td>{((m.mass ?? 0) * 1000).toFixed(0)}</td>
        </>
      )}
      <td>
        <span className={`badge ${m.feasible ? 'badge-ok' : 'badge-fail'}`}>
          {m.feasible ? 'OK' : 'FAIL'}
        </span>
      </td>
    </tr>
  )
}
