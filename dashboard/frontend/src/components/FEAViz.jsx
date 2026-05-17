import { useState } from 'react'
import { fetchFEAViz } from '../api'

const DELTA_LIMIT  = 0.05   // mm
const STRESS_LIMIT = 165.6  // MPa

export function FEAViz({ defaultH = 0.008308, defaultL = 0.1, defaultLoad = 500 }) {
  const [L,    setL]    = useState(defaultL)
  const [H,    setH]    = useState(defaultH)
  const [load, setLoad] = useState(defaultLoad)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const d = await fetchFEAViz(L, H, load)
      setData(d)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const deflStatus = data
    ? data.max_deflection_mm <= DELTA_LIMIT ? 'ok' : 'fail'
    : null
  const stressStatus = data
    ? data.max_von_mises_mpa <= STRESS_LIMIT ? 'ok' : 'fail'
    : null

  return (
    <div className="card fea-section">
      <div className="card-title">FEA Stress Visualization</div>

      <div className="fea-controls">
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', flex: 1 }}>
          <label style={{ fontSize: 12, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
            H (mm):
            <input
              type="number" min="3" max="30" step="0.1"
              value={+(H * 1e3).toFixed(2)}
              onChange={e => setH(+e.target.value / 1e3)}
              style={{ width: 72, padding: '4px 8px', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', fontSize: 13 }}
            />
          </label>
          <label style={{ fontSize: 12, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
            L (mm):
            <input
              type="number" min="50" max="300" step="10"
              value={+(L * 1e3).toFixed(0)}
              onChange={e => setL(+e.target.value / 1e3)}
              style={{ width: 72, padding: '4px 8px', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', fontSize: 13 }}
            />
          </label>
          <label style={{ fontSize: 12, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
            Load (N):
            <input
              type="number" min="10" max="5000" step="10"
              value={load}
              onChange={e => setLoad(+e.target.value)}
              style={{ width: 80, padding: '4px 8px', background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)', fontSize: 13 }}
            />
          </label>
        </div>
        <button
          className="btn-primary"
          style={{ width: 'auto', padding: '8px 20px' }}
          onClick={run}
          disabled={loading}
        >
          {loading ? <span className="spinner" /> : 'Run FEA'}
        </button>
      </div>

      {error && (
        <div style={{ color: 'var(--danger)', fontSize: 12, marginBottom: 12 }}>Error: {error}</div>
      )}

      {data && (
        <div className="fea-metric-row">
          <div className="fea-metric">
            <div className="fea-metric-label">Max Deflection</div>
            <div className={`fea-metric-value ${deflStatus}`}>
              {data.max_deflection_mm.toFixed(4)} mm
            </div>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginTop: 2 }}>limit: {DELTA_LIMIT} mm</div>
          </div>
          <div className="fea-metric">
            <div className="fea-metric-label">Max von Mises</div>
            <div className={`fea-metric-value ${stressStatus}`}>
              {data.max_von_mises_mpa.toFixed(1)} MPa
            </div>
            <div style={{ fontSize: 11, color: 'var(--dim)', marginTop: 2 }}>limit: {STRESS_LIMIT} MPa</div>
          </div>
          <div className="fea-metric">
            <div className="fea-metric-label">Mass/Depth</div>
            <div className="fea-metric-value" style={{ color: 'var(--accent)' }}>
              {data.mass_per_depth.toFixed(4)} kg/m
            </div>
          </div>
        </div>
      )}

      {loading && <div className="fea-loading"><span className="spinner" style={{ marginRight: 10 }} />Running FEA…</div>}

      {data?.plot_b64 && !loading && (
        <img
          src={`data:image/png;base64,${data.plot_b64}`}
          alt="FEA stress and deflection fields"
          className="fea-img"
        />
      )}

      {!data && !loading && (
        <div className="fea-placeholder">
          <span style={{ fontSize: 32 }}>📐</span>
          <span>Set parameters and click Run FEA</span>
        </div>
      )}
    </div>
  )
}
