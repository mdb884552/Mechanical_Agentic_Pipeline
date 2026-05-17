import { useEffect, useRef } from 'react'

export function IterationFeed({ events, domain = 'beam' }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  const iterEvents = events.filter(e => e.type === 'iteration' || e.type === 'reasoning' || e.type === 'done')

  if (iterEvents.length === 0) {
    return <div className="feed-empty">Iterations will stream here during optimization.</div>
  }

  return (
    <div>
      <ColHeader domain={domain} />
      <div className="iter-feed">
        {iterEvents.map((ev, i) => <FeedRow key={i} event={ev} domain={domain} />)}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function ColHeader({ domain }) {
  const cls = `feed-col-header ${domain}-row`
  if (domain === 'beam') {
    return (
      <div className={cls}>
        <span>#</span><span>H (mm)</span><span>Defl (mm)</span>
        <span>Stress (MPa)</span><span>Mass (kg/m)</span><span>Status</span>
      </div>
    )
  }
  return (
    <div className={cls}>
      <span>#</span><span>m (mm)</span><span>b (mm)</span>
      <span>Bend (MPa)</span><span>Contact (MPa)</span><span>Mass (g)</span><span>Status</span>
    </div>
  )
}

function FeedRow({ event, domain }) {
  if (event.type === 'iteration') {
    const ok = event.feasible
    return (
      <div className={`iter-row ${domain}-row ${ok ? 'feasible' : 'infeasible'}`}>
        <span className="iter-num">{event.iter}</span>
        {domain === 'beam' ? (
          <>
            <span>{event.H_mm}</span>
            <span>{event.deflection_mm}</span>
            <span>{event.stress_mpa}</span>
            <span>{event.mass}</span>
          </>
        ) : (
          <>
            <span>{event.m_mm}</span>
            <span>{event.b_mm}</span>
            <span>{event.bending_mpa}</span>
            <span>{event.contact_mpa}</span>
            <span>{event.mass_g}</span>
          </>
        )}
        <span className={`badge ${ok ? 'badge-ok' : 'badge-fail'}`}>
          {ok ? 'OK' : 'INFEASIBLE'}
        </span>
      </div>
    )
  }

  if (event.type === 'reasoning') {
    const next = domain === 'beam'
      ? `→ H = ${event.next_H_mm} mm`
      : `→ m=${event.next_m_mm} mm, b=${event.next_b_mm} mm`
    return (
      <div className="reasoning-row">
        <span className="reasoning-next">{next}</span>
        <span className="reasoning-text">{event.reasoning}</span>
      </div>
    )
  }

  if (event.type === 'done') {
    const summary = domain === 'beam'
      ? `H = ${event.best_H_mm} mm  ·  ${event.best_mass} kg/m`
      : `m = ${event.best_m_mm} mm, b = ${event.best_b_mm} mm  ·  ${event.best_mass_g} g`
    return (
      <div className="done-row">
        <span>✓ Converged</span>
        <span>{event.iterations} iterations</span>
        <span className="done-sub">{summary}</span>
      </div>
    )
  }

  if (event.type === 'error') {
    return (
      <div className="done-row" style={{ background: 'rgba(239,68,68,0.08)', borderColor: 'rgba(239,68,68,0.3)', color: 'var(--danger)' }}>
        Error: {event.message}
      </div>
    )
  }

  return null
}
