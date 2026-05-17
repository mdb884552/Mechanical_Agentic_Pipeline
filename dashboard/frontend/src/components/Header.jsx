export function Header({ activeTab, onTabChange, stats }) {
  return (
    <header className="header">
      <div className="header-brand">
        <span className="brand-name">AERO-OPT</span>
        <span className="brand-sub">AI Structural Optimizer</span>
      </div>

      <nav className="header-nav">
        {['beam', 'gear', 'compare'].map(tab => (
          <button
            key={tab}
            className={`nav-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => onTabChange(tab)}
          >
            {tab === 'beam'    ? 'Beam'         :
             tab === 'gear'    ? 'Gear'         :
                                 'Compare'}
          </button>
        ))}
      </nav>

      {stats && (
        <div style={{ display: 'flex', gap: 20, marginLeft: 20, fontSize: 12, color: 'var(--dim)' }}>
          <span>Beam: <b style={{ color: 'var(--accent)' }}>{stats.beam?.run_count ?? '—'}</b> runs</span>
          <span>Gear: <b style={{ color: 'var(--accent)' }}>{stats.gear?.run_count ?? '—'}</b> runs</span>
        </div>
      )}

      <div className="header-links">
        <a
          className="header-link"
          href="https://github.com/MichaelBarfoot/aero-opt"
          target="_blank"
          rel="noreferrer"
        >
          GitHub
        </a>
      </div>
    </header>
  )
}
