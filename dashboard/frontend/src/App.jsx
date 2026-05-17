import { useState, useEffect } from 'react'
import { fetchStats } from './api'
import { Header }         from './components/Header'
import { BeamPage }       from './components/BeamPage'
import { GearPage }       from './components/GearPage'
import { ComparePage }    from './components/ComparePage'
import { RacePage }       from './components/RacePage'
import { InterviewPage }  from './components/InterviewPage'
import { GuidePage }      from './components/GuidePage'

export default function App() {
  const [tab,   setTab]   = useState('beam')
  const [stats, setStats] = useState(null)

  useEffect(() => {
    fetchStats().then(setStats).catch(() => {})
    const id = setInterval(() => fetchStats().then(setStats).catch(() => {}), 30_000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="app">
      <Header activeTab={tab} onTabChange={setTab} stats={stats} />

      <main className="page-content">
        {tab === 'beam'      && <BeamPage />}
        {tab === 'gear'      && <GearPage />}
        {tab === 'compare'   && <ComparePage />}
        {tab === 'race'      && <RacePage />}
        {tab === 'interview' && <InterviewPage />}
        {tab === 'guide'     && <GuidePage />}
      </main>
    </div>
  )
}
