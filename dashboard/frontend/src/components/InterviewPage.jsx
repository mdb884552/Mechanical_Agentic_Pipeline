import { useState, useRef, useEffect } from 'react'
import { sendInterviewMessage, streamBeamOpt, streamGearOpt } from '../api'
import { ConvergenceChart } from './ConvergenceChart'
import { IterationFeed }    from './IterationFeed'

const AGENT_INTRO = "I'll help you set up a structural optimization. What type of component do you want to optimize — a cantilever beam or a spur gear?"

export function InterviewPage() {
  const [messages,   setMessages]   = useState([{ role: 'assistant', content: AGENT_INTRO }])
  const [input,      setInput]      = useState('')
  const [loading,    setLoading]    = useState(false)
  const [spec,       setSpec]       = useState(null)   // {domain, params, summary} when interview done
  const [optEvents,  setOptEvents]  = useState([])
  const [optRunning, setOptRunning] = useState(false)
  const [optDone,    setOptDone]    = useState(null)
  const cancelRef  = useRef(null)
  const bottomRef  = useRef(null)
  const inputRef   = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, optEvents])

  const send = async () => {
    const text = input.trim()
    if (!text || loading || spec) return

    const userMsg = { role: 'user', content: text }
    const history = [...messages, userMsg]
    setMessages(history)
    setInput('')
    setLoading(true)

    // Build the API messages list (exclude the canned intro from history sent to API —
    // the system prompt provides context so the first real exchange starts with the user)
    const apiMessages = history
      .filter((m, i) => !(i === 0 && m.role === 'assistant'))  // drop canned intro
      .map(m => ({ role: m.role, content: m.content }))

    try {
      const reply = await sendInterviewMessage(apiMessages)

      const assistantMsg = { role: 'assistant', content: reply.content }
      setMessages(prev => [...prev, assistantMsg])

      if (reply.done) {
        setSpec({ domain: reply.domain, params: reply.params, summary: reply.summary })
      }
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }])
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  const startOpt = () => {
    if (!spec || optRunning) return
    setOptEvents([])
    setOptDone(null)
    setOptRunning(true)

    const streamFn = spec.domain === 'beam' ? streamBeamOpt : streamGearOpt
    cancelRef.current = streamFn(spec.params, ev => {
      setOptEvents(prev => [...prev, ev])
      if (ev.type === 'done' || ev.type === 'error') {
        setOptDone(ev)
        setOptRunning(false)
      }
    })
  }

  const reset = () => {
    cancelRef.current?.()
    setMessages([{ role: 'assistant', content: AGENT_INTRO }])
    setInput('')
    setSpec(null)
    setOptEvents([])
    setOptDone(null)
    setOptRunning(false)
  }

  return (
    <div>
      {/* ── Intro banner ── */}
      <div className="card" style={{ marginBottom: 20, borderColor: 'rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.05)' }}>
        <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.7 }}>
          <strong style={{ color: 'var(--text)' }}>Greenfield optimization — no sliders required.</strong>
          {' '}Describe your design intent in plain language. The agent asks clarifying questions,
          extracts the problem specification, then runs the optimizer. This is what Phase 8 looks like in production.
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: spec ? '420px 1fr' : '1fr', gap: 16, alignItems: 'start' }}>
        {/* ── Chat panel ── */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          <div className="card-title" style={{ marginBottom: 0 }}>
            Design Intent Interview
            {spec && (
              <button
                onClick={reset}
                style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--dim)', background: 'none', border: 'none', cursor: 'pointer', padding: '2px 6px' }}
              >
                ↺ Reset
              </button>
            )}
          </div>

          {/* Message list */}
          <div style={{ maxHeight: 420, overflowY: 'auto', padding: '12px 0', display: 'flex', flexDirection: 'column', gap: 10 }}>
            {messages.map((m, i) => (
              <ChatBubble key={i} role={m.role} content={m.content} />
            ))}
            {loading && <ChatBubble role="assistant" content="…" typing />}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          {!spec && (
            <div style={{ display: 'flex', gap: 8, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                placeholder="Type your answer…"
                disabled={loading}
                style={{
                  flex: 1, background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: 6, padding: '8px 12px', fontSize: 13, color: 'var(--text)',
                  outline: 'none',
                }}
              />
              <button
                className="btn-primary"
                onClick={send}
                disabled={loading || !input.trim()}
                style={{ padding: '8px 16px', minWidth: 0 }}
              >
                Send
              </button>
            </div>
          )}

          {/* Spec confirmed */}
          {spec && (
            <div style={{ paddingTop: 12, borderTop: '1px solid var(--border)' }}>
              <SpecCard spec={spec} />
              {!optRunning && !optDone && (
                <button className="btn-primary" style={{ marginTop: 12, width: '100%' }} onClick={startOpt}>
                  ▶ Start Optimization
                </button>
              )}
              {optRunning && (
                <div style={{ marginTop: 12, textAlign: 'center', fontSize: 12, color: 'var(--dim)' }}>
                  <span className="spinner" style={{ marginRight: 6 }} />Optimizing…
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Optimization panel (shown once spec is ready) ── */}
        {spec && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card">
              <div className="card-title">
                Mass Convergence
                {optRunning && <span className="spinner" style={{ marginLeft: 10 }} />}
              </div>
              <ConvergenceChart events={optEvents} domain={spec.domain} baseline={optDone} />
            </div>

            {optDone?.type === 'done' && (
              <div style={{ padding: '12px 16px', background: 'rgba(16,185,129,0.08)', borderRadius: 8, border: '1px solid rgba(16,185,129,0.3)', display: 'flex', gap: 24, fontSize: 13 }}>
                <ResultStat label="Optimal H" value={`${optDone.best_H_mm} mm`} color="var(--success)" />
                <ResultStat label="Mass / Depth" value={`${optDone.best_mass} kg/m`} />
                <ResultStat label="Iterations" value={optDone.iterations} />
                <ResultStat label="vs scipy" value={`${((optDone.best_mass - optDone.scipy_baseline_mass) / optDone.scipy_baseline_mass * 100).toFixed(2)}%`} />
              </div>
            )}

            <div className="card">
              <div className="card-title">Agent Iterations</div>
              <IterationFeed events={optEvents} domain={spec.domain} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function ChatBubble({ role, content, typing }) {
  const isUser = role === 'user'
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', padding: '0 4px' }}>
      <div style={{
        maxWidth: '85%',
        padding: '8px 12px',
        borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
        background: isUser ? 'rgba(59,130,246,0.15)' : 'var(--surface)',
        border: isUser ? '1px solid rgba(59,130,246,0.3)' : '1px solid var(--border)',
        fontSize: 13,
        color: typing ? 'var(--dim)' : 'var(--text)',
        lineHeight: 1.5,
        fontStyle: typing ? 'italic' : 'normal',
      }}>
        {content}
      </div>
    </div>
  )
}

function SpecCard({ spec }) {
  const labels = spec.domain === 'beam'
    ? [
        ['Domain',  'Cantilever Beam'],
        ['Length',  `${(spec.params.L * 1e3).toFixed(0)} mm`],
        ['Load',    `${spec.params.load} N`],
        ['Start H', `${(spec.params.H_init * 1e3).toFixed(0)} mm`],
      ]
    : [
        ['Domain',   'Spur Gear'],
        ['Torque',   `${spec.params.torque} N·m`],
        ['Module₀',  `${(spec.params.m_init * 1e3).toFixed(1)} mm`],
        ['FaceW₀',   `${(spec.params.b_init * 1e3).toFixed(0)} mm`],
      ]

  return (
    <div style={{ padding: '10px 12px', background: 'rgba(16,185,129,0.07)', borderRadius: 6, border: '1px solid rgba(16,185,129,0.3)' }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--success)', marginBottom: 8, letterSpacing: '0.05em' }}>PARAMETERS CONFIRMED</div>
      {labels.map(([label, value]) => (
        <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
          <span style={{ color: 'var(--dim)' }}>{label}</span>
          <span style={{ fontFamily: 'monospace', color: 'var(--text)' }}>{value}</span>
        </div>
      ))}
    </div>
  )
}

function ResultStat({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--dim)' }}>{label}</div>
      <div style={{ fontWeight: 700, fontFamily: 'monospace', color: color ?? 'var(--text)' }}>{value}</div>
    </div>
  )
}
