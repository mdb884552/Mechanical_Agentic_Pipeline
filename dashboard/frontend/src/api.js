const BASE = '/api'

export async function fetchStats() {
  const r = await fetch(`${BASE}/stats`)
  return r.json()
}

export async function fetchRuns(domain = 'beam', limit = 50) {
  const r = await fetch(`${BASE}/runs?domain=${domain}&limit=${limit}`)
  return r.json()
}

export async function fetchBest() {
  const r = await fetch(`${BASE}/best`)
  return r.json()
}

export async function fetchFEAViz(L, H, load) {
  const params = new URLSearchParams({ L, H, load })
  const r = await fetch(`${BASE}/fea/viz?${params}`)
  return r.json()
}

/**
 * Run beam optimization and call onEvent(event) for each SSE event.
 * Returns a cancel function.
 */
export function streamBeamOpt(params, onEvent) {
  const ctrl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(`${BASE}/optimize/beam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: ctrl.signal,
      })
      const reader = res.body.getReader()
      const dec    = new TextDecoder()
      let   buf    = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch {}
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') onEvent({ type: 'error', message: String(e) })
    }
  })()
  return () => ctrl.abort()
}

export function streamScipyOpt(params, onEvent) {
  const ctrl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(`${BASE}/optimize/beam/scipy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: ctrl.signal,
      })
      const reader = res.body.getReader()
      const dec    = new TextDecoder()
      let   buf    = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch {}
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') onEvent({ type: 'error', message: String(e) })
    }
  })()
  return () => ctrl.abort()
}

export async function sendInterviewMessage(messages) {
  const r = await fetch(`${BASE}/interview/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  })
  return r.json()
}

export function streamGearOpt(params, onEvent) {
  const ctrl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(`${BASE}/optimize/gear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: ctrl.signal,
      })
      const reader = res.body.getReader()
      const dec    = new TextDecoder()
      let   buf    = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch {}
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') onEvent({ type: 'error', message: String(e) })
    }
  })()
  return () => ctrl.abort()
}
