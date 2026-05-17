import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'

const TOOLTIP_STYLE = {
  backgroundColor: '#1f2937',
  border: '1px solid #374151',
  borderRadius: 6,
  fontSize: 12,
  color: '#f9fafb',
}

export function ConvergenceChart({ events, domain = 'beam', baseline }) {
  const iterData = events.filter(e => e.type === 'iteration')

  if (iterData.length === 0) {
    return (
      <div className="chart-empty">
        Mass convergence will appear here
      </div>
    )
  }

  const data = iterData.map(e => ({
    iter:  e.iter,
    mass:  domain === 'beam' ? e.mass       : +(e.mass_g / 1000).toFixed(4),
    feasible: e.feasible,
  }))

  const yLabel = domain === 'beam' ? 'Mass (kg/m)' : 'Mass (kg)'
  const baselineVal = domain === 'beam'
    ? baseline?.scipy_baseline_mass
    : baseline ? baseline.scipy_baseline_g / 1000 : undefined

  const CustomDot = (props) => {
    const { cx, cy, payload } = props
    return (
      <circle
        cx={cx} cy={cy} r={4}
        fill={payload.feasible ? '#10b981' : '#ef4444'}
        stroke="none"
      />
    )
  }

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="iter"
            stroke="#6b7280"
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            label={{ value: 'Iteration', position: 'insideBottom', offset: -2, fontSize: 11, fill: '#6b7280' }}
          />
          <YAxis
            stroke="#6b7280"
            tick={{ fontSize: 11, fill: '#9ca3af' }}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 12, fontSize: 11, fill: '#6b7280' }}
            width={64}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={v => `Iteration ${v}`}
            formatter={(val, name) => [val.toFixed(4), name === 'mass' ? yLabel : name]}
          />
          {baselineVal && (
            <ReferenceLine
              y={baselineVal}
              stroke="#6b7280"
              strokeDasharray="5 3"
              label={{ value: 'scipy', position: 'right', fontSize: 10, fill: '#6b7280' }}
            />
          )}
          <Line
            type="monotone"
            dataKey="mass"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={<CustomDot />}
            activeDot={{ r: 6 }}
            name="mass"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
