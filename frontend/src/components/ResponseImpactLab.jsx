import { useEffect, useMemo, useState } from 'react'

const API = 'http://127.0.0.1:8000'
const label = (value) => String(value || '').replaceAll('_', ' ').replace('ZONE ', 'Zone ').replace('RESPONSE', '').trim()
const metricRows = [
  ['instability', 'Instability', (value) => Math.round(value)],
  ['compression', 'Compression', (value) => `${Math.round(value * 100)}%`],
  ['counter_flow', 'Counter-flow', (value) => `${Math.round(value * 100)}%`],
  ['stop_go', 'Stop-Go', (value) => `${Math.round(value * 100)}%`],
  ['motion_disorder', 'Motion Disorder', (value) => `${Math.round(value * 100)}%`],
]

function Delta({ value, suffix = '' }) {
  const direction = value < 0 ? 'improved' : value > 0 ? 'worsened' : 'same'
  return <small className={`impact-delta ${direction}`}>Δ {value > 0 ? '+' : ''}{Math.round(value * 10) / 10}{suffix} {direction === 'worsened' ? 'WORSENED' : direction === 'improved' ? 'IMPROVED' : 'NO CHANGE'}</small>
}

export default function ResponseImpactLab({ plan }) {
  const scenarios = plan.simulations || []; const [scenario, setScenario] = useState(scenarios.find((item) => item.scenario === plan.recommended?.scenario)?.scenario || scenarios[0]?.scenario || 'NO_ACTION'); const [impact, setImpact] = useState(null); const [allImpacts, setAllImpacts] = useState([]); const [why, setWhy] = useState(false)
  useEffect(() => { let active = true; Promise.all(scenarios.map((item) => fetch(`${API}/events/${plan.event_id}/response-commander/plans/${plan.id}/impact?scenario=${item.scenario}`).then((response) => response.json()))).then((data) => { if (active) { setAllImpacts(data); setImpact(data.find((item) => item.scenario === scenario) || data[0]) } }).catch(() => {}); return () => { active = false } }, [plan])
  useEffect(() => { const selected = allImpacts.find((item) => item.scenario === scenario); if (selected) setImpact(selected) }, [scenario, allImpacts])
  const best = useMemo(() => allImpacts.reduce((winner, item) => !winner || item.impact_score > winner.impact_score ? item : winner, null)?.scenario, [allImpacts])
  if (!impact) return <article className="rc-panel rc-impact"><span className="rc-kicker">AI RESPONSE IMPACT LAB</span><h2>Loading measured intervention comparison…</h2></article>
  const improved = impact.effect === 'POSITIVE'; const score = impact.impact_score
  return <article className="rc-panel rc-impact"><div className="impact-heading"><div><span className="rc-kicker">AI RESPONSE IMPACT LAB</span><h2>Observed baseline vs simulated intervention</h2></div><span className={`impact-effect ${impact.effect.toLowerCase()}`}>{impact.effect}</span></div><label className="impact-selector">Selected intervention<select value={scenario} onChange={(event) => setScenario(event.target.value)}>{scenarios.map((item) => <option key={item.scenario} value={item.scenario}>{label(item.scenario)}</option>)}</select></label><div className="impact-columns"><div><b>OBSERVED BASELINE</b><strong>{impact.before.risk}</strong><small>{label(impact.before.highest_zone)} · instability {Math.round(impact.before.instability)}</small></div><div className="impact-arrow">→</div><div><b>SIMULATED AFTER STATE</b><strong>{impact.after.risk}</strong><small>{label(impact.after.highest_zone)} · instability {Math.round(impact.after.instability)}</small></div></div><div className="impact-score"><span>INTERVENTION IMPACT SCORE</span><strong>{score}/100</strong>{best === scenario && <b>BEST ESTIMATED RESPONSE</b>}</div><div className="impact-metrics">{metricRows.map(([key, title, format]) => <div key={key}><span>{title}</span><b>{format(impact.before[key])} → {format(impact.after[key])}</b><Delta value={impact.deltas[key]} suffix={key === 'instability' ? '' : key === 'compression' || key === 'counter_flow' || key === 'stop_go' || key === 'motion_disorder' ? '%' : ''} /></div>)}{Object.keys(impact.before.zone_people || {}).map((zone) => <div key={zone}><span>{label(zone)} people</span><b>{Math.round(impact.before.zone_people[zone])} → {Math.round(impact.after.zone_people[zone])}</b><Delta value={impact.deltas.zones[zone]} suffix=" people" /></div>)}</div><button className="impact-why-button" onClick={() => setWhy((value) => !value)}>{why ? 'Hide why' : 'View why this response works'}</button>{why && <div className="impact-explanation">{improved ? <><p>• Instability changes by {impact.deltas.instability} points.</p><p>• Compression changes by {impact.deltas.compression}%.</p><p>• Counter-flow changes by {impact.deltas.counter_flow}%.</p><p>• {impact.new_hotspot ? `A new hotspot appears in ${label(impact.after.highest_zone)} and is penalized.` : 'No new higher-risk hotspot is created.'}</p></> : <p>THIS RESPONSE DOES NOT IMPROVE THE CURRENT SCENARIO.</p>}<small>Score breakdown: risk {Math.round(impact.impact_breakdown.risk)}, instability {Math.round(impact.impact_breakdown.instability)}, compression {Math.round(impact.impact_breakdown.compression)}, flow {Math.round(impact.impact_breakdown.flow)}, balance {Math.round(impact.impact_breakdown.balance)}.</small></div>}</article>
}
