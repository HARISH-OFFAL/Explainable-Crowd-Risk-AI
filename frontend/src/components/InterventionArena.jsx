import { useEffect, useRef, useState } from 'react'

const API = 'http://127.0.0.1:8000'
const ZONES = ['ZONE_A', 'ZONE_B', 'ZONE_C']
const label = (value) => String(value || '').replaceAll('_', ' ').replace('RESPONSE', '').trim()
const tone = (risk) => risk === 'HIGH' || risk === 'SEVERE' ? '#ff5d69' : risk === 'UNSTABLE' || risk === 'WATCH' || risk === 'MEDIUM' ? '#f4c34f' : '#52e3c5'
const riskFor = (score) => score >= 66 ? 'HIGH' : score >= 46 ? 'UNSTABLE' : score >= 26 ? 'WATCH' : 'STABLE'

function MiniFuture({ impact }) {
  const ref = useRef(null)
  useEffect(() => {
    let frame; const started = performance.now(); const timeline = impact.simulation?.timeline || []
    const draw = (now) => {
      const canvas = ref.current; if (!canvas || !impact) return
      const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; canvas.width = Math.max(1, rect.width * dpr); canvas.height = Math.max(1, rect.height * dpr)
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height)
      const seconds = ((now - started) % 4200) / 4200 * 30; const lower = timeline.reduce((best, item) => item.timestamp <= seconds && item.timestamp >= best.timestamp ? item : best, timeline[0]); const upper = timeline.find((item) => item.timestamp >= seconds) || lower; const ratio = lower && upper && upper.timestamp !== lower.timestamp ? (seconds - lower.timestamp) / (upper.timestamp - lower.timestamp) : 0
      const people = Object.fromEntries(ZONES.map((zone) => { const a = Number(lower?.zones?.[zone] || impact.before.zone_people?.[zone] || 0); const b = Number(upper?.zones?.[zone] || impact.after.zone_people?.[zone] || a); return [zone, a + (b - a) * ratio] })); const max = Math.max(...Object.values(people), 1); const score = Number(impact.before.instability || 0) + (Number(impact.after.instability || 0) - Number(impact.before.instability || 0)) * seconds / 30; const color = tone(riskFor(score))
      ZONES.forEach((zone, index) => { const x = (index + 1) * rect.width / 4; const radius = 12 + people[zone] / max * 24; const glow = ctx.createRadialGradient(x, rect.height / 2, 1, x, rect.height / 2, radius * 2.2); glow.addColorStop(0, `${color}bb`); glow.addColorStop(1, 'rgba(0,0,0,0)'); ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x, rect.height / 2, radius * 2.2, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, rect.height / 2, radius, 0, Math.PI * 2); ctx.stroke(); ctx.fillStyle = '#eaf4ff'; ctx.font = '700 9px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(zone.replace('ZONE_', 'Z'), x, rect.height / 2 + 3) })
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw); return () => cancelAnimationFrame(frame)
  }, [impact])
  return <canvas className="arena-mini-canvas" ref={ref} />
}

export default function InterventionArena({ plan }) {
  const scenarios = plan.simulations || []; const [impacts, setImpacts] = useState([]); const [selected, setSelected] = useState(plan.recommended?.scenario || scenarios[0]?.scenario)
  useEffect(() => { let active = true; Promise.all(scenarios.map((item) => fetch(`${API}/events/${plan.event_id}/response-commander/plans/${plan.id}/impact?scenario=${item.scenario}`).then((response) => response.json()))).then((data) => { if (active) setImpacts(data) }).catch(() => {}); return () => { active = false } }, [plan])
  const best = impacts.reduce((winner, item) => !winner || item.impact_score > winner.impact_score ? item : winner, null); const selectedImpact = impacts.find((item) => item.scenario === selected)
  return <section className="arena-panel"><div className="arena-heading"><div><span>AI INTERVENTION ARENA</span><h2>Simulate and compare response futures</h2></div><small>BACKEND SIMULATIONS - NO PHYSICAL CONTROL</small></div><div className="arena-cards">{impacts.map((impact) => { const current = Math.round(impact.before.instability); const after = Math.round(impact.after.instability); const risk = impact.after.risk || riskFor(after); const target = impact.before.highest_zone || plan.risk_zone; return <button className={`arena-card risk-${risk.toLowerCase()} ${selected === impact.scenario ? 'selected' : ''}`} key={impact.scenario} onClick={() => setSelected(impact.scenario)}>{best?.scenario === impact.scenario && <em>BEST ESTIMATED</em>}<b>{label(impact.scenario)}</b><small>{impact.new_hotspot ? 'Risk transfer detected' : impact.effect === 'POSITIVE' ? 'Crowd condition improves' : impact.effect === 'NEGATIVE' ? 'Risk increases' : 'Current trend continues'}</small><MiniFuture impact={impact} /><span>Risk <strong>{risk}</strong></span><span>Instability <strong>{current} -&gt; {after}</strong></span><span>{label(target)} people <strong>{Math.round(impact.before.zone_people?.[target] || 0)} -&gt; {Math.round(impact.after.zone_people?.[target] || 0)}</strong></span><span>Impact score <strong>{impact.impact_score}/100</strong></span></button>})}</div>{selectedImpact && <div className="arena-selected-detail"><b>{label(selectedImpact.scenario)}: CURRENT vs SIMULATED +30 SEC</b><div>{ZONES.map((zone) => <span key={zone}>{label(zone)} {Math.round(selectedImpact.before.zone_people?.[zone] || 0)} -&gt; {Math.round(selectedImpact.after.zone_people?.[zone] || 0)} people</span>)}</div><small>{selectedImpact.new_hotspot ? 'WHY THIS RESPONSE IS NOT RECOMMENDED: a new high-risk hotspot is created.' : selectedImpact.after.instability < selectedImpact.before.instability ? `WHY THIS RESPONSE HELPS: instability reduces from ${Math.round(selectedImpact.before.instability)} to ${Math.round(selectedImpact.after.instability)}.` : 'WHY THIS RESPONSE IS NOT RECOMMENDED: overall instability does not improve.'}</small></div>}</section>
}

