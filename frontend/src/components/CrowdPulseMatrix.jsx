import { useEffect, useRef, useState } from 'react'

const ZONES = ['ZONE_A', 'ZONE_B', 'ZONE_C']
const label = (value) => String(value || '').replace('ZONE_', 'Zone ')
const levelFor = (score) => score >= 81 ? 'SEVERE' : score >= 66 ? 'HIGH' : score >= 46 ? 'UNSTABLE' : score >= 26 ? 'WATCH' : 'STABLE'
const palette = (score) => score >= 66 ? '#ff5968' : score >= 46 ? '#ff963f' : score >= 26 ? '#f4c34f' : '#41e3c2'

function PulseOrb({ metrics, zone }) {
  const ref = useRef(null)
  useEffect(() => {
    let frame
    const draw = (now) => {
      const canvas = ref.current; if (!canvas) return
      const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; canvas.width = Math.max(1, rect.width * dpr); canvas.height = Math.max(1, rect.height * dpr)
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height)
      const x = rect.width / 2; const y = rect.height / 2; const people = Math.max(0, Number(metrics?.people || 0)); const score = Math.max(0, Number(metrics?.instability_score || 0)); const compression = Math.max(0, Math.min(1, Number(metrics?.compression || 0))); const counter = Math.max(0, Math.min(1, Number(metrics?.counter_flow || 0))); const disorder = Math.max(0, Math.min(1, Number(metrics?.direction_disorder || 0))); const color = zone === 'ZONE_B' ? '#ff4fa3' : zone === 'ZONE_C' ? '#f4c34f' : '#41e3c2'; const period = Math.max(700, 3000 - score * 22); const pulse = 1 + Math.sin(now / period) * (.04 + score / 1200); const count = Math.round(20 + Math.min(90, people) / 90 * 70); const drift = Math.min(12, Number(metrics?.average_relative_speed || 0) / 5); const coreRadius = 20 + compression * 30
      if (compression > 0) { const halo = ctx.createRadialGradient(x, y, 2, x, y, coreRadius * 2.4); halo.addColorStop(0, `${color}${Math.round(70 + compression * 100).toString(16)}`); halo.addColorStop(1, `${color}00`); ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(x, y, coreRadius * 2.4, 0, Math.PI * 2); ctx.fill() }
      for (let i = 0; i < count; i += 1) { const seed = i * 17.37; const angle = seed + now / period; const radius = 10 + ((i * 31) % Math.max(18, coreRadius + 22)) * (1 - compression * .32); const px = x + Math.cos(angle) * radius + Math.sin(now / 1200 + i) * drift; const py = y + Math.sin(angle) * radius * .52; ctx.fillStyle = `${color}${i % 4 ? '99' : 'ee'}`; ctx.beginPath(); ctx.arc(px, py, 1.2 + Math.min(2.5, people / 70), 0, Math.PI * 2); ctx.fill() }
      ctx.strokeStyle = `${color}bb`; ctx.lineWidth = 1.4 + score / 100; ctx.beginPath(); ctx.ellipse(x, y, (42 + compression * 14) * pulse, (18 + compression * 8) * pulse, -.18 + Math.sin(now / 1700) * .12, 0, Math.PI * 2); ctx.stroke(); ctx.beginPath(); ctx.ellipse(x, y, 27 * pulse, (46 + compression * 10) * pulse, .65 + Math.cos(now / 2100) * .1, 0, Math.PI * 2); ctx.stroke()
      if (counter > 0.02) { ctx.strokeStyle = `${color}dd`; ctx.lineWidth = 1.5 + counter * 2; [-1, 1].forEach((direction) => { ctx.beginPath(); ctx.moveTo(x - 30, y + direction * 10); ctx.quadraticCurveTo(x, y - direction * 18, x + 30, y + direction * 10); ctx.stroke() }) }
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw); return () => cancelAnimationFrame(frame)
  }, [metrics])
  return <canvas className="pulse-orb" ref={ref} />
}

export default function CrowdPulseMatrix({ plan, radarData }) {
  const fallback = radarData?.snapshots || plan.instability_signal?.snapshots || [radarData || plan.instability_signal]; const [snapshot, setSnapshot] = useState(fallback[0] || {})
  useEffect(() => { const sync = (event) => { if (event.detail) setSnapshot(event.detail) }; window.addEventListener('crowd-radar-snapshot', sync); return () => window.removeEventListener('crowd-radar-snapshot', sync) }, [])
  useEffect(() => { if (radarData?.snapshots?.length) setSnapshot(radarData.snapshots[0]) }, [radarData])
  const metrics = snapshot.zone_metrics || {}; const instability = Number(snapshot.overall_instability || 0); const stability = Math.round(snapshot.overall_stability ?? 100 - instability); const status = levelFor(instability)
  return <section className="pulse-panel"><div className="pulse-heading"><div><span>LIVE ZONE INTELLIGENCE</span><h2>Crowd Pulse Matrix</h2></div><b className={`risk-${status.toLowerCase()}`}>Highest risk: {label(snapshot.highest_instability_zone)}</b></div><div className={`pulse-stability-mini status-${status.toLowerCase()}`}><span>CROWD STABILITY INDEX</span><strong>{stability}<small>/100</small></strong><b>{status}</b><em>Highest: {label(snapshot.highest_instability_zone)}</em></div><div className="pulse-zones">{ZONES.map((zone) => { const metric = metrics[zone] || {}; const score = Number(metric.instability_score || 0); const level = levelFor(score); return <article className={`pulse-zone ${zone.toLowerCase()} level-${level.toLowerCase()}`} key={zone}><h3>{label(zone)}</h3><small>{Number(metric.people || 0)} people</small><PulseOrb metrics={metric} zone={zone} /><strong>{level}</strong><b>Score {Math.round(score)}</b><div><span>Compression {Math.round(Number(metric.compression || 0) * 100)}%</span><span>Counter-flow {Math.round(Number(metric.counter_flow || 0) * 100)}%</span><span>Stop-Go {Number(metric.stop_go || 0) > .1 ? 'DETECTED' : 'CLEAR'}</span><span>Motion {Math.round(Number(metric.direction_disorder || 0) * 100)}%</span><span>Relative speed {Number(metric.average_relative_speed || 0).toFixed(2)}</span><span>Trend {metric.trend || 'STABLE'}</span></div></article>})}</div></section>
}
