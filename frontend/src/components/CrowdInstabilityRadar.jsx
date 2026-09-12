import { useEffect, useMemo, useRef, useState } from 'react'
import CrowdPulseMatrix from './CrowdPulseMatrix'

const API = 'http://127.0.0.1:8000'
const ZONES = ['ZONE_A', 'ZONE_B', 'ZONE_C']
const label = (value) => String(value || '').replaceAll('_', ' ').replace('ZONE ', 'Zone ')
const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, value))
const levelFor = (score) => score >= 66 ? 'HIGH' : score >= 36 ? 'WATCH' : 'STABLE'

function colorFor(score, alpha = 1) {
  const stops = [[0, [255, 145, 62]], [26, [255, 190, 71]], [46, [255, 125, 48]], [66, [255, 76, 92]], [81, [214, 59, 255]]]
  const value = clamp(score, 0, 100); let left = stops[0]; let right = stops[stops.length - 1]
  for (let index = 0; index < stops.length - 1; index += 1) if (value >= stops[index][0] && value <= stops[index + 1][0]) { left = stops[index]; right = stops[index + 1]; break }
  const ratio = (value - left[0]) / Math.max(right[0] - left[0], 1); const rgb = left[1].map((channel, index) => Math.round(channel + (right[1][index] - channel) * ratio))
  return `rgba(${rgb.join(',')},${alpha})`
}

function CrowdInstabilityRadar({ plan, radarData }) {
  const [radar, setRadar] = useState(radarData || null); const [horizon, setHorizon] = useState(0); const [snapshotIndex, setSnapshotIndex] = useState(0)
  const canvasRef = useRef(null); const videoRef = useRef(null); const animationRef = useRef({ x: 0, y: 0, tx: 0, ty: 0, angle: 0, particles: new Map(), last: 0, ripples: [], ready: false })
  const videoName = plan?.source_state?.source || plan?.source_state?.source_name

  useEffect(() => {
    if (radarData) { setRadar(radarData); return undefined }
    let active = true
    fetch(`${API}/events/${plan.event_id}/instability-radar/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ monitoring_session_id: plan.monitoring_session_id, flow_analysis_id: plan.flow_analysis_id }) })
      .then((response) => response.json()).then((data) => { if (active && data.analysis_id) setRadar(data) }).catch(() => {})
    return () => { active = false }
  }, [plan, radarData])

  const snapshots = radar?.snapshots || []; const live = snapshots[snapshotIndex] || radar
  const forecast = useMemo(() => radar?.forecast?.find((item) => item.horizon_seconds === horizon) || radar?.forecast?.[0], [radar, horizon])
  const forecastMetrics = Object.fromEntries(ZONES.map((zone) => [zone, { ...(live?.zone_metrics?.[zone] || {}), instability_score: forecast?.zone_scores?.[zone] ?? (live?.zone_metrics?.[zone]?.instability_score || 0) }]))
  const view = horizon === 0 ? live : live && { ...live, field: forecast?.field || live.field, zone_metrics: forecastMetrics, propagation: forecast?.propagation || live.propagation }
  const scores = Object.fromEntries(ZONES.map((zone) => [zone, view?.zone_metrics?.[zone]?.instability_score || 0])); const highest = view?.zone_metrics?.[view?.highest_instability_zone]; const radarLevel = levelFor(Number(view?.overall_instability || 0))

  useEffect(() => {
    if (!radar || !snapshots.length) return undefined
    let frame
    let lastNearest = -1
    const tick = () => { const time = videoRef.current?.currentTime; const target = Number.isFinite(time) ? time : snapshots[0].timestamp; let nearest = 0; snapshots.forEach((item, index) => { if (Math.abs(item.timestamp - target) < Math.abs(snapshots[nearest].timestamp - target)) nearest = index }); setSnapshotIndex(nearest); if (nearest !== lastNearest) { lastNearest = nearest; window.dispatchEvent(new CustomEvent('crowd-radar-snapshot', { detail: snapshots[nearest] })) }; frame = requestAnimationFrame(tick) }
    frame = requestAnimationFrame(tick); return () => cancelAnimationFrame(frame)
  }, [radar, snapshots])

  useEffect(() => {
    if (!view) return undefined
    let frame
    const draw = (now) => {
      const canvas = canvasRef.current; if (!canvas) return
      const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; canvas.width = Math.max(1, rect.width * dpr); canvas.height = Math.max(1, rect.height * dpr)
      const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height)
      const sourceWidth = radar.frame_width || 1280; const sourceHeight = radar.frame_height || 720; const point = (item) => [item.x / sourceWidth * rect.width, item.y / sourceHeight * rect.height]
      const state = animationRef.current; const dt = Math.min((now - (state.last || now)) / 1000, .05); state.last = now; const cells = view.field || []
      const weighted = cells.filter((cell) => cell.people > 0); const totalWeight = weighted.reduce((sum, cell) => sum + cell.people * (1 + (cell.instability_score || 0) / 100), 0); const target = weighted.length ? weighted.reduce((sum, cell) => ({ x: sum.x + cell.x * cell.people * (1 + cell.instability_score / 100), y: sum.y + cell.y * cell.people * (1 + cell.instability_score / 100) }), { x: 0, y: 0 }) : { x: sourceWidth / 2, y: sourceHeight / 2 }; state.tx = clamp(target.x / Math.max(totalWeight, 1), .14, .86) * rect.width; state.ty = clamp(target.y / Math.max(totalWeight, 1), .14, .86) * rect.height; if (!state.ready) { state.x = state.tx; state.y = state.ty; state.ready = true } const follow = 1 - Math.exp(-dt / .7); state.x += (state.tx - state.x) * follow; state.y += (state.ty - state.y) * follow; state.angle += dt * .18
      const sx = state.x; const sy = state.y; const risk = view.overall_instability || 0; const aura = 90 + Math.sqrt(weighted.reduce((sum, cell) => sum + cell.people * cell.people, 0)) * 5 + risk * 1.2

      // Pass 1: smooth instability field anchored to live grid cells.
      ctx.globalCompositeOperation = 'lighter'; cells.forEach((cell) => { const [x, y] = point(cell); const radius = Math.max(25, Math.min(95, (cell.width || 64) * 1.35)); const glow = ctx.createRadialGradient(x, y, 2, x, y, radius); glow.addColorStop(0, colorFor(cell.instability_score, .25)); glow.addColorStop(.55, colorFor(cell.instability_score, .08)); glow.addColorStop(1, colorFor(cell.instability_score, 0)); ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill() })
      // Pass 2: cinematic floating scanner with perspective ellipses and sweep.
      ctx.save(); ctx.translate(sx, sy); ctx.rotate(-.08); [1, .78, .56].forEach((scale, index) => { ctx.strokeStyle = colorFor(risk, .18 - index * .035); ctx.lineWidth = 2 - index * .35; ctx.beginPath(); ctx.ellipse(0, 0, aura * scale, aura * .34 * scale, 0, 0, Math.PI * 2); ctx.stroke() }); const core = ctx.createRadialGradient(0, 0, 2, 0, 0, aura * .55); core.addColorStop(0, colorFor(risk, .32)); core.addColorStop(1, colorFor(risk, 0)); ctx.fillStyle = core; ctx.beginPath(); ctx.ellipse(0, 0, aura * .75, aura * .26, 0, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = colorFor(risk, .65); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(0, 0, aura * .7, state.angle, state.angle + Math.PI * .62); ctx.stroke(); ctx.restore();
      // Stabilized drone-like direction markers stay with the scanner and point
      // along the local crowd flow instead of blinking at fixed coordinates.
      const meanFlow = weighted.reduce((sum, cell) => ({ x: sum.x + cell.vx * cell.people, y: sum.y + cell.vy * cell.people }), { x: 0, y: 0 }); const flowAngle = Math.atan2(meanFlow.y, meanFlow.x); const flowLength = Math.min(70, Math.max(24, Math.hypot(meanFlow.x, meanFlow.y) * .55)); ctx.save(); ctx.translate(sx, sy); ctx.rotate(flowAngle); ctx.strokeStyle = colorFor(risk, .8); ctx.shadowColor = colorFor(risk, .85); ctx.shadowBlur = 16; ctx.lineWidth = 3; [-1, 0, 1].forEach((offset) => { const y = offset * 13; ctx.beginPath(); ctx.moveTo(-flowLength, y); ctx.quadraticCurveTo(-flowLength * .35, y - offset * 8, flowLength, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(flowLength, y); ctx.lineTo(flowLength - 12, y - 7); ctx.moveTo(flowLength, y); ctx.lineTo(flowLength - 12, y + 7); ctx.stroke() }); ctx.restore()
      // Pass 3: deterministic crowd-energy particles, moving with cell velocity.
      let particleBudget = 320; cells.forEach((cell, cellIndex) => { if (particleBudget <= 0) return; const [cx, cy] = point(cell); const count = Math.min(12, particleBudget, Math.max(1, Math.round(cell.people / 2))); particleBudget -= count; for (let particleIndex = 0; particleIndex < count; particleIndex += 1) { const key = `${cellIndex}:${particleIndex}`; let particle = state.particles.get(key); if (!particle) { const seed = (cellIndex * 37 + particleIndex * 17) % 100; particle = { ox: (seed - 50) * .35, oy: ((seed * 7) % 100 - 50) * .22, phase: seed / 100 }; state.particles.set(key, particle) } particle.ox += (cell.vx || 0) * dt * .04; particle.oy += (cell.vy || 0) * dt * .04; const px = cx + particle.ox; const py = cy + particle.oy; const size = 1.2 + py / rect.height * 2.8; ctx.fillStyle = `rgba(255, 59, 72, ${.52 + clamp(cell.people / 12) * .4})`; ctx.shadowColor = 'rgba(255, 44, 68, .85)'; ctx.shadowBlur = 8; ctx.beginPath(); ctx.arc(px, py, size, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0 } })
      // Pass 4: flowing ribbons from local velocity; opposite cells visibly cross.
      cells.filter((cell) => cell.people > 1 && Math.hypot(cell.vx, cell.vy) > .5).forEach((cell) => { const [x, y] = point(cell); const length = Math.min(100, Math.max(22, Math.hypot(cell.vx, cell.vy) * .65)); const angle = Math.atan2(cell.vy, cell.vx); ctx.strokeStyle = colorFor(cell.instability_score, .55); ctx.lineWidth = 2 + cell.people / 10; ctx.beginPath(); ctx.moveTo(x - Math.cos(angle) * length, y - Math.sin(angle) * length); ctx.quadraticCurveTo(x, y - Math.sin(angle) * 22, x + Math.cos(angle) * length, y + Math.sin(angle) * length); ctx.stroke() })
      // Pass 5: compression volume attached to the measured centroid.
      if (view.compression_centroid) { const [x, y] = point(view.compression_centroid); const cloud = ctx.createRadialGradient(x, y, 4, x, y, 90); cloud.addColorStop(0, 'rgba(255,89,52,.34)'); cloud.addColorStop(1, 'rgba(255,89,52,0)'); ctx.fillStyle = cloud; ctx.beginPath(); ctx.ellipse(x, y, 115, 58, -.08, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = 'rgba(255,127,72,.8)'; ctx.setLineDash([8, 9]); ctx.stroke(); ctx.setLineDash([]) }
      // Pass 6: short-lived stop-go ripple at the actual detector centroid.
      if (view.stop_go_centroid && (view.stop_go_centroid.x !== state.lastStopX || view.stop_go_centroid.y !== state.lastStopY)) { state.ripples.push({ ...view.stop_go_centroid, age: 0 }); state.lastStopX = view.stop_go_centroid.x; state.lastStopY = view.stop_go_centroid.y }
      state.ripples = state.ripples.filter((ripple) => ripple.age < 1.8); state.ripples.forEach((ripple) => { ripple.age += dt; const [x, y] = point(ripple); ctx.strokeStyle = `rgba(255,196,61,${1 - ripple.age / 1.8})`; ctx.lineWidth = 2; ctx.beginPath(); ctx.ellipse(x, y, 20 + ripple.age * 75, 8 + ripple.age * 28, 0, 0, Math.PI * 2); ctx.stroke() })
      // Pass 7: violet propagation trail only when the backend has evidence.
      if (view.propagation?.from && view.propagation?.to) { const [x1, y1] = point(view.propagation.from); const [x2, y2] = point(view.propagation.to); ctx.strokeStyle = 'rgba(181,112,255,.75)'; ctx.lineWidth = 3; ctx.setLineDash([12, 10]); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo((x1 + x2) / 2, Math.min(y1, y2) - 45, x2, y2); ctx.stroke(); ctx.setLineDash([]) }
      // Pass 8: dynamic zone labels and tracker label.
      ZONES.forEach((zone) => { const center = view.zone_centers?.[zone]; if (!center) return; const [x, y] = point(center); const score = scores[zone]; ctx.fillStyle = 'rgba(4,12,24,.86)'; ctx.strokeStyle = colorFor(score, .9); ctx.lineWidth = 2; ctx.beginPath(); ctx.roundRect(x - 47, y - 24, 94, 48, 8); ctx.fill(); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillStyle = '#f4f8ff'; ctx.font = '700 12px Inter, sans-serif'; ctx.fillText(label(zone), x, y - 4); ctx.fillStyle = colorFor(score); ctx.font = '800 11px Inter, sans-serif'; ctx.fillText(`${view.zone_metrics?.[zone]?.level || 'STABLE'}  |  |  ${Math.round(score)}`, x, y + 14) }); ctx.fillStyle = '#eafcff'; ctx.font = '800 11px Inter, sans-serif'; ctx.textAlign = 'left'; ctx.fillText('ACTIVE INSTABILITY SCAN', sx + 18, sy - 18); ctx.font = '600 10px Inter, sans-serif'; ctx.fillStyle = colorFor(risk); ctx.fillText(`Tracking ${label(view.highest_instability_zone)}  |  |  Score ${Math.round(risk)}  |  |  ${view.level}`, sx + 18, sy - 4); ctx.globalCompositeOperation = 'source-over'
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw); return () => cancelAnimationFrame(frame)
  }, [view, radar, scores])

  const reasons = highest ? [highest.compression >= .15 && 'Compression volume is elevated', highest.counter_flow >= .15 && 'Opposing movement streams detected', highest.stop_go > .1 && 'Stop-go transitions detected', highest.direction_disorder >= .5 && 'Direction disorder is elevated'].filter(Boolean) : []
  return <div className="cr-radar"><header><div><span>CROWD INSTABILITY RADAR</span></div><small>FLOW ANALYSIS #{plan.flow_analysis_id}  |  |  {live ? `${live.timestamp.toFixed(1)}s` : 'ANALYZING'}</small></header><div className="cr-radar-main"><div className="cr-video"><video ref={videoRef} src={videoName ? `${API}/monitoring-videos/${encodeURIComponent(videoName)}` : undefined} autoPlay muted loop playsInline /><canvas ref={canvasRef} /><div className="cr-legend"><b>LIVE CROWD FIELD</b><i className="stable" /> Stable <i className="watch" /> Watch <i className="unstable" /> Unstable <i className="high" /> High <i className="severe" /> Severe</div></div><aside className="cr-panel"><span>CROWD STABILITY INDEX</span><strong>{live ? Math.round(100 - live.overall_instability) : ' |  | '}<small>/100</small></strong><b>Instability: {radarLevel || 'ANALYZING'}</b>{live && <><p>Compression Proxy <em>{Math.round((highest?.compression || 0) * 100)}%</em></p><p>Counter-flow <em>{Math.round((highest?.counter_flow || 0) * 100)}%</em></p><p>Stop-Go <em>{highest?.stop_go > .1 ? 'DETECTED' : 'NOT DETECTED'}</em></p><p>Motion Disorder <em>{Math.round((highest?.direction_disorder || 0) * 100)}%</em></p><p>Relative Speed Index <em>{live.relative_speed_index}</em></p><h3>Highest Instability Zone<br /><strong>{label(view?.highest_instability_zone)}</strong></h3><p>Propagation: {view?.propagation?.label || 'Propagation not established'}</p></>}</aside></div><div className="cr-timeline"><b>RADAR TIMELINE  |  |  {live ? `${live.timestamp.toFixed(1)}s` : 'WAITING'}</b>{[0, 10, 20, 30].map((value) => <button key={value} className={horizon === value ? 'active' : ''} onClick={() => setHorizon(value)}>{value === 0 ? 'NOW' : `+${value}s`}</button>)}</div><div className="cr-bottom"><div className="cr-zone-cards">{ZONES.map((zone) => <article key={zone}><span>{label(zone)}</span><strong>{Math.round(scores[zone])}</strong><b>{levelFor(Number(scores[zone]))}</b></article>)}</div><article className="cr-why"><b>LIVE ANALYTICS DEBUG</b>{reasons.length ? reasons.map((reason) => <p key={reason}> |  |  {reason}</p>) : <p>Propagation: {view?.propagation?.label || 'Propagation not established'}</p>}<small className="cr-debug">t={live?.timestamp?.toFixed(2)}s  |  |  tracks={live?.active_tracks || 0}  |  |  cells={live?.occupied_cells || 0}  |  |  scanner target follows weighted density + instability</small></article></div></div>
}

export default CrowdInstabilityRadar









