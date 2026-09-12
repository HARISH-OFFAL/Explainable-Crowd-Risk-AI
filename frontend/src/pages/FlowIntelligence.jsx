import { Component, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Navbar from '../components/Navbar'
import ForkliftLoader from '../components/ForkliftLoader'

const API_URL = 'http://127.0.0.1:8000'
const ZONES = ['ZONE_A', 'ZONE_B', 'ZONE_C']
const label = (zone) => zone.replace('ZONE_', 'Zone ')

function safeNumber(value, fallback = 0) {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

function safeTrajectories(result) {
  if (!Array.isArray(result?.trajectories)) return []
  const minimumMovement = Math.max(safeNumber(result?.frame?.width, 1), safeNumber(result?.frame?.height, 1)) * .004
  return result.trajectories.map((trajectory, index) => ({
    id: trajectory.track_id ?? `trajectory-${index}`,
    segments: (() => {
      const raw = Array.isArray(trajectory.points) ? trajectory.points.map((point) => ({ x: safeNumber(point?.x, NaN), y: safeNumber(point?.y, NaN), timestamp: safeNumber(point?.timestamp, NaN) })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)) : []
      const latest = raw.length ? raw[raw.length - 1].timestamp : NaN
      const recent = Number.isFinite(latest) ? raw.filter((point) => !Number.isFinite(point.timestamp) || latest - point.timestamp <= 8) : raw
      const diagonal = Math.hypot(safeNumber(result?.frame?.width, 1), safeNumber(result?.frame?.height, 1)); const segments = []; let segment = []
      recent.forEach((point) => { const previous = segment[segment.length - 1]; const distance = previous ? Math.hypot(point.x - previous.x, point.y - previous.y) : 0; const gap = previous && Number.isFinite(point.timestamp) && Number.isFinite(previous.timestamp) ? point.timestamp - previous.timestamp : 0; if (previous && (distance / diagonal > .09 || gap > .75 || (Number.isFinite(point.timestamp) && Number.isFinite(previous.timestamp) && gap < 0))) { if (segment.length >= 2) segments.push(segment); segment = [] } segment.push(point) }); if (segment.length >= 2) segments.push(segment); return segments
    })(),
  })).filter((trajectory) => trajectory.segments.length > 0 && trajectory.segments.some((segment) => segment.slice(1).reduce((distance, point, index) => distance + Math.hypot(point.x - segment[index].x, point.y - segment[index].y), 0) >= minimumMovement))
}

function trailOpacity(point, latest) {
  if (!Number.isFinite(point.timestamp) || !Number.isFinite(latest)) return .42
  const age = Math.max(0, latest - point.timestamp)
  if (age >= 6) return .08
  if (age >= 4) return .17
  if (age >= 2) return .30
  return .52
}

function TrajectoryOverlay({ result }) {
  const width = Math.max(1, safeNumber(result?.frame?.width, 1)); const height = Math.max(1, safeNumber(result?.frame?.height, 1))
  const vectors = result?.zone_vectors && typeof result.zone_vectors === 'object' ? result.zone_vectors : {}
  const trajectories = safeTrajectories(result)
  return <svg className="flow-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" aria-label="Crowd movement trajectories">
    <defs><marker id="flow-arrow-head" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L7,3.5 L0,7" fill="none" stroke="#c084fc" strokeWidth="1.2" /></marker><marker id="person-direction-head" markerWidth="9" markerHeight="7" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L7,3.5 L0,7" fill="#b76cff" fillOpacity=".92" stroke="#f3e8ff" strokeWidth=".7" /></marker></defs>
    {ZONES.slice(0, -1).map((zone, index) => <line className="flow-zone-boundary" key={`boundary-${zone}`} x1={width * ((index + 1) / 3)} y1="0" x2={width * ((index + 1) / 3)} y2={height} />)}
    {trajectories.map((trajectory) => { const latest = trajectory.segments[trajectory.segments.length - 1]?.at(-1)?.timestamp; return trajectory.segments.flatMap((segment, segmentIndex) => segment.slice(1).map((point, pointIndex) => { const previous = segment[pointIndex]; return <line className="person-trajectory" key={`${trajectory.id}-${segmentIndex}-${pointIndex}`} x1={previous.x} y1={previous.y} x2={point.x} y2={point.y} style={{ opacity: trailOpacity(previous, latest) }} /> })) })}
    {trajectories.map((trajectory) => { const segment = trajectory.segments[trajectory.segments.length - 1]; const current = segment[segment.length - 1]; const previous = segment[segment.length - 2]; if (!previous || !current) return null; const dx = current.x - previous.x; const dy = current.y - previous.y; const distance = Math.hypot(dx, dy); if (distance < 1.5) return null; const length = Math.min(Math.min(width, height) * .06, Math.max(34, distance * 2.8)); return <line className="person-direction" key={`direction-${trajectory.id}`} x1={current.x} y1={current.y} x2={current.x + (dx / distance) * length} y2={current.y + (dy / distance) * length} markerEnd="url(#person-direction-head)" /> })}
    {trajectories.map((trajectory) => { const segment = trajectory.segments[trajectory.segments.length - 1]; const point = segment[segment.length - 1]; return <circle className="person-trajectory-head" key={`head-${trajectory.id}`} cx={point.x} cy={point.y} r="2.2" /> })}
    {ZONES.map((zone, index) => { const vector = vectors[zone] || {}; const dx = safeNumber(vector.dx); const dy = safeNumber(vector.dy); const magnitude = Math.hypot(dx, dy); const length = Math.min(Math.min(width, height) * .085, magnitude * Math.min(width, height) * .0018); const x = width * ((index + .5) / 3); const y = height * .18; return <line className="flow-arrow" key={`arrow-${zone}`} x1={x} y1={y} x2={x + (magnitude ? dx / magnitude * length : 0)} y2={y + (magnitude ? dy / magnitude * length : 0)} markerEnd="url(#flow-arrow-head)" /> })}
  </svg>
}

async function requestJson(url, options) {
  const response = await fetch(url, options)
  let data = null
  try { data = await response.json() } catch { data = null }
  if (!response.ok) { const error = new Error(response.status === 404 ? 'The requested Flow Analysis was not found.' : response.status >= 500 ? 'The Flow Analysis service is temporarily unavailable.' : data?.detail || 'Unable to load Flow Intelligence.'); error.status = response.status; throw error }
  return data
}

function FlowShell({ event, flowAnalysisId, monitoringSessionId, children, navigate }) {
  return <div className="app flow-launch-template"><Navbar /><main className="flow-page"><aside className="flow-sidebar"><div className="flow-sidebar-context"><span className="control-kicker">Phase 3 / Intelligence</span><h2>{event?.event_name || 'Event'}</h2></div><div className="flow-sidebar-actions"><button className="active">Flow Intelligence</button><button onClick={() => navigate(`/organizer/events/${event?.id}/digital-twin/${monitoringSessionId}`)}>Digital Twin Simulation</button></div></aside><section className="flow-content"><header className="flow-header"><div><span className="control-kicker">Phase 3A / Flow Intelligence</span><h1>Understand every movement.</h1><p>{event?.event_name || 'Event'} · Event #{event?.id || '—'} · Flow Analysis #{flowAnalysisId}</p></div><span className="flow-status">Observed movement</span></header>{children}</section></main></div>
}

function FlowIntelligence() {
  const { eventId, flowAnalysisId } = useParams(); const navigate = useNavigate()
  const [event, setEvent] = useState(null); const [analysis, setAnalysis] = useState(null); const [loading, setLoading] = useState(true); const [error, setError] = useState(''); const [zone, setZone] = useState('ZONE_A')

  useEffect(() => {
    const controller = new AbortController(); let active = true
    async function load() {
      try {
        const eventData = await requestJson(`${API_URL}/events/${eventId}`, { signal: controller.signal })
        if (active) setEvent(eventData)
        let flowData = await requestJson(`${API_URL}/flow-analyses/${flowAnalysisId}?event_id=${eventId}`, { signal: controller.signal })
        if (flowData.status === 'NOT_STARTED' || (flowData.status === 'COMPLETE' && !flowData.result)) flowData = await requestJson(`${API_URL}/monitoring-sessions/${flowData.monitoring_session_id}/flow-analysis`, { method: 'POST', signal: controller.signal })
        if (active) setAnalysis(flowData)
      } catch (reason) { if (active && reason.name !== 'AbortError') setError(reason.message) } finally { if (active) setLoading(false) }
    }
    load(); return () => { active = false; controller.abort() }
  }, [eventId, flowAnalysisId])

  useEffect(() => {
    if (!analysis || !['QUEUED', 'RUNNING'].includes(analysis.status)) return undefined
    const timer = setInterval(() => requestJson(`${API_URL}/flow-analyses/${flowAnalysisId}?event_id=${eventId}`).then(setAnalysis).catch(() => {}), 1000)
    return () => clearInterval(timer)
  }, [analysis, eventId, flowAnalysisId])

  if (loading) return <FlowShell event={event} flowAnalysisId={flowAnalysisId} navigate={navigate}><ForkliftLoader label="Loading Flow Intelligence" /></FlowShell>
  if (error) return <FlowShell event={event} flowAnalysisId={flowAnalysisId} navigate={navigate}><div className="flow-empty"><h2>Flow Intelligence unavailable</h2><p>Unable to load Flow Analysis #{flowAnalysisId}. {error}</p><button className="flow-run-button" onClick={() => window.location.reload()}>Retry</button><button className="back-control" onClick={() => navigate(`/organizer/events/${eventId}/monitoring`)}>Back to Monitoring</button></div></FlowShell>
  if (analysis?.status === 'ERROR' || analysis?.status === 'FAILED') return <FlowShell event={event} flowAnalysisId={flowAnalysisId} navigate={navigate}><div className="flow-empty"><h2>Flow Analysis failed</h2><p>The recorded analysis could not be completed. Please return to Monitoring and try again.</p><button className="back-control" onClick={() => navigate(`/organizer/events/${eventId}/monitoring`)}>Back to Monitoring</button></div></FlowShell>
  if (analysis?.status !== 'COMPLETE' || !analysis?.result) return <FlowShell event={event} flowAnalysisId={analysis?.flow_analysis_id || flowAnalysisId} monitoringSessionId={analysis?.monitoring_session_id} navigate={navigate}><ForkliftLoader label={`Flow Intelligence ${analysis?.status === 'RUNNING' ? 'in progress' : 'queued'}`} /></FlowShell>

  const result = analysis.result; const selectedZone = result.zones?.[zone] || {}; const trend = Array.isArray(result.trend) ? result.trend : []; const peak = Math.max(...trend.map((point) => safeNumber(point.average_speed)), 1); const trendPoints = trend.map((point, index) => `${trend.length === 1 ? 50 : (index / (trend.length - 1)) * 100},${96 - Math.min(82, safeNumber(point.average_speed) / peak * 82)}`).join(' '); const videoUrl = `${API_URL}/monitoring-videos/${encodeURIComponent(result.source || analysis.source_name || '')}`; const sessionId = analysis.monitoring_session_id
  return <FlowShell event={event} flowAnalysisId={analysis.flow_analysis_id} monitoringSessionId={sessionId} navigate={navigate}>
    <section className="flow-metrics"><article><span>FLOW STATE</span><strong>{result.flow_state || 'INSUFFICIENT DATA'}</strong></article><article><span>ACTIVE TRACKS</span><strong>{safeNumber(result.active_tracks)}</strong></article><article><span>DOMINANT DIRECTION</span><strong>{result.dominant_direction || '—'}</strong></article><article><span>AVERAGE MOVEMENT</span><strong>{safeNumber(result.average_relative_speed).toFixed(1)} <small>px/s</small></strong></article></section>
    <section className="flow-card flow-video-panel"><div className="flow-panel-heading"><div><span className="control-kicker">RECORDED SOURCE</span><h2>{result.source || analysis.source_name || 'Source video'}</h2></div><small>Cached completed analysis · {safeNumber(result.sampled_frames)} sampled frames</small></div><div className="flow-video-stage"><video src={videoUrl} controls playsInline /><TrajectoryOverlay result={result} /></div></section>
    <section className="flow-lower"><article className="flow-card"><div className="flow-panel-heading"><div><span className="control-kicker">ZONE MOVEMENT</span><h2>Flow by zone</h2></div></div><div className="flow-table"><div className="flow-table-row flow-table-head"><span>Zone</span><span>People</span><span>Inflow</span><span>Outflow</span><span>Net</span><span>State</span></div>{ZONES.map((item) => { const metrics = result.zones?.[item] || {}; return <button className={`flow-table-row ${zone === item ? 'selected' : ''}`} key={item} onClick={() => setZone(item)}><span>{label(item)}</span><span>{safeNumber(metrics.people)}</span><span>{safeNumber(metrics.inflow).toFixed(1)}</span><span>{safeNumber(metrics.outflow).toFixed(1)}</span><span>{safeNumber(metrics.net).toFixed(1)}</span><span>{metrics.state || 'Balanced'}</span></button> })}</div><div className="flow-selected"><strong>{label(zone)} selected</strong><span>{selectedZone.state || 'Balanced'} · average movement {safeNumber(selectedZone.average_movement).toFixed(1)}</span></div></article><article className="flow-card"><div className="flow-panel-heading"><div><span className="control-kicker">SPEED TREND</span><h2>Observed movement</h2></div><small>{trend.length} samples</small></div>{trend.length ? <div className="flow-speed-chart"><svg viewBox="0 0 100 100" preserveAspectRatio="none"><polyline points={trendPoints} /></svg></div> : <div className="flow-empty"><p>No movement trend data was returned.</p></div>}<div className="bottleneck-panel"><span>ACCUMULATION SIGNAL</span><strong>{result.most_accumulating_zone ? label(result.most_accumulating_zone) : 'None identified'}</strong><b>{result.counter_flow ? 'Counter flow detected' : 'No counter flow detected'}</b><small>Observed analysis only; this page does not forecast future risk.</small></div></article></section>
  </FlowShell>
}

class FlowErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { failed: false } }
  static getDerivedStateFromError() { return { failed: true } }
  render() { if (!this.state.failed) return this.props.children; return <div className="app"><Navbar /><main className="flow-page"><div className="flow-empty"><h2>Something went wrong loading Flow Intelligence.</h2><p>The page encountered an unexpected display error.</p><button className="flow-run-button" onClick={() => window.location.reload()}>Retry</button></div></main></div> }
}

export default function FlowIntelligenceWithBoundary() { return <FlowErrorBoundary><FlowIntelligence /></FlowErrorBoundary> }
