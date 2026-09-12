import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

const API_URL = 'http://127.0.0.1:8000'
const RECORDED_VIDEOS = ['gem.mp4', 'mall.mp4', 'new_test.mp4', 'test.mp4', 'walk.mp4']
const ZONES = ['ZONE_A', 'ZONE_B', 'ZONE_C']
const EMPTY = { status: 'IDLE', processed_frames: 0, people_count: 0, density_state: 'LOW', density_value: 0, zone_counts: { ZONE_A: 0, ZONE_B: 0, ZONE_C: 0 }, trend: [], tracks: [], heatmap: { grid: [], rows: 24, columns: 32 } }
const label = (zone) => zone.replace('ZONE_', 'Zone ')

function trendPath(points) {
  if (!points.length) return ''
  const coords = points.map((point, index) => [(points.length === 1 ? 300 : (index / (points.length - 1)) * 580 + 10), 210 - (Math.max(0, Math.min(100, Number(point.density_value) || 0)) / 100) * 190])
  return coords.reduce((path, point, index) => {
    if (!index) return `M ${point[0]} ${point[1]}`
    const previous = coords[index - 1]; const midpoint = (previous[0] + point[0]) / 2
    return `${path} C ${midpoint} ${previous[1]}, ${midpoint} ${point[1]}, ${point[0]} ${point[1]}`
  }, '')
}

function ZoneDonut({ counts }) {
  const total = ZONES.reduce((sum, zone) => sum + (Number(counts[zone]) || 0), 0)
  let offset = 0
  const colors = ['#39d98a', '#54b7ff', '#ffc857']
  const segments = ZONES.map((zone, index) => { const percentage = total ? ((Number(counts[zone]) || 0) / total) * 100 : 0; const segment = `${colors[index]} ${offset}% ${offset + percentage}%`; offset += percentage; return { zone, percentage, color: colors[index], segment } })
  return <div className="zone-donut-layout"><div className={`zone-donut ${total ? '' : 'is-empty'}`} style={{ background: total ? `conic-gradient(${segments.map((item) => item.segment).join(', ')})` : 'conic-gradient(#334155 0 100%)' }}><div><strong>{total}</strong><span>People</span></div></div><div className="zone-legend">{segments.map((item) => <div key={item.zone}><i style={{ background: item.color }} /><span>{label(item.zone)}</span><b>{counts[item.zone] || 0}</b><small>{Math.round(item.percentage)}%</small></div>)}</div></div>
}

function FinalZoneOverlay({ analytics }) {
  if (analytics.video_status !== 'COMPLETE') return null
  const counts = analytics.zone_counts || EMPTY.zone_counts
  const levels = analytics.zone_crowd_levels || {}
  return <div className="final-zone-overlay" aria-label="Final zone crowd levels">{ZONES.map((zone) => <div className={`final-zone-card level-${String(levels[zone] || 'LOW').toLowerCase()}`} key={zone}><strong>{label(zone)}</strong><b>{counts[zone] || 0} <small>PEOPLE</small></b><span>{levels[zone] || 'LOW'}</span></div>)}</div>
}

function MonitoringLaunch() {
  const { eventId } = useParams(); const routerNavigate = useNavigate(); function navigate(path) { if (path.includes('/flow-intelligence/')) { openFlowIntelligence(); return } routerNavigate(path) }
  const videoRef = useRef(null); const canvasRef = useRef(null); const captureCanvasRef = useRef(null); const sessionRef = useRef(null)
  const cameraStreamRef = useRef(null); const cameraTimerRef = useRef(null); const cameraStartedAtRef = useRef(0); const sendingFrameRef = useRef(false)
  const [event, setEvent] = useState(null); const [sourceName, setSourceName] = useState('gem.mp4'); const [sourceMode, setSourceMode] = useState('recorded')
  const [session, setSession] = useState(null); const [analytics, setAnalytics] = useState(EMPTY); const [loading, setLoading] = useState(true); const [starting, setStarting] = useState(false); const [error, setError] = useState(''); const [uploading, setUploading] = useState(false); const [showHeatmap, setShowHeatmap] = useState(true); const [showZones, setShowZones] = useState(true)

  useEffect(() => { fetch(`${API_URL}/events/${eventId}`).then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Unable to load event.'); setEvent(data) }).catch((requestError) => setError(requestError.message)).finally(() => setLoading(false)) }, [eventId])

  const stopCamera = useCallback(() => { if (cameraTimerRef.current) window.clearInterval(cameraTimerRef.current); cameraTimerRef.current = null; if (cameraStreamRef.current) cameraStreamRef.current.getTracks().forEach((track) => track.stop()); cameraStreamRef.current = null; if (videoRef.current) videoRef.current.srcObject = null; sendingFrameRef.current = false }, [])
  const stopMonitoring = useCallback(async () => {
    // Detach the session before awaiting the API call so polling effects stop
    // immediately and cannot request analytics from an engine being removed.
    const activeSession = sessionRef.current
    sessionRef.current = null
    stopCamera()
    setSession(null)
    setAnalytics(EMPTY)
    if (activeSession) await fetch(`${API_URL}/monitoring-sessions/${activeSession.id}/stop`, { method: 'POST' }).catch(() => {})
  }, [stopCamera])
  const openFlowIntelligence = async () => { if (!session?.id) return; setError(''); try { const response = await fetch(`${API_URL}/monitoring-sessions/${session.id}/flow-analysis`, { method: 'POST' }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Unable to prepare Flow Intelligence.'); if (!data.flow_analysis_id) throw new Error('The backend did not return a Flow Analysis ID.'); routerNavigate(`/organizer/events/${eventId}/flow-intelligence/${data.flow_analysis_id}`) } catch (reason) { setError(reason.message) } }
  useEffect(() => () => { stopMonitoring() }, [stopMonitoring])

  const startMonitoring = async () => {
    setStarting(true); setError(''); let stream = null
    try {
      if (sourceMode === 'live') { if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not support camera access.'); stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false }) }
      const create = await fetch(`${API_URL}/events/${eventId}/monitoring-sessions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source_type: sourceMode === 'live' ? 'live' : 'recorded', source_name: sourceMode === 'live' ? 'browser-camera' : sourceName }) }); const created = await create.json(); if (!create.ok) throw new Error(created.detail || 'Unable to create monitoring session.')
      const start = await fetch(`${API_URL}/monitoring-sessions/${created.session.id}/web-start`, { method: 'POST' }); const started = await start.json(); if (!start.ok) throw new Error(started.detail || 'Unable to start monitoring.')
      sessionRef.current = started.session; if (stream) { cameraStreamRef.current = stream; cameraStartedAtRef.current = performance.now() }; setSession(started.session)
    } catch (requestError) { if (stream) stream.getTracks().forEach((track) => track.stop()); setError(requestError.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access to use Live Monitoring.' : requestError.message) } finally { setStarting(false) }
  }

  const switchSourceMode = async (nextMode) => { if (nextMode === sourceMode) return; if (sessionRef.current) await stopMonitoring(); setSourceMode(nextMode); setAnalytics(EMPTY); setError('') }
  const uploadVideo = async (inputEvent) => { const file = inputEvent.target.files?.[0]; if (!file) return; setUploading(true); setError(''); try { const body = new FormData(); body.append('file', file); const response = await fetch(`${API_URL}/monitoring-videos/upload`, { method: 'POST', body }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Unable to upload video.'); setSourceName(data.source_name) } catch (requestError) { setError(requestError.message) } finally { setUploading(false); inputEvent.target.value = '' } }

  useEffect(() => {
    if (!session || sourceMode === 'live') return undefined
    const controller = new AbortController()
    let active = true
    const poll = async () => {
      try {
        const response = await fetch(`${API_URL}/monitoring-sessions/${session.id}/analytics`, { signal: controller.signal })
        if (!response.ok || !active) return
        const data = await response.json()
        if (active) {
          setAnalytics(data)
          if (data.error) setError(data.error)
        }
      } catch (requestError) {
        if (requestError.name !== 'AbortError' && active) setError(requestError.message)
      }
    }
    const timer = setInterval(poll, 650)
    poll()
    return () => { active = false; controller.abort(); clearInterval(timer) }
  }, [session, sourceMode])

  // Let recorded media use native browser playback. The detector runs on a
  // lower cadence, so pausing the video to chase analytics timestamps can
  // leave the player stuck on its first frame while the backend catches up.
  useEffect(() => {
    if (!session || sourceMode !== 'recorded') return undefined
    const video = videoRef.current
    if (video) video.play().catch(() => {})
    return undefined
  }, [session, sourceMode, sourceName])

  useEffect(() => {
    if (!session || sourceMode !== 'live') return undefined
    if (videoRef.current && cameraStreamRef.current) { videoRef.current.srcObject = cameraStreamRef.current; videoRef.current.play().catch(() => {}) }
    const sendCameraFrame = () => { const video = videoRef.current; const capture = captureCanvasRef.current; if (!video || !capture || !video.videoWidth || !video.videoHeight || sendingFrameRef.current) return; sendingFrameRef.current = true; capture.width = Math.min(video.videoWidth, 1280); capture.height = Math.round(capture.width * video.videoHeight / video.videoWidth); capture.getContext('2d').drawImage(video, 0, 0, capture.width, capture.height); capture.toBlob((blob) => { if (!blob) { sendingFrameRef.current = false; return }; const reader = new FileReader(); reader.onloadend = async () => { try { const response = await fetch(`${API_URL}/monitoring-sessions/${session.id}/camera-frame`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_base64: reader.result, source_timestamp_sec: (performance.now() - cameraStartedAtRef.current) / 1000 }) }); const data = await response.json(); if (response.ok) setAnalytics(data); else if (data.detail) setError(data.detail) } catch (requestError) { setError(requestError.message) } finally { sendingFrameRef.current = false } }; reader.readAsDataURL(blob) }, 'image/jpeg', 0.72) }
    cameraTimerRef.current = window.setInterval(sendCameraFrame, 150); return () => { if (cameraTimerRef.current) window.clearInterval(cameraTimerRef.current); cameraTimerRef.current = null }
  }, [session, sourceMode])

  useEffect(() => { let frame; const draw = () => { const canvas = canvasRef.current; const video = videoRef.current; if (canvas && video && analytics.decoded_frame?.width) { const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; const width = Math.max(1, rect.width); const height = Math.max(1, rect.height); canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height); const sourceWidth = analytics.decoded_frame.width; const sourceHeight = analytics.decoded_frame.height; const scale = Math.min(width / sourceWidth, height / sourceHeight); const ox = (width - sourceWidth * scale) / 2; const oy = (height - sourceHeight * scale) / 2; const grid = analytics.heatmap?.grid || []; const rows = analytics.heatmap?.rows || 24; const columns = analytics.heatmap?.columns || 32
      if (showHeatmap && grid.length) { const field = document.createElement('canvas'); field.width = columns; field.height = rows; const fieldContext = field.getContext('2d'); const image = fieldContext.createImageData(columns, rows); for (let row = 0; row < rows; row += 1) for (let col = 0; col < columns; col += 1) { const value = Math.max(0, Math.min(1, grid[row]?.[col] || 0)); const hue = Math.round(235 - value * 235); const saturation = 88; const lightness = 54; const chroma = (1 - Math.abs(2 * lightness / 100 - 1)) * saturation / 100; const x = chroma * (1 - Math.abs((hue / 60) % 2 - 1)); const match = lightness / 100 - chroma / 2; const rgb = hue < 60 ? [chroma, x, 0] : hue < 120 ? [x, chroma, 0] : hue < 180 ? [0, chroma, x] : hue < 240 ? [0, x, chroma] : hue < 300 ? [x, 0, chroma] : [chroma, 0, x]; const offset = (row * columns + col) * 4; image.data[offset] = (rgb[0] + match) * 255; image.data[offset + 1] = (rgb[1] + match) * 255; image.data[offset + 2] = (rgb[2] + match) * 255; image.data[offset + 3] = value <= 0.005 ? 0 : 45 + value * 150 }; fieldContext.putImageData(image, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(field, ox, oy, sourceWidth * scale, sourceHeight * scale) }
      if (analytics.tracks?.length) { ctx.lineWidth = 1.5; ctx.strokeStyle = '#dbeafe'; ctx.fillStyle = '#dbeafe'; ctx.font = '600 10px Inter, sans-serif'; analytics.tracks.forEach((track) => { const [x1, y1, x2, y2] = track.bbox; const bx = ox + x1 * scale; const by = oy + y1 * scale; ctx.strokeRect(bx, by, (x2 - x1) * scale, (y2 - y1) * scale); ctx.fillText(`${Math.round(track.confidence * 100)}%`, bx, Math.max(10, by - 3)) }) }
      if (showZones) { ctx.setLineDash([]); ctx.lineWidth = Math.max(2, Math.round(sourceWidth / 500)); ctx.strokeStyle = 'rgba(3, 10, 20, .95)'; [1 / 3, 2 / 3].forEach((ratio) => { ctx.beginPath(); ctx.moveTo(ox + sourceWidth * ratio * scale, oy); ctx.lineTo(ox + sourceWidth * ratio * scale, oy + sourceHeight * scale); ctx.stroke() }); ctx.font = '700 11px Inter, sans-serif'; ctx.fillStyle = '#f8fafc'; ctx.fillText('ZONE A', ox + 10, oy + 18); ctx.fillText('ZONE B', ox + sourceWidth * scale / 2 - 20, oy + 18); ctx.fillText('ZONE C', ox + sourceWidth * scale - 52, oy + 18) } } frame = requestAnimationFrame(draw) }; frame = requestAnimationFrame(draw); return () => cancelAnimationFrame(frame) }, [analytics, showHeatmap, showZones])

  if (loading) return <main className="monitoring-page ember-monitoring-page loading-panel">Loading event monitoring context...</main>
  if (!event) return <main className="monitoring-page ember-monitoring-page loading-panel"><h2>Monitoring unavailable</h2><p>{error}</p></main>
  const counts = analytics.zone_counts || EMPTY.zone_counts; const mostCrowded = analytics.most_crowded_zone ? label(analytics.most_crowded_zone) : '—'; const videoSource = `${API_URL}/monitoring-videos/${encodeURIComponent(sourceName)}`; const trend = analytics.trend || []; const currentTrend = trend.length ? Math.round(trend[trend.length - 1].density_value) : null
  return <div className="monitoring-page-dark ember-monitoring-page"><header className="control-header"><div><span className="control-kicker">CROWD INTELLIGENCE / EVENT #{event.id}</span><h1>{event.event_name}</h1><p>{event.location} · {event.event_datetime || 'Date not provided'}</p></div><div className="header-state"><i /> {session ? sourceMode === 'live' ? 'LIVE' : analytics.status || 'STARTING' : 'READY'}</div></header><main className="control-shell">
    <section className="monitor-toolbar source-toolbar"><div><strong>MONITORING SOURCE</strong><span>{sourceMode === 'live' ? 'Real-time camera analytics' : 'Recorded crowd-density analytics'}</span></div><div className="source-mode-tabs">{session && sourceMode === 'recorded' && analytics.video_status === 'COMPLETE' && <button className="flow-link flow-top-link" onClick={() => navigate(`/organizer/events/${eventId}/flow-intelligence/${session.id}`)}>View Flow Intelligence</button>}<button className={sourceMode === 'recorded' ? 'active' : ''} onClick={() => switchSourceMode('recorded')} disabled={starting}>Recorded Video</button><button className={sourceMode === 'live' ? 'active' : ''} onClick={() => switchSourceMode('live')} disabled={starting}>Live Camera</button></div>{sourceMode === 'recorded' && <><select value={sourceName} onChange={(inputEvent) => setSourceName(inputEvent.target.value)} disabled={Boolean(session)}>{RECORDED_VIDEOS.map((video) => <option key={video}>{video}</option>)}</select><label className="upload-video-control">{uploading ? 'Uploading...' : 'Upload video'}<input type="file" accept="video/*" onChange={uploadVideo} disabled={Boolean(session) || uploading} /></label></>}{!session ? <button onClick={startMonitoring} disabled={starting}>{starting ? 'Starting...' : sourceMode === 'live' ? 'Start Camera' : 'Start Monitoring'}</button> : <button className="stop-control" onClick={stopMonitoring}>{sourceMode === 'live' ? 'Stop Camera' : 'Stop Monitoring'}</button>}<button className="back-control" onClick={() => navigate(`/organizer/events/${eventId}/phase1-summary`)}>Event Details</button></section>
    {error && <div className="monitor-error">{error}</div>}<section className="control-grid"><div className="video-panel"><div className="panel-topline"><span>{sourceMode === 'live' ? 'Camera' : sourceName}</span><span>{sourceMode === 'live' && session ? '● LIVE' : analytics.video_status || 'VIDEO READY'}</span></div><div className="video-stage">{session ? <video ref={videoRef} className="control-video" src={sourceMode === 'recorded' ? videoSource : undefined} autoPlay muted playsInline controls={sourceMode === 'recorded'} onLoadedMetadata={(event) => event.currentTarget.play().catch(() => {})} onEnded={() => setAnalytics((current) => ({ ...current, video_status: 'COMPLETE' }))} /> : <div className="video-placeholder"><strong>{sourceMode === 'live' ? 'Camera ready' : 'Recorded video ready'}</strong><span>{sourceMode === 'live' ? 'Start Camera and allow browser access.' : `Select ${sourceName} and start monitoring.`}</span></div>}<canvas ref={canvasRef} className="control-overlay" /><canvas ref={captureCanvasRef} className="camera-capture-buffer" /><FinalZoneOverlay analytics={analytics} /></div><div className="panel-footer"><span>Adaptive heatmap · AI {analytics.ai_target_fps || 8} FPS</span><span>{sourceMode === 'live' ? 'Camera display: native timing' : 'Playback: native video timing'}</span></div></div><aside className="metrics-column"><article className="metric-card primary-metric"><span>PEAK DENSITY INDEX</span><strong>{analytics.processed_frames === 0 ? '—' : Math.round(analytics.peak_density_index || 0)}<small> / 100</small></strong><small>running local-density scale</small></article><article className="metric-card"><span>CROWD DENSITY</span><strong className={`density-${String(analytics.density_state || 'LOW').toLowerCase().replace(' ', '-')}`}>{analytics.processed_frames === 0 ? 'Waiting' : analytics.density_state}</strong><small>observed current state</small></article><article className="metric-card"><span>MOST CONCENTRATED ZONE</span><strong>{mostCrowded}</strong><small>dynamic zone score</small></article><article className="metric-card status-card"><div><span>CURRENT PEOPLE</span><b>{analytics.processed_frames === 0 ? '—' : analytics.people_count ?? '—'}</b></div><div><span>{sourceMode === 'live' ? 'ELAPSED TIME' : 'VIDEO TIME'}</span><b>{analytics.processed_frames === 0 ? '—' : `${Math.floor(analytics.source_timestamp_sec || 0)}s`}</b></div><div><span>SOURCE / AI FPS</span><b>{analytics.metadata?.fps ? `${analytics.metadata.fps} / ${analytics.processing_fps || '—'}` : analytics.processing_fps || '—'}</b></div></article></aside></section>
    <section className="lower-grid"><article className="chart-panel trend-panel"><div className="panel-heading"><div><span className="control-kicker">LIVE OBSERVATION</span><h2>Density trend {currentTrend !== null && <em>CURRENT {currentTrend}</em>}</h2></div><small>bounded history · {sourceMode === 'live' ? 'elapsed time' : 'video time'}</small></div><div className="telemetry-chart"><svg viewBox="0 0 600 240" preserveAspectRatio="none" role="img" aria-label="Crowd density trend"><line x1="10" x2="590" y1="20" y2="20" /><line x1="10" x2="590" y1="115" y2="115" /><line x1="10" x2="590" y1="210" y2="210" /><text x="0" y="25">100</text><text x="0" y="120">50</text><text x="0" y="215">0</text>{trendPath(trend) && <path className="trend-signal" d={trendPath(trend)} />}</svg><div className="trend-axis"><span>{trend.length ? `${Math.round(trend[0].timestamp)}s` : '0s'}</span><span>{trend.length ? `${Math.round(trend[trend.length - 1].timestamp)}s` : '—'}</span></div></div></article><article className="chart-panel zones-panel"><div className="panel-heading"><div><span className="control-kicker">SPATIAL DISTRIBUTION</span><h2>Zone distribution</h2></div><small>{analytics.people_count || 0} current people</small></div><ZoneDonut counts={counts} /></article></section>
    <section className="view-controls"><span>Overlay layers</span><label><input type="checkbox" checked={showHeatmap} onChange={(inputEvent) => setShowHeatmap(inputEvent.target.checked)} /> Heatmap</label><label><input type="checkbox" checked={showZones} onChange={(inputEvent) => setShowZones(inputEvent.target.checked)} /> Zones</label><span className="legend"><i className="blue" /> low <i className="green" /> moderate <i className="yellow" /> dense <i className="red" /> highest</span></section><div className="event-facts"><span>Expected crowd <b>{event.expected_crowd_size?.toLocaleString() || '—'}</b></span><span>Venue capacity <b>{event.venue_capacity?.toLocaleString() || '—'}</b></span><span>Session <b>{session ? `#${session.id}` : 'not started'}</b></span>{session && sourceMode === 'recorded' && analytics.video_status === 'COMPLETE' && <button className="flow-link" onClick={() => navigate(`/organizer/events/${eventId}/flow-intelligence/${session.id}`)}>View Flow Intelligence</button>}</div>
  </main></div>
}

export default MonitoringLaunch
