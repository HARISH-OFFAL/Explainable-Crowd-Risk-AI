import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Navbar from '../components/Navbar'

const API_URL = 'http://127.0.0.1:8000'
const RECORDED_VIDEOS = ['mall.mp4', 'new_test.mp4', 'test.mp4', 'walk.mp4']
const EMPTY_ANALYTICS = {
  status: 'IDLE', total_people: 0, focus_zone: null, history_windows: 0,
  current_risk: { level: 'COLLECTING', risk_score: null, explanation: [] },
  forecast: { status: 'COLLECTING', probability: null },
  zones: Object.fromEntries(['ZONE_A', 'ZONE_B', 'ZONE_C'].map((zone) => [zone, { people: 0, flow: 'NO DATA', motion_percent: 0, level: 'COLLECTING', risk_score: null }])),
}

function riskClass(level) {
  return String(level || 'collecting').toLowerCase().replaceAll(' ', '-')
}

async function fetchWithRetry(url, options = {}, attempts = 3) {
  let lastError
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await fetch(url, options)
    } catch (requestError) {
      lastError = requestError
      await new Promise((resolve) => setTimeout(resolve, 700))
    }
  }
  throw lastError || new Error('Backend is unavailable.')
}

function MonitoringLaunch() {
  const navigate = useNavigate()
  const { eventId } = useParams()
  const videoRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const latestTracksRef = useRef([])
  const latestAnalyticsRef = useRef(EMPTY_ANALYTICS)
  const captureRef = useRef(null)
  const liveStreamRef = useRef(null)
  const liveBusyRef = useRef(false)
  const [event, setEvent] = useState(null)
  const [mode, setMode] = useState('demo')
  const [sourceName, setSourceName] = useState('mall.mp4')
  const [session, setSession] = useState(null)
  const sessionRef = useRef(null)
  const [analytics, setAnalytics] = useState(EMPTY_ANALYTICS)
  const [annotatedFrame, setAnnotatedFrame] = useState(null)
  const [status, setStatus] = useState('IDLE')
  const [loading, setLoading] = useState(true)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_URL}/events/${eventId}`)
      .then(async (response) => {
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || 'Unable to load event.')
        setEvent(data)
      })
      .catch((requestError) => setError(requestError.message))
      .finally(() => setLoading(false))
  }, [eventId])

  useEffect(() => {
    sessionRef.current = session
  }, [session])

  const stopMonitoring = useCallback(async () => {
    if (captureRef.current) {
      clearTimeout(captureRef.current)
      cancelAnimationFrame(captureRef.current)
    }
    if (liveStreamRef.current) liveStreamRef.current.getTracks().forEach((track) => track.stop())
    liveStreamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    const activeSession = sessionRef.current
    if (activeSession) {
      await fetch(`${API_URL}/monitoring-sessions/${activeSession.id}/stop`, { method: 'POST' }).catch(() => {})
    }
    sessionRef.current = null
    setStatus('STOPPED')
    setSession(null)
  }, [])

  useEffect(() => () => { stopMonitoring() }, [stopMonitoring])

  const sendLiveFrame = useCallback(async () => {
    const activeSession = sessionRef.current || session
    if (!videoRef.current || !activeSession || !liveStreamRef.current || liveBusyRef.current) return
    if (videoRef.current.readyState < 2 || !videoRef.current.videoWidth) {
      captureRef.current = setTimeout(sendLiveFrame, 180)
      return
    }
    liveBusyRef.current = true
    const sourceTimestamp = performance.now() / 1000
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth || 640
    canvas.height = videoRef.current.videoHeight || 360
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    try {
      const response = await fetch(`${API_URL}/monitoring-sessions/${activeSession.id}/camera-frame`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: canvas.toDataURL('image/jpeg', 0.65), source_timestamp_sec: sourceTimestamp }),
      })
      const data = await response.json()
      if (response.ok) {
        setAnalytics(data)
        latestAnalyticsRef.current = data
        latestTracksRef.current = data.tracks || []
        if (data.error) setError(data.error)
      }
    } catch { /* keep the camera view alive if one inference request fails */ }
    liveBusyRef.current = false
    captureRef.current = setTimeout(sendLiveFrame, 180)
  }, [session])

  useEffect(() => {
    if (!session || mode !== 'live' || !liveStreamRef.current || !videoRef.current) return undefined
    const video = videoRef.current
    video.srcObject = liveStreamRef.current
    video.muted = true
    video.play().catch(() => setError('Camera stream is available, but browser playback was blocked. Click Start Camera again.'))
    const timer = setTimeout(() => sendLiveFrame(), 120)
    return () => clearTimeout(timer)
  }, [session, mode, sendLiveFrame])

  useEffect(() => {
    let animationFrame
    const drawOverlay = () => {
      const canvas = overlayCanvasRef.current
      const video = videoRef.current
      if (canvas && video && session) {
        const rect = canvas.getBoundingClientRect()
        const dpr = window.devicePixelRatio || 1
        const width = Math.max(1, Math.round(rect.width))
        const height = Math.max(1, Math.round(rect.height))
        if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
          canvas.width = width * dpr
          canvas.height = height * dpr
        }
        const context = canvas.getContext('2d')
        context.setTransform(dpr, 0, 0, dpr, 0, 0)
        context.clearRect(0, 0, width, height)
        const source = latestAnalyticsRef.current.decoded_frame || {}
        const sourceWidth = source.width || video.videoWidth || 1
        const sourceHeight = source.height || video.videoHeight || 1
        const scale = Math.min(width / sourceWidth, height / sourceHeight)
        const displayedWidth = sourceWidth * scale
        const displayedHeight = sourceHeight * scale
        const displayedX = (width - displayedWidth) / 2
        const displayedY = (height - displayedHeight) / 2

        context.strokeStyle = 'rgba(96, 165, 250, .78)'
        context.lineWidth = 1
        for (const ratio of [1 / 3, 2 / 3]) {
          const x = displayedX + sourceWidth * ratio * scale
          context.beginPath(); context.moveTo(x, displayedY); context.lineTo(x, displayedY + displayedHeight); context.stroke()
        }
        context.font = '700 12px Segoe UI, sans-serif'
        context.fillStyle = '#dbeafe'
        context.fillText('ZONE A', displayedX + 10, displayedY + 18)
        context.fillText('ZONE B', displayedX + displayedWidth / 2 - 20, displayedY + 18)
        context.fillText('ZONE C', displayedX + displayedWidth - 55, displayedY + 18)

        const tracks = latestTracksRef.current
        tracks.forEach((track) => {
          const [x1, y1, x2, y2] = track.bbox || []
          if (![x1, y1, x2, y2].every(Number.isFinite)) return
          const x = displayedX + x1 * scale
          const y = displayedY + y1 * scale
          const boxWidth = (x2 - x1) * scale
          const boxHeight = (y2 - y1) * scale
          context.strokeStyle = track.motion_valid ? '#22c55e' : '#fbbf24'
          context.lineWidth = 2
          context.strokeRect(x, y, boxWidth, boxHeight)
          context.fillStyle = 'rgba(15, 23, 42, .82)'
          const label = `ID ${track.track_id ?? '--'} · ${track.zone?.replace('_', ' ') || ''}`
          const labelWidth = context.measureText(label).width + 8
          context.fillRect(x, Math.max(displayedY, y - 18), labelWidth, 18)
          context.fillStyle = '#f8fafc'
          context.fillText(label, x + 4, Math.max(displayedY + 13, y - 5))
        })
        context.fillStyle = 'rgba(15, 23, 42, .88)'
        context.fillRect(displayedX + 8, displayedY + 28, 122, 24)
        context.fillStyle = '#f8fafc'
        context.fillText(`PEOPLE: ${latestAnalyticsRef.current.frame_people_count ?? tracks.length}`, displayedX + 16, displayedY + 45)
      }
      animationFrame = requestAnimationFrame(drawOverlay)
    }
    animationFrame = requestAnimationFrame(drawOverlay)
    return () => cancelAnimationFrame(animationFrame)
  }, [session])

  const startMonitoring = async () => {
    setStarting(true)
    setError('')
    setAnnotatedFrame(null)
    let pendingStream = null
    try {
      if (mode === 'live') {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is unavailable in this browser.')
        pendingStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 15, max: 30 } }, audio: false })
      }
      const sessionResponse = await fetchWithRetry(`${API_URL}/events/${eventId}/monitoring-sessions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_type: mode === 'live' ? 'live' : 'recorded', source_name: mode === 'live' ? 'browser-camera' : sourceName }),
      })
      const sessionData = await sessionResponse.json()
      if (!sessionResponse.ok) throw new Error(sessionData.detail || 'Unable to create monitoring session.')
      const startResponse = await fetchWithRetry(`${API_URL}/monitoring-sessions/${sessionData.session.id}/web-start`, { method: 'POST' })
      const startData = await startResponse.json()
      if (!startResponse.ok) throw new Error(startData.detail || 'Unable to start AI monitoring.')
      sessionRef.current = startData.session
      setSession(startData.session)
      setStatus(mode === 'live' ? 'CAMERA_STARTING' : 'LOADING')
      if (mode === 'live') {
        liveStreamRef.current = pendingStream
        setStatus('ANALYZING')
      } else {
        setStatus('ANALYZING')
      }
    } catch (requestError) {
      if (pendingStream) pendingStream.getTracks().forEach((track) => track.stop())
      setError(requestError.name === 'NotAllowedError' ? 'Camera permission was denied.' : requestError.message === 'Failed to fetch' ? 'Backend is not reachable. Start FastAPI on port 8000 and try again.' : requestError.message)
    } finally {
      setStarting(false)
    }
  }

  useEffect(() => {
    if (!session || mode === 'live') return undefined
    const timer = setInterval(async () => {
      const response = await fetch(`${API_URL}/monitoring-sessions/${session.id}/analytics`).catch(() => null)
      if (!response) return
      const data = await response.json()
      setAnalytics(data)
      latestAnalyticsRef.current = data
      latestTracksRef.current = data.tracks || []
      setStatus(data.status)
      if (data.error) setError(data.error)
    }, 700)
    return () => clearInterval(timer)
  }, [session, mode])

  if (loading) return <div className="app"><Navbar /><main className="empty-state"><div className="surface-card"><h2>Loading monitoring context...</h2></div></main></div>
  if (!event) return <div className="app"><Navbar /><main className="empty-state"><div className="surface-card"><h2>Monitoring unavailable</h2><p>{error}</p></div></main></div>

  const preRisk = event.pre_event_risk || {}
  const current = analytics.current_risk || EMPTY_ANALYTICS.current_risk
  const zones = analytics.zones || EMPTY_ANALYTICS.zones
  const forecast = analytics.forecast || EMPTY_ANALYTICS.forecast
  const recordedVideoSource = mode === 'demo' && session ? `${API_URL}/monitoring-videos/${encodeURIComponent(sourceName)}` : null

  return (
    <div className="app app-light monitoring-page">
      <Navbar />
      <main className="monitoring-shell">
        <header className="monitoring-heading">
          <div><span className="eyebrow">CROWD MONITORING / EVENT #{event.id}</span><h1>Explainable <span>AI Monitor</span></h1><p>{event.event_name} · {event.location}</p></div>
          <div className={`analysis-status status-${riskClass(status)}`}><i /> {String(status).replaceAll('_', ' ')}</div>
        </header>

        <section className="monitor-mode surface-card">
          <div><strong>Monitor Crowd</strong><span>Choose the source for the same Phase 2 AI pipeline.</span></div>
          <div className="mode-switch">
            <button className={mode === 'live' ? 'active' : ''} onClick={() => !session && setMode('live')}>LIVE CAMERA</button>
            <button className={mode === 'demo' ? 'active' : ''} onClick={() => !session && setMode('demo')}>DEMO / RECORDED VIDEO</button>
          </div>
          {!session && mode === 'demo' && <select value={sourceName} onChange={(inputEvent) => setSourceName(inputEvent.target.value)}>{RECORDED_VIDEOS.map((video) => <option key={video}>{video}</option>)}</select>}
          {!session && mode === 'live' && <span className="mode-note">Browser camera permission will be requested when analysis starts.</span>}
          {!session ? <button className="gradient-btn" onClick={startMonitoring} disabled={starting}>{starting ? 'Initializing AI...' : mode === 'live' ? 'Start Camera' : 'Start Analysis'}</button> : <button className="outline-btn danger" onClick={stopMonitoring}>Stop Monitoring</button>}
        </section>

        {error && <div className="monitor-error">{error}</div>}

        <section className="monitor-grid">
          <aside className="monitor-side left-side">
            <article className="analytics-card"><span className="card-kicker">EVENT CONTEXT</span><h2>{event.event_name}</h2><p>{event.location}</p><p>Event ID #{event.id}</p><div className="separate-risk"><span>PRE-EVENT RISK</span><strong className={`risk-text ${riskClass(preRisk.risk_level)}`}>{preRisk.risk_level || '—'}</strong><small>Score {preRisk.risk_score ?? '—'}</small></div></article>
            <article className="analytics-card"><span className="card-kicker">LIVE CROWD SUMMARY</span><div className="metric-large">{analytics.stable_people_count ?? analytics.total_people ?? 0}<small> people detected</small></div><div className="summary-row"><span>Active tracks</span><b>{analytics.active_track_count ?? 0}</b></div><div className="summary-row"><span>Source</span><b>{mode === 'live' ? 'LIVE CAMERA' : 'DEMO VIDEO'}</b></div><div className="summary-row"><span>Progress</span><b>{analytics.progress?.total ? `${analytics.progress.frame}/${analytics.progress.total}` : status}</b></div></article>
            <article className="analytics-card"><span className="card-kicker">ZONE SNAPSHOT</span>{['ZONE_A', 'ZONE_B', 'ZONE_C'].map((zone) => <div className="zone-row" key={zone}><b>{zone.replace('_', ' ')}</b><strong>{zones[zone]?.people ?? 0}</strong><span>People: {zones[zone]?.people ?? 0} · Motion: {Math.round(zones[zone]?.motion_percent || 0)}% · Flow: {zones[zone]?.flow || 'NO DATA'}</span></div>)}</article>
          </aside>

          <section className="monitor-video surface-card">
            <div className="video-label">{session ? (mode === 'live' ? 'LIVE CAMERA' : sourceName) : 'VIDEO PREVIEW'}</div>
            {session && mode === 'live' && <video ref={videoRef} className="monitor-media live-camera-feed" muted autoPlay playsInline />}
            {session && mode === 'demo' && <video className="monitor-media demo-video" src={recordedVideoSource} autoPlay controls playsInline />}
            {session && <canvas ref={overlayCanvasRef} className="ai-overlay-canvas" aria-label="AI detection overlay" />}
            {!session && <div className="video-empty"><strong>Ready for analysis</strong><span>Select a source and start monitoring.</span></div>}
            <div className="video-caption"><span>YOLO · ByteTrack · Spatio-Temporal</span><span>{status}</span></div>
          </section>

          <aside className="monitor-side right-side">
            <article className="analytics-card spatio-card"><span className="card-kicker">SPATIO-TEMPORAL ANALYSIS</span><div className="summary-row"><span>Avg norm speed</span><b>{Number(zones[analytics.focus_zone || 'ZONE_A']?.avg_speed || 0).toFixed(3)} /s</b></div><div className="summary-row"><span>Motion trend</span><b>{Number(zones[analytics.focus_zone || 'ZONE_A']?.motion_trend || 0) >= 0 ? 'RISING' : 'FALLING'}</b></div><div className="summary-row"><span>Crowd trend</span><b>{Number(zones[analytics.focus_zone || 'ZONE_A']?.crowd_trend || 0) >= 0 ? '+' : ''}{Number(zones[analytics.focus_zone || 'ZONE_A']?.crowd_trend || 0).toFixed(1)}</b></div></article>
            <article className="analytics-card current-risk-card"><span className="card-kicker">CURRENT CROWD RISK</span><strong className={`risk-hero ${riskClass(current.level)}`}>{current.level || 'COLLECTING'}</strong><div className="score-hero">{current.risk_score == null ? '—' : `${Number(current.risk_score).toFixed(0)} / 100`}</div><div className="risk-mini-rows">{['ZONE_A', 'ZONE_B', 'ZONE_C'].map((zone) => <div key={zone}><span>{zone.replace('ZONE_', '')}</span><b className={`risk-text ${riskClass(zones[zone]?.level)}`}>{zones[zone]?.level || '—'}</b><small>{zones[zone]?.risk_score == null ? '—' : Number(zones[zone].risk_score).toFixed(0)}</small></div>)}</div></article>
            <article className="analytics-card"><span className="card-kicker">MOTION / SPATIO-TEMPORAL</span><div className="summary-row"><span>Motion activity</span><b>{current ? `${Math.round(zones[analytics.focus_zone || 'ZONE_A']?.motion_percent || 0)}%` : '—'}</b></div><div className="summary-row"><span>Dominant flow</span><b>{zones[analytics.focus_zone || 'ZONE_A']?.flow || 'NO DATA'}</b></div><div className="summary-row"><span>Direction stability</span><b>{zones[analytics.focus_zone || 'ZONE_A']?.direction_consistency == null ? '—' : `${Math.round(zones[analytics.focus_zone || 'ZONE_A'].direction_consistency * 100)}%`}</b></div></article>
            <article className="analytics-card"><span className="card-kicker">EXPERIMENTAL FUTURE FORECAST</span><p className="muted-small">Next 6 seconds · Research baseline</p><strong className="forecast-value">{forecast.probability == null ? `Collecting history ${analytics.history_windows || 0}/3` : `${(forecast.probability * 100).toFixed(0)}% evidence`}</strong></article>
            <article className="analytics-card xai-card"><span className="card-kicker">XAI · TOP FACTORS</span><p className="focus-label">Focus: {analytics.focus_zone || 'waiting'}</p>{current.explanation?.length ? <ol>{current.explanation.slice(0, 3).map((reason, index) => <li key={index}>{reason}</li>)}</ol> : <p className="muted-small">Waiting for first risk update...</p>}</article>
          </aside>
        </section>
      </main>
      <footer>CrowdRisk AI · Browser-based explainable monitoring</footer>
    </div>
  )
}

export default MonitoringLaunch
