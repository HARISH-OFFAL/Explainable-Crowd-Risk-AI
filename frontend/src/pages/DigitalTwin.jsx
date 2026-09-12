import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Navbar from '../components/Navbar'

const API_URL = 'http://127.0.0.1:8000'
const ZONES = ['ZONE_A', 'ZONE_B', 'ZONE_C']
const label = (zone) => zone.replace('ZONE_', 'Zone ')

function DigitalTwin() {
  const { eventId, sessionId } = useParams()
  const navigate = useNavigate()
  const [event, setEvent] = useState(null)
  const [state, setState] = useState(null)
  const [simulation, setSimulation] = useState(null)
  const [action, setAction] = useState('NO_ACTION')
  const [horizon, setHorizon] = useState(30)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/events/${eventId}`).then((response) => response.json()),
      fetch(`${API_URL}/monitoring-sessions/${sessionId}/digital-twin/state`).then(async (response) => {
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || 'Digital Twin state unavailable.')
        return data
      }),
    ]).then(([eventData, stateData]) => {
      setEvent(eventData)
      setState(stateData)
    }).catch((reason) => setError(reason.message)).finally(() => setLoading(false))
  }, [eventId, sessionId])

  const runSimulation = async () => {
    setRunning(true)
    setError('')
    try {
      const selectedAction = { type: action }
      if (action === 'RESTRICT_ENTRY') selectedAction.restrict_percent = 25
      if (action === 'INCREASE_EXIT') selectedAction.additional_capacity = 12
      if (action === 'REDIRECT') Object.assign(selectedAction, { redirect_percent: 20, from_zone: 'ZONE_B', to_zone: 'ZONE_C' })
      const response = await fetch(`${API_URL}/monitoring-sessions/${sessionId}/digital-twin/simulate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ horizon_seconds: Number(horizon), action: selectedAction }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Simulation failed.')
      setSimulation(data)
    } catch (reason) {
      setError(reason.message)
    } finally {
      setRunning(false)
    }
  }

  if (loading) return <div className="app digital-twin-app"><Navbar /><main className="digital-twin-page"><div className="flow-empty">Loading exact session snapshot...</div></main></div>

  const currentZones = state?.current_state?.zones || {}
  const provenance = state?.source_provenance || state?.current_state?.source_provenance || {}

  return <div className="app digital-twin-app"><Navbar /><main className="digital-twin-page">
    <aside className="flow-sidebar"><span className="control-kicker">Phase 3 / Intelligence</span><h2>{event?.event_name || 'Event'}</h2><button onClick={() => navigate(`/organizer/events/${eventId}/flow-intelligence/${state?.flow_analysis_id || sessionId}`)}>Flow Intelligence</button><button className="active">Digital Twin Simulation</button></aside>
    <section className="flow-content"><header className="flow-header"><div><span className="control-kicker">Phase 3B / Decision support</span><h1>Crowd Digital Twin</h1><p>{event?.event_name || ''} · Session #{sessionId}</p></div><span className="flow-status">Estimated simulation</span></header>
      {error && <div className="monitor-error">{error}</div>}
      <section className="flow-card twin-source"><span className="control-kicker">MEASURED SOURCE SNAPSHOT</span><strong>Monitoring Session #{state?.monitoring_session_id || sessionId} · {state?.source}</strong><small>Flow Analysis #{state?.flow_analysis_id || '—'} · final valid frame: {provenance.source_frame_index ?? '—'} · video time: {provenance.source_timestamp_sec ?? '—'}s · total detected: {state?.current_state?.total_people ?? '—'}</small></section>
      <section className="twin-layout"><article className="flow-card twin-map"><div className="flow-panel-heading"><div><span className="control-kicker">CURRENT CROWD STATE</span><h2>Zone network</h2></div><small>Measured values — simulation never overwrites this view</small></div><svg viewBox="0 0 720 300" role="img" aria-label="Measured current crowd state zone network"><defs><marker id="twin-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6" fill="none" stroke="#67e8f9" /></marker></defs><path className="twin-connection" d="M120 150 H300 M360 150 H540" markerEnd="url(#twin-arrow)" /><path className="twin-connection" d="M300 185 V245 H620" markerEnd="url(#twin-arrow)" /><text x="45" y="72">ENTRY</text><text x="610" y="238">EXIT 2</text>{ZONES.map((zone, index) => { const metrics = currentZones[zone] || {}; const x = 55 + index * 240; const stateName = metrics.flow_state || 'Balanced'; return <g key={zone} className={`twin-zone twin-${String(stateName).toLowerCase()}`}><rect x={x} y="105" width="150" height="90" rx="10" /><text x={x + 18} y="132">{label(zone).toUpperCase()}</text><text className="twin-count" x={x + 18} y="164">{Math.round(metrics.current_people || 0)}</text><text x={x + 72} y="164">people</text><text className="twin-state" x={x + 18} y="184">{stateName}</text></g>})}</svg></article>
        <aside className="flow-card twin-controls"><span className="control-kicker">WHAT-IF CONTROLS</span><h2>Run an estimated scenario</h2><label>Action<select value={action} onChange={(inputEvent) => setAction(inputEvent.target.value)}><option value="NO_ACTION">No Action</option><option value="RESTRICT_ENTRY">Restrict Entry 25%</option><option value="INCREASE_EXIT">Increase Exit Capacity</option><option value="REDIRECT">Redirect 20% Zone B → Zone C</option></select></label><label>Simulation horizon<select value={horizon} onChange={(inputEvent) => setHorizon(inputEvent.target.value)}><option value="15">15 seconds</option><option value="30">30 seconds</option><option value="60">60 seconds</option></select></label><button className="flow-run-button" onClick={runSimulation} disabled={running}>{running ? 'Simulating...' : 'Run Simulation'}</button><small>Decision-support estimate only. The measured current state above remains unchanged.</small></aside>
      </section>
      {simulation && <><section className="flow-metrics twin-metrics"><article><span>SIMULATED AFTER {horizon}s</span><strong>Scenario result</strong></article>{ZONES.map((zone) => <article key={zone}><span>{label(zone).toUpperCase()}</span><strong>{Math.round(simulation.selected.zones[zone].people)} <small>people</small></strong><b>{Math.round(simulation.selected.zones[zone].density_index)} density index</b></article>)}</section><section className="twin-bottom"><article className="flow-card"><div className="flow-panel-heading"><div><span className="control-kicker">BEFORE VS SIMULATED AFTER</span><h2>{simulation.selected.action.label}</h2></div><small>{simulation.simulation_label}</small></div>{ZONES.map((zone) => { const before = simulation.baseline.zones[zone].people; const after = simulation.selected.zones[zone].people; return <div className="before-after-row" key={zone}><span>{label(zone)}</span><b>{Math.round(before)}</b><i>→</i><strong>{Math.round(after)}</strong><small>{after - before >= 0 ? '+' : ''}{Math.round(after - before)} people</small></div> })}</article><article className="flow-card"><div className="flow-panel-heading"><div><span className="control-kicker">RECOMMENDATION</span><h2>Best simulated action</h2></div></div><div className="twin-recommendation"><strong>{simulation.best?.action?.label || 'No improvement identified'}</strong><span>Action score: {simulation.best?.score ?? 0}</span><p>Estimated effect only; it does not change real event settings.</p></div></article></section></>}
    </section></main></div>
}

export default DigitalTwin
