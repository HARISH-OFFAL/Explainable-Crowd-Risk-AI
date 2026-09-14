import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Navbar from '../components/Navbar'
import CrowdStoryboard from '../components/CrowdStoryboard'

const API = 'http://127.0.0.1:8000'
const ZONES = ['ZONE_A', 'ZONE_B', 'ZONE_C']
const horizons = [0, 15, 30, 60]
const label = (value) => String(value || '').replaceAll('_', ' ').replace('ZONE ', 'Zone ')

function RobotAssistant({ risk, speaking }) {
  return <div className={`phase5-robot-stage phase5-risk-${String(risk || 'STABLE').toLowerCase()} ${speaking ? 'is-speaking' : ''}`} aria-label="CrowdGuard AI robot assistant"><div className="phase5-robot-aura" /><div className="phase5-robot"><div className="phase5-robot-antenna" /><div className="phase5-robot-head"><i /><i /></div><div className="phase5-robot-body"><span /><span /><b /></div><div className="phase5-robot-arm left" /><div className="phase5-robot-arm right" /></div><small>PHASE 5 · SAFETY ASSISTANT</small></div>
}

function Waveform({ speaking }) {
  return <div className={`phase5-waveform ${speaking ? 'is-speaking' : ''}`} aria-label={speaking ? 'Assistant is speaking' : 'Assistant voice idle'}>{Array.from({ length: 18 }, (_, index) => <i key={index} />)}</div>
}

function TabBar({ tab, setTab }) {
  return <div className="phase5-tabs" role="tablist" aria-label="CrowdGuard intelligence views"><button type="button" role="tab" aria-selected={tab === 'assistant'} className={tab === 'assistant' ? 'active' : ''} onClick={() => setTab('assistant')}>AI ASSISTANT</button><button type="button" role="tab" aria-selected={tab === 'storyboard'} className={tab === 'storyboard' ? 'active' : ''} onClick={() => setTab('storyboard')}>CROWD STORYBOARD</button></div>
}

function ConversationalPanel({ data }) {
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState([])
  const [thinking, setThinking] = useState(false)
  const [listening, setListening] = useState(false)
  const recognitionRef = useRef(null)
  const lineage = data.source_lineage
  const speak = (text, language) => {
    if (!window.speechSynthesis || !text) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = language === 'ta' ? 'ta-IN' : 'en-US'
    window.speechSynthesis.speak(utterance)
  }
  const send = async (value) => {
    const message = (value ?? draft).trim()
    if (!message || thinking) return
    setMessages((old) => [...old, { role: 'user', content: message }]); setDraft(''); setThinking(true)
    try {
      const response = await fetch(API + '/events/' + data.event_id + '/assistant/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...lineage, message, conversation: messages.slice(-10), language: /[தமிழ்]/.test(message) ? 'ta' : 'auto' }) })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail || 'Assistant is unavailable.')
      setMessages((old) => [...old, { role: 'assistant', ...result }])
    } catch (reason) { setMessages((old) => [...old, { role: 'assistant', content: reason.message, error: true }]) } finally { setThinking(false) }
  }
  const listen = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) return
    const recognition = new SpeechRecognition(); recognition.lang = 'en-US'; recognition.onstart = () => setListening(true); recognition.onend = () => setListening(false); recognition.onresult = (event) => { const text = event.results[0][0].transcript; setDraft(text); send(text) }; recognitionRef.current = recognition; recognition.start()
  }
  useEffect(() => () => { recognitionRef.current?.stop(); window.speechSynthesis?.cancel() }, [])
  return <section className="phase5-chat" aria-label="CrowdGuard conversational assistant"><div className="phase5-chat-heading"><div><span>CROWDGUARD ASSISTANT</span><h2>Ask anything about this event</h2></div><button type="button" onClick={() => setMessages([])}>NEW CHAT</button></div><div className="phase5-chat-messages">{!messages.length && <div className="phase5-chat-welcome"><strong>Live analytics are the source of truth.</strong><p>Try “How many people are there now?”, “Why is Zone C risky?”, or “What should security do?”</p></div>}{messages.map((item, index) => <div className={'phase5-chat-message ' + item.role} key={item.role + index}><span>{item.role === 'user' ? 'YOU' : item.mode === 'GENERAL_AI' ? 'GENERAL AI RESPONSE' : 'CROWDGUARD'}</span><p>{item.answer || item.content}</p>{item.evidence && <small>Evidence: Monitoring #{item.evidence.monitoring_session_id || lineage.monitoring_session_id}</small>}{item.role === 'assistant' && !item.error && <button type="button" onClick={() => speak(item.answer, item.language)}>SPEAK ANSWER</button>}</div>)}</div><div className="phase5-chat-suggestions">{['What is happening now?', 'What happens next?', 'What should security do?', 'Explain compression simply'].map((item) => <button type="button" key={item} onClick={() => send(item)}>{item}</button>)}</div><form className="phase5-chat-input" onSubmit={(event) => { event.preventDefault(); send() }}><input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask CrowdGuard anything…" aria-label="Ask CrowdGuard anything" /><button type="button" onClick={listen} disabled={!((window.SpeechRecognition || window.webkitSpeechRecognition))}>{listening ? 'LISTENING…' : 'MIC'}</button><button type="submit" disabled={thinking || !draft.trim()}>{thinking ? 'THINKING…' : 'SEND'}</button></form></section>
}

function AssistantView({ data }) {
  const [speaking, setSpeaking] = useState(false)
  const [horizon, setHorizon] = useState(30)
  const recognitionRef = useRef(null)
  const forecast = useMemo(() => data?.forecast || {}, [data])
  const current = data.current
  const selected = forecast[String(horizon)] || forecast['30']
  const risk = selected?.risk_level || data.current?.risk_level || 'STABLE'
  const speakText = data?.explanation?.voice_summary || ''
  const riskZone = data.explanation.risk_zone
  const toggleSpeech = () => {
    if (!window.speechSynthesis || !speakText) return
    if (speaking) { window.speechSynthesis.cancel(); setSpeaking(false); return }
    const utterance = new SpeechSynthesisUtterance(speakText)
    const voices = window.speechSynthesis.getVoices()
    utterance.voice = voices.find((voice) => /^en(-|_)/i.test(voice.lang)) || voices[0] || null
    utterance.onstart = () => setSpeaking(true); utterance.onend = () => setSpeaking(false); utterance.onerror = () => setSpeaking(false)
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(utterance)
  }
  const ask = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) return
    const recognition = new SpeechRecognition(); recognition.lang = 'en-US'
    recognition.onresult = (event) => { const question = event.results[0][0].transcript.toLowerCase(); const answer = question.includes('30') ? `In 30 seconds, ${label(forecast['30']?.risk_zone)} is forecast as ${forecast['30']?.risk_level}.` : question.includes('60') ? `In 60 seconds, ${label(forecast['60']?.risk_zone)} is forecast as ${forecast['60']?.risk_level}.` : question.includes('recommend') ? `The recommended response is ${data.explanation.recommended_response}.` : speakText; window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance(answer)) }
    recognitionRef.current = recognition; recognition.start()
  }
  useEffect(() => () => { window.speechSynthesis?.cancel(); recognitionRef.current?.stop() }, [])
  return <><header className="phase5-header"><div><span>PHASE 5 / PREDICTIVE VOICE INTELLIGENCE</span><h1>Explainable AI Safety Assistant</h1><p>{data.selected_event?.event_name || data.event_name} · Evidence-led crowd safety guidance</p></div><b className={`phase5-status phase5-status-${risk.toLowerCase()}`}>{risk}</b></header><section className="phase5-hero"><RobotAssistant risk={risk} speaking={speaking} /><article className="phase5-assistant"><div className="phase5-kicker">CROWDGUARD AI SAFETY ASSISTANT <strong>{speaking ? 'SPEAKING' : 'READY'}</strong></div><h2>{label(riskZone)} requires attention.</h2><p>{data.explanation.summary}</p><Waveform speaking={speaking} /><div className="phase5-actions"><button className="phase5-primary" onClick={toggleSpeech}>{speaking ? 'STOP SPEAKING' : 'HEAR AI EXPLANATION'}</button><button className="phase5-secondary" onClick={ask} disabled={!((window.SpeechRecognition || window.webkitSpeechRecognition))}>ASK CROWDGUARD</button></div><small className="phase5-hint">Voice playback uses your browser's built-in speech synthesis.</small></article></section><section className="phase5-timeline"><div className="phase5-section-heading"><span>RISK FORECAST TIMELINE</span><h2>What is likely to happen next?</h2></div><div className="phase5-timeline-grid">{horizons.map((value, index) => { const item = value === 0 ? { risk_level: current.risk_level, instability_score: current.instability_score, risk_zone: current.highest_zone } : forecast[String(value)]; return <button type="button" className={`phase5-forecast-card phase5-status-${String(item?.risk_level || 'STABLE').toLowerCase()} ${horizon === value ? 'selected' : ''}`} key={value} onClick={() => setHorizon(value)}><span>{value === 0 ? 'NOW' : `+${value} SEC`}</span><strong>{item?.risk_level || 'UNAVAILABLE'}</strong><b>{item?.instability_score ?? '—'} <small>/ 100</small></b><em>{label(item?.risk_zone)}</em>{index < horizons.length - 1 && <i className="phase5-timeline-link" />}</button> })}</div></section><section className="phase5-evidence"><div className="phase5-section-heading"><span>EXPLAINABLE RISK INTELLIGENCE</span><h2>Why is the AI predicting this?</h2></div><div className="phase5-driver-list">{data.explanation.drivers.map((driver) => <article key={driver.name}><div><strong>{driver.name}</strong><b>{driver.score}%</b></div><div className="phase5-driver-track"><i style={{ width: `${driver.score}%` }} /></div><p>{driver.evidence}</p></article>)}</div><div className="phase5-summary-grid"><article><span>HIGHEST RISK ZONE</span><strong>{label(riskZone)}</strong></article><article><span>CURRENT PEOPLE</span><strong>{current.total_people}</strong></article><article><span>PREDICTION CONFIDENCE</span><strong>{data.explanation.confidence == null ? 'Unavailable' : `${data.explanation.confidence}%`}</strong></article><article><span>RECOMMENDED RESPONSE</span><strong>{data.explanation.recommended_response}</strong></article></div></section><section className="phase5-report"><span>AI SITUATION REPORT</span><p>{data.explanation.voice_summary}</p><small>Lineage: Monitoring #{data.source_lineage.monitoring_session_id} · Flow #{data.source_lineage.flow_analysis_id} · Time Machine #{data.source_lineage.time_machine_session_id}</small></section></>
}

function ExplainableAIPage() {
  const { eventId } = useParams(); const navigate = useNavigate(); const [data, setData] = useState(null); const [error, setError] = useState(''); const [loading, setLoading] = useState(true); const [tab, setTab] = useState('assistant')
  useEffect(() => { const controller = new AbortController(); async function load() { try { const [eventResponse, monitoringResponse, timeResponse] = await Promise.all([fetch(`${API}/events/${eventId}`, { signal: controller.signal }), fetch(`${API}/events/${eventId}/monitoring-sessions`, { signal: controller.signal }), fetch(`${API}/events/${eventId}/time-machine/sessions`, { signal: controller.signal })]); const event = await eventResponse.json(); const monitoring = await monitoringResponse.json(); const times = await timeResponse.json(); const monitor = (monitoring.sessions || [])[0]; if (!monitor) throw new Error('Start Monitoring to enable the safety assistant.'); const flowResponse = await fetch(`${API}/monitoring-sessions/${monitor.id}/flow-analysis`, { signal: controller.signal }); const flow = await flowResponse.json(); if (!flow.flow_analysis_id || flow.status !== 'COMPLETE') throw new Error('Complete Flow Intelligence to enable movement explanation.'); const time = (times.sessions || []).find((item) => item.status === 'COMPLETED' && item.video_name === monitor.source_name) || (times.sessions || []).find((item) => item.status === 'COMPLETED'); if (!time) throw new Error('Complete Crowd Time Machine to enable forecast explanation.'); const response = await fetch(`${API}/events/${eventId}/explainable-ai?monitoring_session_id=${monitor.id}&flow_analysis_id=${flow.flow_analysis_id}&time_machine_session_id=${time.id}&horizon=30`, { signal: controller.signal }); const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'Explainable AI is unavailable.'); if (!controller.signal.aborted) setData({ ...result, selected_event: event }); } catch (reason) { if (!controller.signal.aborted && reason.name !== 'AbortError') setError(reason.message) } finally { if (!controller.signal.aborted) setLoading(false) } } load(); return () => controller.abort() }, [eventId])
  if (loading) return <div className="app"><Navbar /><main className="phase5-page"><div className="phase5-loading" role="status">Loading Explainable AI Safety Assistant...</div></main></div>
  if (error) return <div className="app"><Navbar /><main className="phase5-page"><TabBar tab={tab} setTab={setTab} />{tab === 'storyboard' ? <><header className="phase5-header"><div><span>PHASE 6 / EVIDENCE-LED EVENT STORY</span><h1>AI Crowd Storyboard</h1><p>Generate a synchronized story directly from a monitoring session.</p></div><b className="phase5-status phase5-status-stable">READY</b></header><CrowdStoryboard eventId={eventId} /></> : <div className="phase5-empty"><h1>Explainable AI Safety Assistant</h1><p>{error}</p><button onClick={() => navigate(`/organizer/events/${eventId}/monitoring`)}>Back to Monitoring</button></div>}</main></div>
  if (!data) return null
  return <div className="app"><Navbar /><main className="phase5-page"><TabBar tab={tab} setTab={setTab} />{tab === 'storyboard' ? <><header className="phase5-header"><div><span>PHASE 6 / EVIDENCE-LED EVENT STORY</span><h1>AI Crowd Storyboard</h1><p>{data.selected_event?.event_name || data.event_name} · Dynamic monitoring moments</p></div><b className="phase5-status phase5-status-stable">READY</b></header><CrowdStoryboard eventId={eventId} /></> : <><AssistantView data={data} /><ConversationalPanel data={data} /></>}</main></div>
}

export default ExplainableAIPage
