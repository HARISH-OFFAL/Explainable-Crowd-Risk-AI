import { useEffect, useMemo, useRef, useState } from 'react'

const API = 'http://127.0.0.1:8000'
const audiences = ['ORGANIZER', 'SECURITY', 'PUBLIC', 'EMERGENCY']
const label = (value) => String(value || '').replaceAll('_', ' ').replace('ZONE ', 'Zone ')

function useEnglishVoice() {
  const [voices, setVoices] = useState([])
  useEffect(() => {
    const load = () => setVoices(window.speechSynthesis?.getVoices?.() || [])
    load()
    const synth = window.speechSynthesis
    if (!synth) return undefined
    synth.addEventListener?.('voiceschanged', load)
    const timer = window.setInterval(load, 500)
    return () => { synth.removeEventListener?.('voiceschanged', load); window.clearInterval(timer) }
  }, [])
  return voices.find((voice) => voice.lang?.toLowerCase() === 'en-in') || voices.find((voice) => ['en-gb', 'en-us'].includes(voice.lang?.toLowerCase())) || voices.find((voice) => voice.lang?.toLowerCase().startsWith('en'))
}

export default function CommunicationCenterV2({ eventId, monitoringId, flowId, timeMachineId, responsePlanId }) {
  const [plan, setPlan] = useState(null)
  const [audience, setAudience] = useState('PUBLIC')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])
  const [recipient, setRecipient] = useState('')
  const speech = useRef(null)
  const voice = useEnglishVoice()
  const language = 'en'

  const refresh = () => fetch(`${API}/events/${eventId}/communication/history?monitoring_session_id=${monitoringId}`).then((response) => response.json()).then((data) => setHistory(data.plans || [])).catch(() => {})
  useEffect(() => { refresh(); return () => window.speechSynthesis?.cancel() }, [eventId, monitoringId])

  const current = useMemo(() => plan?.messages?.find((item) => item.audience === audience && item.language === language), [plan, audience])
  const text = current?.message || 'Generate a communication draft from the selected monitoring lineage.'
  const status = plan?.delivery_status || plan?.status || 'DRAFT'
  const approved = ['APPROVED', 'SENT'].includes(plan?.status)

  const speak = () => {
    window.speechSynthesis?.cancel()
    if (!voice) { setError('English voice is not available in this browser.'); return Promise.resolve(false) }
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = 'en-IN'; utterance.voice = voice; utterance.rate = 1
      utterance.onstart = () => resolve(true)
      utterance.onerror = () => { setError('English voice playback failed.'); resolve(false) }
      speech.current = utterance
      window.speechSynthesis.speak(utterance)
    })
  }

  const generate = async () => {
    setBusy(true); setError('')
    const response = await fetch(`${API}/events/${eventId}/communication/plans`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ monitoring_session_id: Number(monitoringId), flow_analysis_id: flowId ? Number(flowId) : undefined, time_machine_session_id: timeMachineId ? Number(timeMachineId) : undefined, response_plan_id: responsePlanId ? Number(responsePlanId) : undefined }) })
    const data = await response.json(); if (response.ok) setPlan(data); else setError(data.detail || 'Draft generation failed'); setBusy(false)
  }

  const approve = async () => {
    setBusy(true); setError('')
    const response = await fetch(`${API}/events/${eventId}/communication/plans/${plan.communication_plan_id}/approve`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ approved_by: 'organizer' }) })
    const data = await response.json(); if (response.ok) setPlan(data); else setError(data.detail || 'Approval failed'); setBusy(false)
  }

  const send = async (channel) => {
    if (!approved) return
    setBusy(true); setError('')
    if (channel === 'PA_SIMULATION' && !(await speak())) { setBusy(false); return }
    const response = await fetch(`${API}/events/${eventId}/communication/plans/${plan.communication_plan_id}/send`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ audience, language, channel, recipient, speech_started: channel === 'PA_SIMULATION' }) })
    const data = await response.json()
    if (response.ok) { setPlan(data.plan); refresh() } else { setError(data.detail || 'Delivery failed'); const latest = await fetch(`${API}/events/${eventId}/communication/plans/${plan.communication_plan_id}`).then((item) => item.json()).catch(() => null); if (latest) setPlan(latest) }
    setBusy(false)
  }

  const downloadBrief = async () => {
    if (!approved || !plan) return
    setBusy(true); setError('')
    try {
      const response = await fetch(`${API}/events/${eventId}/communication/plans/${plan.communication_plan_id}/download?audience=${encodeURIComponent(audience)}&language=en`)
      if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.detail || 'Safety brief download failed') }
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `CrowdGuard_Safety_Brief_Event_${eventId}_Plan_${plan.communication_plan_id}.pdf`; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url)
    } catch (reason) { setError(reason.message) } finally { setBusy(false) }
  }

  return <section className="communication-center">
    <div className="cc-hero"><div><span className="cc-kicker">CROWDGUARD / CONTROLLED GUIDANCE</span><h2>Communication Center</h2><p>Translate crowd intelligence into clear, controlled safety guidance.</p></div><div className="cc-guardrail">DRAFT · REVIEW · APPROVE · DELIVER</div></div>
    <div className="cc-context"><span>EVENT #{eventId}</span><span>MONITORING #{monitoringId || 'N/A'}</span><span>FLOW #{flowId || 'N/A'}</span><span>RESPONSE PLAN #{responsePlanId || 'N/A'}</span><b>{status}</b></div>
    <div className="cc-grid">
      <article className="cc-panel cc-situation"><div className="cc-panel-heading"><span>CURRENT SITUATION</span><i>{plan?.source_state?.risk || 'SOURCE LINK REQUIRED'}</i></div><div className="cc-stat-grid"><div><small>RECORDED PEOPLE</small><strong>{plan?.source_state?.current_people ?? 'N/A'}</strong></div><div><small>CONCERN ZONE</small><strong>{label(plan?.source_state?.target_zone) || 'N/A'}</strong></div><div><small>DENSITY</small><strong>{plan?.source_state?.density || 'N/A'}</strong></div><div><small>DRIVER</small><strong>{plan?.source_state?.main_driver || 'N/A'}</strong></div></div><p>{plan?.recommended_response?.explanation?.why || 'Generate a draft using the exact CrowdGuard source lineage.'}</p></article>
      <article className="cc-panel cc-plan"><div className="cc-panel-heading"><span>COMMUNICATION PLAN</span><i>{status}</i></div><div className="cc-audience-tabs">{audiences.map((item) => <button key={item} className={audience === item ? 'active' : ''} onClick={() => setAudience(item)}>{label(item)}</button>)}</div><div className="cc-language"><button className="active">English</button><small>{voice ? 'Voice available' : 'Voice unavailable'}</small></div><div className="cc-message-meta"><b>{label(audience)}</b><span>INTENT · {label(plan?.intent || 'MONITOR_ONLY')}</span><span>PRIORITY · {plan?.priority || 'N/A'}</span></div><div className="cc-message">{text}</div><div className="cc-voice"><span className="cc-wave">{Array.from({ length: 12 }, (_, i) => <i key={i} style={{ height: `${12 + (i % 5) * 5}px` }} />)}</span><button onClick={speak} disabled={!plan || !voice}>PREVIEW VOICE</button><button onClick={() => window.speechSynthesis?.cancel()}>STOP</button></div></article>
      <article className="cc-panel cc-delivery"><div className="cc-panel-heading"><span>DELIVERY & APPROVAL</span><i>{status}</i></div><p className="cc-safety-note">Approval is required before any delivery.</p><button className="cc-generate" onClick={generate} disabled={busy || !monitoringId}>{busy ? 'WORKING…' : 'GENERATE DRAFT'}</button><button className="cc-approve" onClick={approve} disabled={!plan || plan.status !== 'PENDING_APPROVAL'}>{approved ? 'APPROVED' : 'APPROVE COMMUNICATION'}</button>{approved && <><input className="cc-recipient" value={recipient} onChange={(event) => setRecipient(event.target.value)} placeholder="Recipient email" /><div className="cc-channel-buttons"><button onClick={() => send('DASHBOARD')} disabled={busy}>DASHBOARD</button><button onClick={() => send('PA_SIMULATION')} disabled={busy}>PA SIMULATION</button><button onClick={() => send('EMAIL')} disabled={busy || !recipient}>SEND EMAIL</button><button onClick={downloadBrief} disabled={busy}>DOWNLOAD PDF</button></div></>}{error && <p className="cc-error">{error}</p>}</article>
    </div>
    <div className="cc-bottom"><article className="cc-panel"><div className="cc-panel-heading"><span>DELIVERY STATE</span><i>{plan?.history_count || 0} HISTORY RECORDS</i></div><strong>{status}</strong><p>{plan?.history_count ? 'Delivery is backed by a persisted channel record.' : 'No successful delivery record exists yet.'}</p></article><article className="cc-panel"><div className="cc-panel-heading"><span>COMMUNICATION HISTORY</span><i>{history.length} PLANS</i></div>{history.slice(0, 4).map((item) => <div className="cc-history-row" key={item.communication_plan_id}><b>#{item.communication_plan_id}</b><span>{item.delivery_status || item.status}</span><small>{item.priority} · {label(item.intent)} · {item.history_count || 0} send record(s)</small></div>)}{!history.length && <p>No communications recorded for this monitoring session.</p>}</article></div>
  </section>
}
