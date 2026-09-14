import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import Navbar from '../components/Navbar'
import CommunicationCenter from '../components/CommunicationCenterV2'

const API = 'http://127.0.0.1:8000'

export default function CommunicationCenterPage() {
  const { eventId } = useParams(); const [monitoring, setMonitoring] = useState(null); const [flow, setFlow] = useState(null); const [error, setError] = useState('')
  useEffect(() => { let alive = true; fetch(`${API}/events/${eventId}/monitoring-sessions`).then((r) => r.json()).then((data) => { const item = (data.sessions || [])[0]; if (!item) throw new Error('No monitoring session available.'); if (alive) { setMonitoring(item); return fetch(`${API}/monitoring-sessions/${item.id}/flow-analysis`) } return null }).then((r) => r && r.json()).then((data) => alive && data && setFlow(data)).catch((reason) => alive && setError(reason.message)); return () => { alive = false } }, [eventId])
  return <div className="app"><Navbar /><main className="response-commander"><div className="rc-top-tabs"><a href={`/organizer/events/${eventId}/response-commander`}>RESPONSE COMMANDER</a><button className="active">COMMUNICATION CENTER</button></div>{error ? <div className="monitor-error">{error}</div> : <CommunicationCenter eventId={eventId} monitoringId={monitoring?.id} flowId={flow?.flow_analysis_id} />}</main></div>
}
