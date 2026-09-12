import { useEffect, useState } from 'react'

const API = 'http://127.0.0.1:8000'
export default function LiveCommandFeed({ plan }) {
  const [events, setEvents] = useState([]); const [filter, setFilter] = useState('ALL')
  useEffect(() => { let active = true; const load = () => fetch(`${API}/events/${plan.event_id}/response-commander/plans/${plan.id}/command-feed`).then((response) => response.json()).then((data) => { if (active) setEvents(data.events || []) }).catch(() => {}); load(); const timer = window.setInterval(load, 2000); return () => { active = false; window.clearInterval(timer) } }, [plan])
  const visible = events.filter((event) => filter === 'ALL' || (filter === 'WARNINGS' ? ['WARNING', 'CRITICAL'].includes(event.type) : filter === 'PREDICTIONS' ? event.type === 'PREDICTION' : filter === 'ACTIONS' ? ['SIMULATION', 'RECOMMENDATION', 'APPROVAL'].includes(event.type) : true))
  return <article className="rc-panel rc-command-feed"><div className="feed-heading"><div><span className="rc-kicker">LIVE AI COMMAND FEED</span><h2>Chronological decision-support activity</h2></div><div className="feed-filters">{['ALL', 'WARNINGS', 'PREDICTIONS', 'ACTIONS'].map((item) => <button key={item} className={filter === item ? 'active' : ''} onClick={() => setFilter(item)}>{item}</button>)}</div></div><div className="command-events">{visible.map((event, index) => <div className={`command-event ${event.type.toLowerCase()}`} key={`${event.type}-${event.title}-${index}`}><i /> <time>{Number(event.timestamp).toFixed(1)}s</time><b>{event.type}</b><strong>{event.title}</strong><p>{event.message}</p></div>)}</div>{!visible.length && <p className="command-empty">No matching state transition events.</p>}</article>
}
