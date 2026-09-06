import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Navbar from '../components/Navbar'

const API_URL = 'http://127.0.0.1:8000'

const DOCUMENT_STATUSES = [
  'Approved',
  'Pending',
  'Under Review',
  'Rejected',
  'Additional Information Required',
]

function Phase1Summary() {
  const navigate = useNavigate()
  const { eventId } = useParams()
  const [event, setEvent] = useState(null)
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    const loadSummary = async () => {
      setLoading(true)
      setError('')

      try {
        const [eventResponse, documentsResponse] = await Promise.all([
          fetch(`${API_URL}/events/${eventId}`),
          fetch(`${API_URL}/events/${eventId}/documents`),
        ])

        const eventData = await eventResponse.json()
        const documentsData = await documentsResponse.json()

        if (!eventResponse.ok) {
          throw new Error(eventData.detail || 'Unable to load event details.')
        }

        if (!documentsResponse.ok) {
          throw new Error(
            documentsData.detail || 'Unable to load event documents.'
          )
        }

        if (active) {
          setEvent(eventData)
          setDocuments(
            Array.isArray(documentsData.documents)
              ? documentsData.documents
              : []
          )
        }
      } catch (requestError) {
        if (active) {
          setError(
            requestError instanceof TypeError
              ? 'Unable to connect to the backend server.'
              : requestError.message
          )
        }
      } finally {
        if (active) setLoading(false)
      }
    }

    loadSummary()
    return () => {
      active = false
    }
  }, [eventId])

  const statusCounts = useMemo(() => {
    const counts = Object.fromEntries(DOCUMENT_STATUSES.map((status) => [status, 0]))
    documents.forEach((document) => {
      if (Object.prototype.hasOwnProperty.call(counts, document.status)) {
        counts[document.status] += 1
      }
    })
    return counts
  }, [documents])

  const getStatusClass = (status) => {
    if (status === 'Approved') return 'document-status-approved'
    if (status === 'Rejected') return 'document-status-rejected'
    if (status === 'Under Review') return 'document-status-review'
    if (status === 'Additional Information Required') return 'document-status-info'
    return 'document-status-pending'
  }

  const openDocument = (documentId) => {
    window.open(`${API_URL}/documents/${documentId}/file`, '_blank', 'noopener,noreferrer')
  }

  if (loading) {
    return (
      <div className="app">
        <div className="background-grid" />
        <Navbar />
        <main className="empty-state"><div className="glass-card"><h2>Loading Phase 1 summary…</h2></div></main>
      </div>
    )
  }

  if (error || !event) {
    return (
      <div className="app">
        <div className="background-grid" />
        <Navbar />
        <main className="empty-state">
          <div className="glass-card">
            <h2>Phase 1 summary unavailable</h2>
            <p>{error || 'The selected event could not be found.'}</p>
            <button className="secondary-btn" onClick={() => navigate('/events')}>
              Back to Organizer Events
            </button>
          </div>
        </main>
      </div>
    )
  }

  const risk = event.pre_event_risk || {}
  const riskLevel = String(risk.risk_level || '').toLowerCase()

  const eventDetails = [
    ['Event Name', event.event_name],
    ['Location', event.location],
    ['Date and Time', event.event_datetime ? new Date(event.event_datetime).toLocaleString() : '—'],
    ['Expected Crowd Size', event.expected_crowd_size?.toLocaleString()],
    ['Venue Capacity', event.venue_capacity?.toLocaleString()],
    ['Entry Gates', event.entry_gates],
    ['Exit Gates', event.exit_gates],
    ['Emergency Exits', event.emergency_exits],
    ['Event Duration', event.event_duration_minutes ? `${event.event_duration_minutes} minutes` : '—'],
  ]

  return (
    <div className="app phase1-summary-page">
      <div className="background-grid" />
      <div className="glow glow-one" />
      <Navbar />

      <main className="page-main">
        <div className="phase1-summary-header">
          <div>
            <div className="eyebrow">PHASE 1 SAFETY SUMMARY</div>
            <h1>{event.event_name}<span> · Event #{event.id}</span></h1>
            <p>Read-only view of the registration, document review, risk assessment, and preparation recommendations for this event.</p>
          </div>
          <div className="phase1-summary-actions">
            <button className="secondary-btn" onClick={() => navigate('/events')}>← Back to Organizer Events</button>
            <button className="secondary-btn" onClick={() => navigate('/organizer/dashboard')}>Organizer Dashboard</button>
          </div>
        </div>

        <section className="phase1-flow glass-card" aria-label="Phase 1 workflow">
          {[
            'Event Registration',
            'Document / Permission Management',
            'Authorized Official Review',
            'Pre-Event Risk Assessment',
            'Safety Preparation Recommendations',
          ].map((stage, index) => (
            <div className="phase1-flow-stage" key={stage}>
              <span>{String(index + 1).padStart(2, '0')}</span>
              <strong>{stage}</strong>
              {index < 4 && <b>→</b>}
            </div>
          ))}
        </section>

        <section className="phase1-section">
          <div className="phase1-section-heading"><span>01</span><div><h2>Event Details</h2><p>Information submitted for the event registration.</p></div></div>
          <div className="phase1-detail-grid">
            {eventDetails.map(([label, value]) => <div className="phase1-detail-card glass-card" key={label}><span>{label}</span><strong>{value ?? '—'}</strong></div>)}
          </div>
        </section>

        <section className="phase1-section">
          <div className="phase1-section-heading"><span>02</span><div><h2>Document / Permission Management</h2><p>Uploaded documents and their current official review information.</p></div></div>
          <div className="phase1-info-note">The system supports the official review process; it does not independently grant legal permission.</div>
          {documents.length === 0 ? <div className="glass-card phase1-empty">No documents have been uploaded for this event.</div> : <div className="phase1-document-list">{documents.map((document) => <article className="phase1-document-card glass-card" key={document.id}><div><span className="document-type">{document.document_type}</span><h3>{document.document_name}</h3><p>Document ID #{document.id}</p></div><span className={`document-status ${getStatusClass(document.status)}`}>{document.status || 'Pending'}</span><div className="phase1-document-remarks"><span>AUTHORITY REMARKS</span><p>{document.remarks || 'No remarks provided yet.'}</p></div><button className="small-btn" onClick={() => openDocument(document.id)}>View Document</button></article>)}</div>}
        </section>

        <section className="phase1-section">
          <div className="phase1-section-heading"><span>03</span><div><h2>Authorized Official Review</h2><p>Read-only summary based on the document statuses returned by the system.</p></div></div>
          <div className="phase1-review-grid">{DOCUMENT_STATUSES.map((status) => <div className="phase1-review-card glass-card" key={status}><span className={`document-status ${getStatusClass(status)}`}>{status}</span><strong>{statusCounts[status]}</strong><small>documents</small></div>)}</div>
        </section>

        <section className="phase1-section">
          <div className="phase1-section-heading"><span>04</span><div><h2>Pre-Event Risk Assessment</h2><p>Assessment returned by the existing backend risk engine.</p></div></div>
          <div className="phase1-risk-layout"><article className={`phase1-risk-level glass-card phase1-risk-${riskLevel}`}><span>RISK LEVEL</span><strong>{risk.risk_level || '—'}</strong><small>Score {risk.risk_score ?? '—'}</small></article><article className="phase1-content-card glass-card"><h3>Reasons</h3><ul className="result-list">{risk.reasons?.length ? risk.reasons.map((reason, index) => <li key={index}>{reason}</li>) : <li>No reasons returned by the backend.</li>}</ul></article></div>
        </section>

        <section className="phase1-section">
          <div className="phase1-section-heading"><span>05</span><div><h2>Safety Preparation Recommendations</h2><p>Preparation guidance returned by the existing backend assessment.</p></div></div>
          <article className="phase1-content-card glass-card"><ul className="result-list">{risk.recommendations?.length ? risk.recommendations.map((recommendation, index) => <li key={index}>{recommendation}</li>) : <li>No recommendations returned by the backend.</li>}</ul></article>
        </section>
      </main>
      <footer>CrowdGuard · Phase 1 safety summary</footer>
    </div>
  )
}

export default Phase1Summary
