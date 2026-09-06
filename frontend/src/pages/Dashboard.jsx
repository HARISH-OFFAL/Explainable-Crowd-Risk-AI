import { useLocation, useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'

function Dashboard() {
  const navigate = useNavigate()
  const authority = useLocation().pathname.startsWith('/authority')
  const actions = authority
    ? [
        ['≡', 'Submitted Events', 'View registered events submitted for official document review.', 'View Events', '/authority/events'],
        ['✓', 'Document Review', 'Review uploaded documents and record an approved workflow status.', 'Open Reviews', '/authority/documents'],
      ]
    : [
        ['＋', 'Event Registration', 'Create an event with the details used by the existing risk engine.', 'Register Event', '/events/register'],
        ['≡', 'My Events', 'View details, edit your submission, or delete your own event.', 'Manage Events', '/events'],
        ['⇧', 'Upload Documents', 'Submit required event documents and respond to reviewer remarks.', 'Upload Documents', '/organizer/documents'],
        ['≡', 'Track Application Status', 'See document statuses, remarks, and the separate AI risk result.', 'Track Status', '/organizer/application-status'],
      ]
  return <div className="app"><div className="background-grid" /><div className="glow glow-one" /><Navbar /><main className="page-main">
    <section className="dashboard-hero-layout">
      <div className="dashboard-hero-copy"><div className="eyebrow">{authority ? 'AUTHORITY / REVIEWER WORKSPACE' : 'EVENT ORGANIZER WORKSPACE'}</div><h1>{authority ? <>Authority <span>Dashboard</span></> : <>Organizer <span>Dashboard</span></>}</h1><p>{authority ? 'Review submitted event documents and record official remarks. The system performs risk assessment separately.' : 'Register your event, submit documents, track official review, and view the existing pre-event assessment.'}</p><section className="action-grid">{actions.map(([icon, title, description, action, path]) => <article className="action-card glass-card" key={title}><span className="card-icon">{icon}</span><h3>{title}</h3><p>{description}</p><button onClick={() => navigate(path)}>{action} →</button></article>)}</section></div>
      <div className="radar-panel glass-card" aria-label="Crowd safety radar visual">
        <div className="radar-panel-heading"><span>SAFETY INTELLIGENCE</span><small>VISUAL PREVIEW</small></div>
        <div className="safety-radar">
          <div className="radar-grid-line horizontal" /><div className="radar-grid-line vertical" />
          <div className="radar-ring radar-ring-outer" /><div className="radar-ring radar-ring-middle" /><div className="radar-ring radar-ring-inner" />
          <div className="radar-sweep" />
          <i className="radar-point point-one" /><i className="radar-point point-two" /><i className="radar-point point-three" /><i className="radar-point point-four" /><i className="radar-point point-five" />
          <div className="radar-core"><span>CG</span></div>
        </div>
        <p className="radar-caption">Visual system preview · no live monitoring data</p>
      </div>
    </section>
  </main><footer>CrowdGuard · Planning and preparation workspace</footer></div>
}
export default Dashboard
