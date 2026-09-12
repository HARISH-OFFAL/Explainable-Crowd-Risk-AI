import { useLocation, useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import Icon from '../components/Icon'

function Dashboard() {
  const navigate = useNavigate()
  const authority = useLocation().pathname.startsWith('/authority')
  const actions = authority
    ? [
        ['list', 'Submitted Events', 'View registered events submitted for official document review.', 'View Events', '/authority/events'],
        ['check', 'Document Review', 'Review uploaded documents and record official workflow remarks.', 'Open Reviews', '/authority/documents'],
      ]
    : [
        ['plus', 'Event Registration', 'Create an event with the details used by the existing risk engine.', 'Register Event', '/events/register'],
        ['list', 'My Events', 'View details, edit your submission, or delete your own event.', 'Manage Events', '/events'],
        ['upload', 'Upload Documents', 'Submit required event documents and respond to reviewer remarks.', 'Upload Documents', '/organizer/documents'],
        ['list', 'Track Application Status', 'See document statuses, remarks, and the separate AI risk result.', 'Track Status', '/organizer/application-status'],
      ]

  return <div className="app dashboard-app">
    <Navbar />
    <main className="dashboard-main">
      <div className="dashboard-topline"><span><i /> COMMAND CENTER</span><small>EXPLAINABLE CROWD RISK AI</small></div>
      <section className="dashboard-hero-layout">
        <div className="dashboard-hero-copy">
          <div className="eyebrow">{authority ? 'AUTHORITY / REVIEWER WORKSPACE' : 'EVENT ORGANIZER WORKSPACE'}</div>
          <h1>{authority ? <>Authority <span>Dashboard</span></> : <>Organizer <span>Dashboard</span></>}</h1>
          <p>{authority ? 'Review submitted event documents and record official remarks. The system performs risk assessment separately.' : 'Register your event, submit documents, track official review, and view the existing pre-event assessment.'}</p>
          <div className="dashboard-hero-actions"><button className="dashboard-primary" onClick={() => navigate(actions[0][4])}>{actions[0][3]} <span>↗</span></button><span className="dashboard-hint">{actions.length} operational modules available</span></div>
        </div>
        <div className="radar-panel glass-card" aria-label="Crowd safety radar visual">
          <div className="radar-panel-heading"><span>SAFETY INTELLIGENCE</span><small>CONTROL SURFACE / 01</small></div>
          <div className="safety-radar"><div className="radar-grid-line horizontal" /><div className="radar-grid-line vertical" /><div className="radar-ring radar-ring-outer" /><div className="radar-ring radar-ring-middle" /><div className="radar-ring radar-ring-inner" /><div className="radar-sweep" /><i className="radar-point point-one" /><i className="radar-point point-two" /><i className="radar-point point-three" /><i className="radar-point point-four" /><i className="radar-point point-five" /><div className="radar-core"><span>CG</span></div></div>
          <div className="radar-footer"><span><i /> SYSTEM READY</span><small>LIVE MODULES CONNECTED</small></div>
        </div>
      </section>
      <section className="dashboard-module-header"><div><span className="eyebrow">WORKSPACE MODULES</span><h2>Move from planning to action.</h2></div><p>Use the connected workflow to prepare, submit and monitor event safety operations.</p></section>
      <section className="action-grid">{actions.map(([icon, title, description, action, path], index) => <article className="action-card glass-card" key={title}><div className="action-card-top"><span className="card-index">0{index + 1}</span><span className="card-icon"><Icon name={icon} /></span></div><h3>{title}</h3><p>{description}</p><button onClick={() => navigate(path)}>{action} <span aria-hidden="true">→</span></button></article>)}</section>
    </main>
    <footer className="dashboard-footer">CrowdGuard <span>/</span> Planning and preparation workspace <span>/</span> {authority ? 'Authority' : 'Organizer'} mode</footer>
  </div>
}

export default Dashboard
