import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'

function Home() {
  const navigate = useNavigate()
  return <div className="app"><div className="background-grid" /><div className="glow glow-one" /><div className="glow glow-two" /><Navbar showLinks={false} />
    <main>
      <section className="hero">
        <div>
          <div className="eyebrow"><span className="pulse-dot" />EXPLAINABLE SPATIO-TEMPORAL AI</div>
          <h1>Predict risk.<br /><span>Protect crowds.</span></h1>
          <p>Plan safer events with explainable crowd-risk assessments and practical safety recommendations before the gathering begins.</p>
          <div className="hero-actions"><button className="primary-btn" onClick={() => navigate('/organizer/dashboard')}>Event Organizer Portal →</button><button className="secondary-btn" onClick={() => navigate('/authority/dashboard')}>Authority / Reviewer Portal</button></div>
        </div>
        <div className="hero-visual glass-card"><div className="visual-copy"><small>EARLY RISK DECISION SUPPORT</small><h2>Make every event easier to prepare for.</h2><p>Enter event conditions, receive a risk level, understand why it was assigned, and act on the recommendations.</p></div></div>
      </section>
      <section className="page-main" style={{ paddingTop: 0 }}><div className="feature-grid"><article className="feature-card glass-card"><h3>Event Organizer</h3><p>Register an event, upload required documents, and track review remarks.</p><button className="text-btn" onClick={() => navigate('/organizer/dashboard')}>Open organizer portal →</button></article><article className="feature-card glass-card"><h3>Authority / Reviewer</h3><p>Review submitted events and update document statuses with remarks.</p><button className="text-btn" onClick={() => navigate('/authority/dashboard')}>Open authority portal →</button></article><article className="feature-card glass-card"><h3>System Assessment</h3><p>The existing system produces explainable pre-event risk results and recommendations.</p></article></div></section>
    </main><footer>CrowdGuard · Explainable Crowd Safety Decision Support</footer>
  </div>
}
export default Home
