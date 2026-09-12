import { useState } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { PORTAL_SWITCH_PASSWORD, ROLES } from '../config/demoAuth'

function Navbar({ showLinks = true }) {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const authority = pathname.startsWith('/authority')
  const base = authority ? '/authority' : '/organizer'
  const phase3EventId = pathname.match(/^\/organizer\/events\/(\d+)/)?.[1]
  const { logout, switchRole } = useAuth()
  const [switchOpen, setSwitchOpen] = useState(false)
  const [switchPassword, setSwitchPassword] = useState('')
  const [switchError, setSwitchError] = useState('')
  const targetRole = authority ? ROLES.ORGANIZER : ROLES.AUTHORITY

  const confirmPortalSwitch = (event) => {
    event.preventDefault()
    if (switchPassword !== PORTAL_SWITCH_PASSWORD) {
      setSwitchError('Incorrect portal password.')
      return
    }
    switchRole(targetRole)
    setSwitchOpen(false)
    setSwitchPassword('')
    setSwitchError('')
    navigate(authority ? '/organizer/dashboard' : '/authority/dashboard', { replace: true })
  }

  return (
    <nav className="navbar">
      <div
        className="brand"
        onClick={() => navigate('/')}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            navigate('/')
          }
        }}
      >
        <div className="brand-mark">
          <span></span>
          <span></span>
          <span></span>
        </div>

        <div>
          <h2>CrowdGuard</h2>
          <p>Safety intelligence</p>
        </div>
      </div>

      {showLinks && <div className="nav-links">
        <NavLink to={authority ? '/authority/dashboard' : '/organizer/dashboard'}>
          {authority ? 'Authority Dashboard' : 'Organizer Dashboard'}
        </NavLink>

        {authority ? (
          <>
            <NavLink to={`${base}/events`}>Submitted Events</NavLink>
            <NavLink to={`${base}/documents`}>Document Review</NavLink>
          </>
        ) : (
          <>
            <NavLink to="/events/register">Register Event</NavLink>
            <NavLink to="/events">My Events</NavLink>
            <NavLink to={`${base}/documents`}>Upload Documents</NavLink>
            <NavLink to={`${base}/application-status`}>Track Status</NavLink>
            {phase3EventId && <NavLink to={`/organizer/events/${phase3EventId}/crowd-time-machine`}>Crowd Time Machine</NavLink>}
            {phase3EventId && <NavLink to={`/organizer/events/${phase3EventId}/response-commander`}>Response Commander</NavLink>}
          </>
        )}

        <button className="nav-switch" type="button" onClick={() => { setSwitchOpen(true); setSwitchError(''); setSwitchPassword('') }}>
          Switch Portal
        </button>
        <button className="nav-logout" type="button" onClick={() => { logout(); navigate('/login', { replace: true }) }}>Log out</button>
      </div>}
      {switchOpen && <div className="portal-switch-backdrop" role="presentation" onClick={() => setSwitchOpen(false)}><form className="portal-switch-modal" role="dialog" aria-modal="true" onSubmit={confirmPortalSwitch} onClick={(event) => event.stopPropagation()}><span className="login-kicker">PORTAL ACCESS</span><h2>Switch to {authority ? 'Event Organizer' : 'Authority'}</h2><p>Enter the portal switch password to continue.</p><label htmlFor="portal-switch-password">Password</label><input id="portal-switch-password" autoFocus type="password" value={switchPassword} onChange={(event) => { setSwitchPassword(event.target.value); setSwitchError('') }} placeholder="Enter password" />{switchError && <div className="login-error" role="alert">{switchError}</div>}<div className="portal-switch-actions"><button type="button" onClick={() => setSwitchOpen(false)}>Cancel</button><button type="submit">Continue</button></div></form></div>}
    </nav>
  )
}

export default Navbar
