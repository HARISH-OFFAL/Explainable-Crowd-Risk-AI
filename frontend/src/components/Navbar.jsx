import { NavLink, useLocation, useNavigate } from 'react-router-dom'

function Navbar({ showLinks = true }) {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const authority = pathname.startsWith('/authority')
  const base = authority ? '/authority' : '/organizer'

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
          <p>SAFETY INTELLIGENCE</p>
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
          </>
        )}

        <button className="nav-switch" type="button" onClick={() => navigate(authority ? '/organizer/dashboard' : '/authority/dashboard')}>
          Switch Portal
        </button>
      </div>}
    </nav>
  )
}

export default Navbar
