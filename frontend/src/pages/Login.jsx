import { useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { ROLE_LABELS, ROLES } from '../config/demoAuth'

function Login() {
  const { isAuthenticated, role: currentRole, login } = useAuth()
  const navigate = useNavigate()
  const [role, setRole] = useState(ROLES.ORGANIZER)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState(false)
  const [busy, setBusy] = useState(false)

  if (isAuthenticated) return <Navigate to={currentRole === ROLES.AUTHORITY ? '/authority/dashboard' : '/organizer/dashboard'} replace />

  const submit = (event) => {
    event.preventDefault()
    if (busy) return
    if (!email.trim()) return setError('Email is required.')
    if (!/^\S+@\S+\.\S+$/.test(email)) return setError('Enter a valid email address.')
    if (!password) return setError('Password is required.')
    setBusy(true)
    const result = login(email, password, role)
    if (!result.success) {
      setError(`Invalid email or password for ${ROLE_LABELS[role]}.`)
      setBusy(false)
      return
    }
    navigate(role === ROLES.AUTHORITY ? '/authority/dashboard' : '/organizer/dashboard', { replace: true })
  }

  const selectRole = (nextRole) => {
    setRole(nextRole)
    setError('')
  }

  return <main className="login-page">
    <section className="login-brand-panel">
      <div className="login-orbit orbit-one" /><div className="login-orbit orbit-two" /><div className="login-grid" />
      <div className="login-brand-content"><div className="login-logo-mark"><span /><span /><i /></div><div className="login-wordmark">CROWDGUARD<small>SAFETY INTELLIGENCE</small></div><div className="login-geometric-shape"><span /><span /><span /></div><p>AI-powered crowd monitoring<br />and decision support.</p><small className="login-status"><i /> SYSTEM READY / SECURE ACCESS</small></div>
    </section>
    <section className="login-panel-wrap"><div className="login-panel">
      <div className="login-heading"><span className="login-kicker">CROWDGUARD PORTAL</span><h1>Welcome</h1><p>Login to your CrowdGuard account</p></div>
      <div className="role-switch" aria-label="Choose login portal"><button type="button" className={role === ROLES.AUTHORITY ? 'active' : ''} onClick={() => selectRole(ROLES.AUTHORITY)}>Authority</button><button type="button" className={role === ROLES.ORGANIZER ? 'active' : ''} onClick={() => selectRole(ROLES.ORGANIZER)}>Event Organizer</button></div>
      <form className="login-form" onSubmit={submit} noValidate>
        <label htmlFor="login-email">Email</label><input id="login-email" name="email" type="email" autoComplete="email" placeholder="Enter your email" value={email} onChange={(event) => { setEmail(event.target.value); setError('') }} />
        <label htmlFor="login-password">Password</label><div className="password-input"><input id="login-password" name="password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" placeholder="Enter your password" value={password} onChange={(event) => { setPassword(event.target.value); setError('') }} /><button type="button" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword(!showPassword)}>{showPassword ? '◉' : '◌'}</button></div>
        <div className="login-options"><button type="button" onClick={() => setNotice(true)}>Forgot Password?</button><span>DEMO ACCESS</span></div>
        {error && <div className="login-error" role="alert">{error}</div>}
        <button className="login-submit" type="submit" disabled={busy}>{busy ? 'Signing in…' : 'LOGIN'} <span>→</span></button>
      </form>
      <p className="login-footer-note">CrowdGuard Safety Intelligence<br /><span>Decision support for safer events.</span></p>
    </div></section>
    {notice && <div className="login-modal-backdrop" role="presentation" onClick={() => setNotice(false)}><div className="login-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}><span className="login-kicker">DEMO ACCESS</span><h2>Predefined credentials</h2><p>Demo accounts use predefined credentials. Contact the project administrator for access.</p><button type="button" onClick={() => setNotice(false)}>Continue</button></div></div>}
  </main>
}

export default Login
