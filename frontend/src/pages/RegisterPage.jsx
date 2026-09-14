import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ROLES } from '../config/demoAuth'

const API = 'http://127.0.0.1:8000'

function RegisterPage() {
  const navigate = useNavigate()
  const [role, setRole] = useState(ROLES.ORGANIZER)
  const [form, setForm] = useState({ full_name: '', email: '', password: '', confirm_password: '', organization: '' })
  const [errors, setErrors] = useState({})
  const [busy, setBusy] = useState(false)
  const [show, setShow] = useState({ password: false, confirm_password: false })
  const update = (name, value) => { setForm((old) => ({ ...old, [name]: value })); setErrors((old) => ({ ...old, [name]: '' })) }
  const submit = async (event) => {
    event.preventDefault()
    const next = {}
    if (!form.full_name.trim()) next.full_name = 'Please enter your full name.'
    if (!/^\S+@\S+\.\S+$/.test(form.email.trim())) next.email = 'Enter a valid email address.'
    if (form.password.length < 8 || !/[A-Z]/.test(form.password) || !/[a-z]/.test(form.password) || !/\d/.test(form.password)) next.password = 'Password must be at least 8 characters with uppercase, lowercase, and a number.'
    if (form.password !== form.confirm_password) next.confirm_password = 'Passwords do not match.'
    if (Object.keys(next).length) { setErrors(next); return }
    setBusy(true)
    try {
      const response = await fetch(API + '/auth/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ full_name: form.full_name.trim(), email: form.email.trim().toLowerCase(), password: form.password, role: role === ROLES.ORGANIZER ? 'ORGANIZER' : 'AUTHORITY', organization: form.organization.trim() }) })
      const data = await response.json()
      if (!response.ok) { setErrors({ form: data.detail || 'Unable to create the account.' }); return }
      navigate('/login', { replace: true, state: { registered: true } })
    } catch { setErrors({ form: 'Unable to connect to the backend server.' }) } finally { setBusy(false) }
  }
  return <main className="login-page"><section className="login-brand-panel"><div className="login-orbit orbit-one" /><div className="login-orbit orbit-two" /><div className="login-grid" /><div className="login-brand-content"><div className="login-logo-mark"><span /><span /><i /></div><div className="login-wordmark">CROWDGUARD<small>SAFETY INTELLIGENCE</small></div><div className="login-geometric-shape"><span /><span /><span /></div><p>AI-powered crowd monitoring<br />and decision support.</p><small className="login-status"><i /> SYSTEM READY / SECURE ACCESS</small></div></section><section className="login-panel-wrap"><div className="login-panel"><div className="login-heading"><span className="login-kicker">CROWDGUARD PORTAL</span><h1>Create Account</h1><p>Create your CrowdGuard account</p></div><div className="role-switch"><button type="button" className={role === ROLES.AUTHORITY ? 'active' : ''} onClick={() => setRole(ROLES.AUTHORITY)}>Authority</button><button type="button" className={role === ROLES.ORGANIZER ? 'active' : ''} onClick={() => setRole(ROLES.ORGANIZER)}>Event Organizer</button></div>{role === ROLES.AUTHORITY ? <div className="login-error" role="alert">Authority accounts are provisioned by the system administrator.<br /><Link to="/login">Back to Login</Link></div> : <form className="login-form" onSubmit={submit} noValidate><label htmlFor="register-name">Full Name</label><input id="register-name" type="text" autoComplete="name" placeholder="Enter your full name" value={form.full_name} onChange={(event) => update('full_name', event.target.value)} />{errors.full_name && <div className="field-error">{errors.full_name}</div>}<label htmlFor="register-email">Email</label><input id="register-email" type="email" autoComplete="email" placeholder="Enter your email" value={form.email} onChange={(event) => update('email', event.target.value)} />{errors.email && <div className="field-error">{errors.email}</div>}<label htmlFor="register-password">Password</label><div className="password-input"><input id="register-password" type={show.password ? 'text' : 'password'} autoComplete="new-password" placeholder="Create a password" value={form.password} onChange={(event) => update('password', event.target.value)} /><button type="button" onClick={() => setShow((old) => ({ ...old, password: !old.password }))}>{show.password ? '◉' : '◌'}</button></div>{errors.password && <div className="field-error">{errors.password}</div>}<label htmlFor="register-confirm">Confirm Password</label><div className="password-input"><input id="register-confirm" type={show.confirm_password ? 'text' : 'password'} autoComplete="new-password" placeholder="Confirm your password" value={form.confirm_password} onChange={(event) => update('confirm_password', event.target.value)} /><button type="button" onClick={() => setShow((old) => ({ ...old, confirm_password: !old.confirm_password }))}>{show.confirm_password ? '◉' : '◌'}</button></div>{errors.confirm_password && <div className="field-error">{errors.confirm_password}</div>}<label htmlFor="register-org">Organization / Event Name <small>(optional)</small></label><input id="register-org" type="text" placeholder="Enter organization name" value={form.organization} onChange={(event) => update('organization', event.target.value)} />{errors.form && <div className="login-error" role="alert">{errors.form}</div>}<button className="login-submit" type="submit" disabled={busy}>{busy ? 'Creating account…' : 'CREATE ACCOUNT'} <span>→</span></button></form>}<p className="login-create-link">Already have an account? <Link to="/login">Login</Link></p></div></section></main>
}

export default RegisterPage
