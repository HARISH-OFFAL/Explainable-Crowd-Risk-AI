import { createContext, useContext, useMemo, useState } from 'react'
import { DEMO_CREDENTIALS, ROLES } from '../config/demoAuth'

const STORAGE_KEY = 'crowdguard_demo_session'
const AuthContext = createContext(null)

function readSession() {
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY)
    const session = stored ? JSON.parse(stored) : null
    return session?.isAuthenticated && Object.values(ROLES).includes(session.role) ? session : null
  } catch {
    return null
  }
}

export function AuthProvider({ children }) {
  const [session, setSession] = useState(readSession)

  const value = useMemo(() => ({
    isAuthenticated: Boolean(session?.isAuthenticated),
    role: session?.role || null,
    async login(email, password, role) {
      try {
        const response = await fetch('http://127.0.0.1:8000/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password, role }) })
        const data = await response.json()
        if (response.ok && data.user) {
          const next = { isAuthenticated: true, role: data.user.role, user: data.user, token: data.token }
          sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next)); setSession(next); return { success: true }
        }
        if (response.status !== 401) return { success: false, message: data.detail }
      } catch { /* Keep demo login available when the backend is offline. */ }
      const credential = DEMO_CREDENTIALS[role]
      if (!credential || email.trim().toLowerCase() !== credential.email || password !== credential.password) {
        return { success: false }
      }
      const next = { isAuthenticated: true, role }
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      setSession(next)
      return { success: true }
    },
    logout() {
      sessionStorage.removeItem(STORAGE_KEY)
      setSession(null)
    },
    switchRole(nextRole) {
      if (!session?.isAuthenticated || !Object.values(ROLES).includes(nextRole)) return
      const next = { isAuthenticated: true, role: nextRole }
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      setSession(next)
    },
  }), [session])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside AuthProvider')
  return context
}
